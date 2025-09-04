#!/usr/bin/env python3
r"""
build_cea_uv_decks.py  — Starch (C6H10O5) only

Input CSV (case-insensitive headers; any ONE of these for mass is fine):
  - Run
  - Mass Starch   OR   Mass PP   [also accepts Mass Fuel for convenience]
  - Po            (initial absolute pressure, MPa)

Behavior:
  - Computes n_O2 from Po (ideal gas, 298.15 K) and bomb volume.
  - Converts Mass Starch (g) -> n_starch (mol) using MW_STARCH.
  - Writes UV decks with ONLY one fuel (starch) + O2 oxidizer.
  - Uses a user-defined composition line for starch with ΔHf° at 298 K.

Manifest columns (examples):
  run_label, deck_file, pressure_mpa_abs, mass_starch_g, n_starch_mol, n_o2_mol,
  total_mass_kg, v_spec_m3_per_kg
"""

from __future__ import annotations

import argparse
import zipfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ---- constants (starch) ----
R_J_PER_MOLK = 8.314462618         # J/mol/K
MW_O2 = 31.998                      # g/mol
MW_STARCH = 162.1406                # g/mol   (C6H10O5 empirical repeat unit)
# Your prior default for starch surrogate ΔHf° at 298 K:
STARCH_DHF_KJ_PER_MOL = -960.0      # kJ/mol (per C6H10O5 unit)

@dataclass(frozen=True)
class BuildConfig:
    csv_path: Path
    project_root: Path
    chamber_volume_m3: float = 240e-6   # 240 mL free volume
    start_temperature_k: float = 298.15
    starch_dhf_kj_per_mol: float = STARCH_DHF_KJ_PER_MOL

def prepare_structure(root: Path) -> Dict[str, Path]:
    """Create/clean core subfolders under the project root."""
    inputs = root / "input_decks"
    manifests = root / "manifests"
    docs = root / "docs"
    for d in (inputs, manifests, docs):
        d.mkdir(parents=True, exist_ok=True)
        # "generally clean": remove existing files/dirs inside these subfolders
        for p in d.iterdir():
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                pass
    return {"inputs": inputs, "manifests": manifests, "docs": docs}

def parse_run_label(run_value: object) -> str:
    """Coerce CSV 'Run' to a 4-digit label (e.g., 2115 -> '2115')."""
    if pd.isna(run_value):
        raise ValueError("Run is NaN/empty.")
    s = str(run_value).strip()
    try:
        run_int = int(float(s))  # handles '2115', '2115.0'
    except ValueError:
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            raise ValueError(f"Run value '{s}' is not numeric.")
        run_int = int(digits)
    if run_int < 0:
        raise ValueError(f"Run value must be non-negative: {run_int}")
    return f"{run_int:04d}"

def _get_mass_starch_ci(row: pd.Series) -> float:
    """Get starch mass (grams) from any accepted header."""
    # Accept these common labels, case-insensitive:
    candidates = ["mass starch", "mass pp", "mass fuel"]
    lower_map = {c.strip().lower(): c for c in row.index}
    for key in candidates:
        if key in lower_map:
            return float(row[lower_map[key]])
    raise KeyError(f"Missing starch mass column (accepted: {candidates}). Found: {list(row.index)}")

def compute_row(vals: Dict[str, str], cfg: BuildConfig) -> Dict[str, float]:
    """Compute n_O2, n_starch, total mass, and specific volume for starch + O2."""
    # case-insensitive fetch
    def get_ci(key_want: str) -> float:
        for k, v in vals.items():
            if k.strip().lower() == key_want:
                return float(v)
        raise KeyError(f"Missing column '{key_want}'")

    p_mpa_abs = get_ci("po")                      # MPa (absolute)
    m_starch_g = get_ci("mass starch")            # grams (provided by normalize step below)

    # Ideal gas oxygen at 298.15 K in the bomb volume
    p_pa = p_mpa_abs * 1e6
    V = cfg.chamber_volume_m3
    T0 = cfg.start_temperature_k
    n_o2 = (p_pa * V) / (R_J_PER_MOLK * T0)

    # Starch moles (per C6H10O5 unit)
    n_starch = m_starch_g / MW_STARCH

    # Total mass (gas O2 + solid starch), kg
    m_o2_g = n_o2 * MW_O2
    m_total_kg = (m_o2_g + m_starch_g) / 1000.0
    if m_total_kg <= 0:
        raise ValueError("Total mass computed non-positive; check inputs.")

    # Specific volume (m^3/kg)
    v_spec = V / m_total_kg

    return {
        "pressure_mpa_abs": p_mpa_abs,
        "mass_starch_g": m_starch_g,
        "n_starch_mol": n_starch,
        "n_o2_mol": n_o2,
        "total_mass_kg": m_total_kg,
        "v_spec_m3_per_kg": v_spec,
    }

def render_inp(run_label: str, d: Dict[str, float], cfg: BuildConfig) -> str:
    """Render a UV input deck for starch + O2 (no temperature guess on problem line)."""
    return "\n".join([
        f"! Run: {run_label}",
        "problem",
        f"    uv   v,m**3/kg={d['v_spec_m3_per_kg']:.6f}",
        "react",
        # User-defined starch fuel with composition & ΔHf°
        f"  fuel=starch  moles={d['n_starch_mol']:.7f}  t,k={cfg.start_temperature_k:.2f}",
        f"    h,kj/mol={cfg.starch_dhf_kj_per_mol:.1f}  C 6 H 10 O 5",
        f"  oxid=O2      moles={d['n_o2_mol']:.5f}  t,k={cfg.start_temperature_k:.2f}",
        "output",
        "    plot t p h s g cp",
        "end",
        ""
    ])

def write_readme(docs_dir: Path, cfg: BuildConfig, source_csv: Path) -> None:
    readme = f"""# CEA UV Project — Starch (C6H10O5)

This folder was generated automatically.

## Structure
- `input_decks/` : NASA CEA `.inp` files (one per run).
- `manifests/`   : CSV manifest summarizing computed values per run.
- `docs/`        : This README and any future documentation.

## Inputs
- Source CSV: `{source_csv.name}`
- Assumptions:
  - Bomb free volume: {cfg.chamber_volume_m3:.6e} m³ (240 mL)
  - Reactant start temperature: {cfg.start_temperature_k:.2f} K
  - Starch surrogate ΔHf°(298 K, per C6H10O5): {cfg.starch_dhf_kj_per_mol:.1f} kJ/mol
  - Fuel formula: C6H10O5; MW = {MW_STARCH:.4f} g/mol

## Units
- `Po` is absolute pressure in **MPa**.
- `Mass Starch` (or `Mass PP`) is in **grams**.
- Specific volume written to decks is in **m³/kg**.

## Notes
- The `problem` line does **not** include a temperature guess (UV problem).
"""
    (docs_dir / "README.md").write_text(readme, encoding="utf-8")

def zip_inputs(inputs_dir: Path, dest_zip: Path) -> None:
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(inputs_dir.glob("*.inp")):
            zf.write(p, arcname=p.name)

def main() -> None:
    ap = argparse.ArgumentParser(description="Build NASA CEA UV input decks for starch (C6H10O5) + O2.")
    ap.add_argument("csv", type=Path, help="Input CSV with columns: Run, Mass Starch (or Mass PP), Po (MPa abs)")
    ap.add_argument("project_root", type=Path, help="Output project directory")
    ap.add_argument("--volume-ml", type=float, default=240.0, help="Bomb free volume in mL (default 240)")
    ap.add_argument("--t0-k", type=float, default=298.15, help="Initial temperature, K (default 298.15)")
    ap.add_argument("--starch-dhf-kjmol", type=float, default=STARCH_DHF_KJ_PER_MOL,
                    help="ΔHf°,298 of starch surrogate per C6H10O5 (kJ/mol), default -960")
    args = ap.parse_args()

    cfg = BuildConfig(
        csv_path=args.csv,
        project_root=args.project_root,
        chamber_volume_m3=args.volume_ml * 1e-6,
        start_temperature_k=args.t0_k,
        starch_dhf_kj_per_mol=args.starch_dhf_kjmol,
    )

    # Prepare structure
    paths = prepare_structure(cfg.project_root)
    inputs_dir = paths["inputs"]
    manifests_dir = paths["manifests"]
    docs_dir = paths["docs"]

    # Load CSV (case-insensitive header handling)
    df = pd.read_csv(cfg.csv_path)

    # Build decks & manifest
    rows: List[Dict[str, float]] = []
    for _, r in df.iterrows():
        # Run label
        run_col = next((c for c in r.index if c.strip().lower() == "run"), None)
        if run_col is None:
            raise ValueError("Row missing 'Run' value.")
        run_label = parse_run_label(r[run_col])

        # Normalize a dict with lowercased keys for compute_row
        # Also map the mass column (Mass Starch / Mass PP / Mass Fuel) to a canonical 'mass starch'
        r_ci = {c.strip().lower(): r[c] for c in r.index}
        # inject canonical key for compute_row
        try:
            m_starch_g = _get_mass_starch_ci(r)
        except KeyError as e:
            raise
        r_ci["mass starch"] = m_starch_g

        vals = compute_row(r_ci, cfg)
        deck_text = render_inp(run_label, vals, cfg)

        fname = f"{run_label}.inp"
        (inputs_dir / fname).write_text(deck_text, encoding="utf-8")

        rows.append({
            "run_label": run_label,
            "deck_file": str((inputs_dir / fname).resolve()),
            **vals
        })

    # Write manifest
    manifest = pd.DataFrame(rows)
    manifest_path = manifests_dir / "input_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # README + zip
    write_readme(docs_dir, cfg, cfg.csv_path.resolve())
    zip_inputs(inputs_dir, cfg.project_root / "input_decks.zip")

    print(f"[OK] Wrote {len(rows)} decks to: {inputs_dir}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Zip: {cfg.project_root / 'input_decks.zip'}")

if __name__ == "__main__":
    main()
