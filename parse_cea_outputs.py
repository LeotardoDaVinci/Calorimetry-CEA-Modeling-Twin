#!/usr/bin/env python3
r"""
parse_cea_outputs.py — single-fuel projects (fuel-agnostic)

- No Boron/PP/EG specifics; everything is "fuel" + O2
- ASCII-only units; CSV written with UTF-8 BOM for Excel
- Robust CP and GAMMA regex (case-insensitive, tolerant)
- Inputs gathered from manifest when available; otherwise parsed from .inp
- Automatically detects legacy manifest column names for fuel by pattern
- Can infer fuel molecular weight from a composition line (e.g., "C 6 H 10 O 5")

CSV columns (subset):
  run_label, P0_MPa, v_spec_m3_per_kg,
  n_fuel_mol, n_O2_mol,
  mass_fuel_g, mass_O2_g,
  total_mass_kg,
  x0_fuel, x0_O2, w0_fuel, w0_O2,
  P_MPa, delta_P_MPa,
  T_K, RHO_kg_m3, H_kJ_kg, U_kJ_kg, G_kJ_kg, S_kJ_kgK, CP_kJ_kgK, GAMMA,
  <species columns...>
"""

from __future__ import annotations
import argparse, json, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------- Config ----------------
@dataclass(frozen=True)
class ParseConfig:
    project_root: Path
    manifest_path: Optional[Path] = None
    workers: int = 4
    fail_on_missing_listing: bool = False

R_J_PER_MOLK = 8.314462618  # J/mol/K
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"

# Atomic / molecular weights (g/mol) used for mass inference
AW = {
    "H": 1.00794,
    "C": 12.011,
    "N": 14.0067,
    "O": 15.999,
    "B": 10.81,
    "S": 32.065,
}
MW_O2 = 31.998

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------- Small utils ----------------
def canonical_label(s: str) -> str:
    return str(s).strip()

def first_float_after(pattern: str, text: str, flags=re.M | re.I) -> Optional[float]:
    m = re.search(pattern, text, flags)
    if not m:
        return None
    tail = text[m.end():].splitlines()[0] if "\n" in text[m.end():] else text[m.end():]
    f = re.search(FLOAT, tail)
    return float(f.group(0)) if f else None

def grab_scalar(patterns: List[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.M | re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                v = first_float_after(pat, text)
                if v is not None:
                    return v
    return None

# ---------------- Composition parsing ----------------
def _infer_fuel_mw_from_composition(txt: str) -> Optional[float]:
    """
    Look for a line with enthalpy + element counts (typical user-defined fuel):
      e.g., "h,kj/mol=...  C 6 H 10 O 5"
    Parse element counts and compute MW.
    """
    # Grab the element block following an h,kj/mol token on that same line
    m = re.search(
        r"h\s*,\s*kj/mol\s*=\s*" + FLOAT + r"(?:[^A-Za-z]|$)(?P<blk>(?:\s*[A-Z][a-z]?\s*" + FLOAT + r"\s*){1,12})",
        txt, flags=re.I
    )
    if not m:
        # Fallback: search any short element block anywhere (fuel decks are small)
        m = re.search(r"(?P<blk>(?:\s*[A-Z][a-z]?\s*" + FLOAT + r"\s*){2,12})", txt)
        if not m:
            return None
    blk = m.group("blk")

    elems: Dict[str, float] = {}
    for em in re.finditer(r"([A-Z][a-z]?)\s*(" + FLOAT + r")", blk):
        el = em.group(1)
        n = float(em.group(2))
        if el in AW:
            elems[el] = elems.get(el, 0.0) + n

    if not elems:
        return None
    mw = sum(AW[el] * count for el, count in elems.items() if el in AW)
    return mw if mw > 0 else None

# ---------------- MOLE FRACTIONS ----------------
def parse_mole_frac_block(text: str) -> Dict[str, float]:
    start = re.search(r"^\s*MOLE\s+FRACTIONS\s*$", text, flags=re.M | re.I)
    if not start:
        return {}

    # end at PRODUCTS... or THERMODYNAMIC/TRANSPORT/blank
    end = None
    for ep in [
        r"^\s*PRODUCTS\s+WHICH\s+WERE\s+CONSIDERED.*$",
        r"^\s*THERMODYNAMIC",
        r"^\s*TRANSPORT",
        r"^\s*$",
    ]:
        m = re.search(ep, text[start.end():], flags=re.M | re.I)
        if m:
            end = start.end() + m.start()
            break
    if end is None:
        end = len(text)

    block = text[start.end():end]

    species: Dict[str, float] = {}
    for m in re.finditer(r"\*?([A-Za-z0-9][A-Za-z0-9\-\+\(\)]*)\s+(" + FLOAT + r")", block):
        name = m.group(1)
        val = float(m.group(2))
        # Ignore stray header tokens
        if name.upper() == "TO":
            continue
        species[name] = val
    return species

# ---------------- Manifest / INP ----------------
def load_manifest(cfg: ParseConfig) -> pd.DataFrame:
    """
    Load <project_root>/manifests/input_manifest.csv (or --manifest).
    Expects: run_label, deck_file, P0 (or pressure_mpa_abs), v_spec_m3_per_kg,
             fuel + O2 amounts/masses (generic naming preferred).
    """
    path = cfg.manifest_path or (cfg.project_root / "manifests" / "input_manifest.csv")
    if not path.exists():
        logging.warning(f"Manifest not found at {path}. Will rely on .inp parsing only.")
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Ensure run_label exists
    lower_cols = {c.strip().lower(): c for c in df.columns}
    if "run_label" not in lower_cols:
        labels = []
        for _, row in df.iterrows():
            label = None
            if "run" in lower_cols:
                label = row[lower_cols["run"]]
            elif "deck_file" in lower_cols:
                try:
                    label = Path(row[lower_cols["deck_file"]]).stem
                except Exception:
                    label = None
            labels.append(canonical_label(label) if label is not None else "")
        df["run_label"] = labels
    elif lower_cols["run_label"] != "run_label":
        df = df.rename(columns={lower_cols["run_label"]: "run_label"})

    return df

def _pick_key_by_regex(d: Dict[str, object], pattern: str, exclude_substrings: Tuple[str, ...] = ()) -> Optional[str]:
    keys = list(d.keys())
    # direct preferred generic name first
    preferred = [k for k in keys if re.fullmatch(pattern, k, flags=re.I)]
    if len(preferred) == 1:
        return preferred[0]
    # broader search
    cands = [k for k in keys if re.match(pattern, k, flags=re.I)]
    cands = [k for k in cands if not any(ex.lower() in k.lower() for ex in exclude_substrings)]
    if not cands:
        return None
    # deterministic choice: sort by key name
    return sorted(cands, key=str.lower)[0]

def parse_inp_for_inputs(inp_path: Path) -> Dict[str, float]:
    """
    Parse a single .inp to recover:
      v_spec_m3_per_kg, n_fuel_mol, n_O2_mol, masses, total_mass_kg, P0_MPa (ideal gas from O2)
    Treat the first 'fuel=' line as the fuel. If possible, infer fuel MW from a composition line.
    """
    txt = inp_path.read_text(errors="ignore")
    vals: Dict[str, float] = {}

    # specific volume
    m = re.search(r"v,\s*m\*\*3/kg\s*=\s*(" + FLOAT + r")", txt, flags=re.I)
    if m:
        vals["v_spec_m3_per_kg"] = float(m.group(1))

    # reactants (first 'fuel=' is the fuel; any 'oxid=O2' captured as oxidizer)
    for kind in ("fuel", "oxid"):
        for rm in re.finditer(
            rf"^\s*{kind}\s*=\s*([^\s]+).*?moles\s*=\s*(" + FLOAT + r")(?:(?:.*?t,k\s*=\s*(" + FLOAT + r")))?",
            txt, flags=re.M | re.I
        ):
            species = rm.group(1).strip()
            moles = float(rm.group(2))
            t_k = float(rm.group(3)) if rm.lastindex and rm.group(3) else None

            if kind.lower() == "oxid" and species.upper().startswith("O2"):
                vals["n_O2_mol"] = moles
                if t_k is not None:
                    vals["t_O2_K"] = t_k
            elif kind.lower() == "fuel" and "n_fuel_mol" not in vals:
                vals["n_fuel_mol"] = moles
                if t_k is not None:
                    vals["t_fuel_K"] = t_k

    # Masses if moles known
    if "n_O2_mol" in vals:
        vals["mass_O2_g"] = vals["n_O2_mol"] * MW_O2

    # Try to infer fuel MW from composition (e.g., "C 6 H 10 O 5") if we have fuel moles
    if "n_fuel_mol" in vals:
        mw_fuel = _infer_fuel_mw_from_composition(txt)
        if mw_fuel:
            vals["mass_fuel_g"] = vals["n_fuel_mol"] * mw_fuel

    # Total mass (gas + fuel), kg
    tot_g = sum(vals.get(k, 0.0) for k in ("mass_fuel_g", "mass_O2_g"))
    if tot_g > 0:
        vals["total_mass_kg"] = tot_g / 1000.0

    # Default temperatures if absent
    vals.setdefault("t_O2_K", 298.15)

    # Back-compute initial absolute pressure from ideal gas (O2) and specific volume
    if all(k in vals for k in ("n_O2_mol", "v_spec_m3_per_kg", "total_mass_kg", "t_O2_K")):
        V = vals["v_spec_m3_per_kg"] * vals["total_mass_kg"]
        p_pa = vals["n_O2_mol"] * R_J_PER_MOLK * vals["t_O2_K"] / V
        vals["P0_MPa"] = p_pa / 1e6

    return vals

def inputs_from_manifest_or_inp(run_label: str, run_dir: Path, manifest_df: pd.DataFrame) -> Dict[str, float]:
    inputs: Dict[str, float] = {}
    man = None
    if not manifest_df.empty:
        hits = manifest_df[manifest_df["run_label"].astype(str) == run_label]
        if hits.empty and "deck_file" in manifest_df.columns:
            stems = manifest_df["deck_file"].map(lambda p: Path(str(p)).stem if isinstance(p, str) else "")
            hits = manifest_df[stems == run_label]
        man = hits.iloc[0].to_dict() if not hits.empty else None

    if man:
        # case-insensitive key view
        man_ci = {k.strip(): v for k, v in man.items() if k is not None}

        def get_direct(*names: str) -> Optional[float]:
            for nm in names:
                if nm in man_ci and pd.notna(man_ci[nm]):
                    try:
                        return float(man_ci[nm])
                    except Exception:
                        pass
            return None

        # direct known names (generic)
        p0 = get_direct("P0_MPa", "Po", "po", "pressure_mpa_abs")
        vsp = get_direct("v_spec_m3_per_kg", "v_spec", "v,m**3/kg")
        n_o2 = get_direct("n_O2_mol", "n_o2_mol")
        n_fuel = get_direct("n_fuel_mol")

        # if generic names missing, detect by pattern (avoid hardcoding specific fuels)
        if n_fuel is None:
            key = _pick_key_by_regex(man_ci, r"n_[a-z0-9]+_mol", exclude_substrings=("o2",))
            if key:
                try: n_fuel = float(man_ci[key])
                except Exception: pass

        mass_o2 = get_direct("mass_O2_g", "mass_o2_g")
        mass_fuel = get_direct("mass_fuel_g")

        if mass_fuel is None:
            key = _pick_key_by_regex(man_ci, r"mass_[a-z0-9]+_g", exclude_substrings=("o2",))
            if key:
                try: mass_fuel = float(man_ci[key])
                except Exception: pass

        tot_mass = get_direct("total_mass_kg")

        # compose inputs
        if p0 is not None: inputs["P0_MPa"] = p0
        if vsp is not None: inputs["v_spec_m3_per_kg"] = vsp
        if n_o2 is not None: inputs["n_O2_mol"] = n_o2
        if n_fuel is not None: inputs["n_fuel_mol"] = n_fuel
        if mass_o2 is not None: inputs["mass_O2_g"] = mass_o2
        if mass_fuel is not None: inputs["mass_fuel_g"] = mass_fuel
        if tot_mass is not None: inputs["total_mass_kg"] = tot_mass

        # derive missing O2 mass from moles
        if "mass_O2_g" not in inputs and "n_O2_mol" in inputs:
            inputs["mass_O2_g"] = inputs["n_O2_mol"] * MW_O2

    # Fill missing by parsing .inp
    if any(k not in inputs for k in ("v_spec_m3_per_kg", "P0_MPa", "n_fuel_mol", "n_O2_mol")):
        inp = run_dir / f"{run_label}.inp"
        if inp.exists():
            parsed = parse_inp_for_inputs(inp)
            for k, v in parsed.items():
                inputs.setdefault(k, v)

    # Initial mole and mass fractions (fuel/O2 only)
    n_tot = sum(inputs.get(k, 0.0) for k in ("n_fuel_mol", "n_O2_mol"))
    if n_tot > 0:
        inputs["x0_fuel"] = (inputs.get("n_fuel_mol", 0.0) / n_tot) or None
        inputs["x0_O2"]   = (inputs.get("n_O2_mol",   0.0) / n_tot) or None

    m_tot_g = sum(inputs.get(k, 0.0) for k in ("mass_fuel_g", "mass_O2_g"))
    if m_tot_g > 0:
        inputs["w0_fuel"] = (inputs.get("mass_fuel_g", 0.0) / m_tot_g) or None
        inputs["w0_O2"]   = (inputs.get("mass_O2_g",   0.0) / m_tot_g) or None

    return inputs

# ---------------- Listing parser ----------------
def parse_listing(listing_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    txt = listing_path.read_text(errors="ignore")

    P_bar = grab_scalar([
        r"^\s*P,\s*BAR\s*=\s*(" + FLOAT + r")",
        r"^\s*P,\s*BAR\s+(" + FLOAT + r")",
    ], txt)
    T_K = grab_scalar([
        r"^\s*T,\s*K\s*=\s*(" + FLOAT + r")",
        r"^\s*T,\s*K\s+(" + FLOAT + r")",
    ], txt)
    RHO = grab_scalar([
        r"^\s*RHO,\s*KG/CU\s*M\s*=\s*(" + FLOAT + r")",
        r"^\s*RHO,\s*KG/CU\s*M\s+(" + FLOAT + r")",
        r"^\s*RHO,.*?(" + FLOAT + r")",
    ], txt)
    H = grab_scalar([
        r"^\s*H,\s*KJ/KG\s*=\s*(" + FLOAT + r")",
        r"^\s*H,\s*KJ/KG\s+(" + FLOAT + r")",
    ], txt)
    U = grab_scalar([
        r"^\s*U,\s*KJ/KG\s*=\s*(" + FLOAT + r")",
        r"^\s*U,\s*KJ/KG\s+(" + FLOAT + r")",
    ], txt)
    G = grab_scalar([
        r"^\s*G,\s*KJ/KG\s*=\s*(" + FLOAT + r")",
        r"^\s*G,\s*KJ/KG\s+(" + FLOAT + r")",
    ], txt)
    S = grab_scalar([
        r"^\s*S,\s*KJ/\(KG\)\(K\)\s*=\s*(" + FLOAT + r")",
        r"^\s*S,\s*KJ/\(KG\)\(K\)\s+(" + FLOAT + r")",
        r"^\s*S,\s*KJ/ ?KG/ ?K\s*=\s*(" + FLOAT + r")",
    ], txt)
    # Robust CP & GAMMA (case-insensitive, accept ':' or '=')
    CP = grab_scalar([
        r"^\s*C[Pp]\s*,\s*KJ/\(KG\)\(K\)\s*[=:]\s*(" + FLOAT + r")",
        r"^\s*C[Pp]\s*,\s*KJ/ ?KG/ ?K\s*[=:]\s*(" + FLOAT + r")",
    ], txt)
    GAMMA = grab_scalar([
        r"^\s*GAMMA\S*\s*[=:]\s*(" + FLOAT + r")",
    ], txt)

    scalars: Dict[str, float] = {}
    if P_bar is not None:
        scalars["P_MPa"] = P_bar * 0.1
    scalars.update({
        "T_K": T_K,
        "RHO_kg_m3": RHO,
        "H_kJ_kg": H,
        "U_kJ_kg": U,
        "G_kJ_kg": G,
        "S_kJ_kgK": S,
        "CP_kJ_kgK": CP,
        "GAMMA": GAMMA,
    })

    species = parse_mole_frac_block(txt)
    return scalars, species

# ---------------- JSON + Summary ----------------
def units_row(columns: List[str]) -> Dict[str, str]:
    units: Dict[str, str] = {c: "" for c in columns}
    units.update({
        "run_label": "",
        "P0_MPa": "MPa",
        "v_spec_m3_per_kg": "m^3/kg",
        "n_fuel_mol": "mol",
        "n_O2_mol": "mol",
        "mass_fuel_g": "g",
        "mass_O2_g": "g",
        "total_mass_kg": "kg",
        "x0_fuel": "mole frac",
        "x0_O2": "mole frac",
        "w0_fuel": "mass frac",
        "w0_O2": "mass frac",
        "P_MPa": "MPa",
        "delta_P_MPa": "MPa",
        "T_K": "K",
        "RHO_kg_m3": "kg/m^3",
        "H_kJ_kg": "kJ/kg",
        "U_kJ_kg": "kJ/kg",
        "G_kJ_kg": "kJ/kg",
        "S_kJ_kgK": "kJ/(kg*K)",
        "CP_kJ_kgK": "kJ/(kg*K)",
        "GAMMA": "dimensionless",
    })
    for c in columns:
        if c not in units and c != "run_label":
            units[c] = "mole frac"
    return units

def build_row(run_label: str, inputs: Dict[str, float], scalars: Dict[str, float], species: Dict[str, float]) -> Dict[str, float]:
    row: Dict[str, float] = {"run_label": run_label}
    # Inputs
    for f in [
        "P0_MPa", "v_spec_m3_per_kg",
        "n_fuel_mol", "n_O2_mol",
        "mass_fuel_g", "mass_O2_g",
        "total_mass_kg",
        "x0_fuel", "x0_O2",
        "w0_fuel", "w0_O2",
    ]:
        if f in inputs:
            row[f] = inputs[f]

    # Scalars
    for k in ["P_MPa","T_K","RHO_kg_m3","H_kJ_kg","U_kJ_kg","G_kJ_kg","S_kJ_kgK","CP_kJ_kgK","GAMMA"]:
        row[k] = scalars.get(k)

    # ΔP = P - P0
    row["delta_P_MPa"] = (row["P_MPa"] - row["P0_MPa"]) if (row.get("P_MPa") is not None and row.get("P0_MPa") is not None) else None

    row["_species"] = species
    return row

def write_run_json(project_root: Path, run_label: str, listing_name: str, inputs: Dict[str, float], scalars: Dict[str, float], species: Dict[str, float]) -> Path:
    parsed_dir = project_root / "outputs" / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    out_path = parsed_dir / f"{run_label}_parsed.json"
    payload = {
        "run_label": run_label,
        "sources": {"listing_file": listing_name},
        "units": {
            "P0_MPa": "MPa",
            "v_spec_m3_per_kg": "m^3/kg",
            "n_fuel_mol": "mol",
            "n_O2_mol": "mol",
            "mass_fuel_g": "g",
            "mass_O2_g": "g",
            "total_mass_kg": "kg",
            "x0_fuel": "mole frac",
            "x0_O2": "mole frac",
            "w0_fuel": "mass frac",
            "w0_O2": "mass frac",
            "P_MPa": "MPa",
            "delta_P_MPa": "MPa",
            "T_K": "K",
            "RHO_kg_m3": "kg/m^3",
            "H_kJ_kg": "kJ/kg",
            "U_kJ_kg": "kJ/kg",
            "G_kJ_kg": "kJ/kg",
            "S_kJ_kgK": "kJ/(kg*K)",
            "CP_kJ_kgK": "kJ/(kg*K)",
            "GAMMA": "dimensionless",
            "species": "mole frac",
        },
        "inputs": inputs,
        "thermo": scalars,
        "species_mole_fractions": species,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

# ---------------- Driver ----------------
def discover_run_dirs(project_root: Path) -> List[Path]:
    base = project_root / "outputs" / "raw"
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])

def find_listing(run_dir: Path, run_label: str) -> Optional[Path]:
    candidates = []
    for suf in (".out",".OUT",".lis",".LIS",".lst",".LST"):
        candidates.append(run_dir / f"{run_label}{suf}")
    for suf in (".out",".OUT",".lis",".LIS",".lst",".LST"):
        candidates.extend(run_dir.glob(f"*{suf}"))
    for p in candidates:
        try:
            if p.exists() and p.stat().st_size > 0:
                return p
        except Exception:
            pass
    return None

def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Parse NASA CEA listings and build summary (single-fuel, fuel-agnostic).")
    ap.add_argument("project_root", type=Path, help="Project root (contains outputs/raw and manifests/)")
    ap.add_argument("--manifest", type=Path, default=None, help="Path to input_manifest.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fail-on-missing-listing", action="store_true")
    args = ap.parse_args()

    cfg = ParseConfig(
        project_root=args.project_root.resolve(),
        manifest_path=args.manifest.resolve() if args.manifest else None,
        workers=max(1, args.workers),
        fail_on_missing_listing=args.fail_on_missing_listing
    )

    manifest_df = load_manifest(cfg)
    run_dirs = discover_run_dirs(cfg.project_root)
    if not run_dirs:
        logging.error("No run directories found under outputs/raw/. Nothing to parse.")
        return

    parsed_rows: List[Dict[str, float]] = []
    species_union: set = set()

    for run_dir in run_dirs:
        run_label = canonical_label(run_dir.name)
        listing = find_listing(run_dir, run_label)
        if not listing:
            msg = f"No listing (.out/.lis/.lst) found for run {run_label} in {run_dir}"
            if cfg.fail_on_missing_listing:
                raise FileNotFoundError(msg)
            logging.warning(msg + " — skipping.")
            continue

        inputs = inputs_from_manifest_or_inp(run_label, run_dir, manifest_df)
        scalars, species = parse_listing(listing)

        row = build_row(run_label, inputs, scalars, species)
        parsed_rows.append(row)
        species_union.update(species.keys())
        write_run_json(cfg.project_root, run_label, listing.name, inputs, scalars, species)
        logging.info(f"[OK] {run_label}: parsed → {listing.name}")

    if not parsed_rows:
        logging.warning("No runs parsed; summary will not be written.")
        return

    base_cols = [
        "run_label",
        "P0_MPa", "v_spec_m3_per_kg",
        "n_fuel_mol", "n_O2_mol",
        "mass_fuel_g", "mass_O2_g",
        "total_mass_kg",
        "x0_fuel", "x0_O2",
        "w0_fuel", "w0_O2",
        "P_MPa", "delta_P_MPa",
        "T_K", "RHO_kg_m3", "H_kJ_kg", "U_kJ_kg", "G_kJ_kg", "S_kJ_kgK",
        "CP_kJ_kgK", "GAMMA",
    ]
    species_cols = sorted(species_union)
    all_cols = base_cols + species_cols

    rows_prepped: List[Dict[str, float]] = []
    for row in parsed_rows:
        out = {k: row.get(k) for k in base_cols}
        for sp in species_cols:
            out[sp] = row["_species"].get(sp) if "_species" in row else None
        rows_prepped.append(out)

    df = pd.DataFrame(rows_prepped, columns=all_cols)

    # UNITS row at top (ASCII-only)
    units_df = pd.DataFrame([units_row(all_cols)])
    summary = pd.concat([units_df, df], ignore_index=True)

    results_dir = cfg.project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "summary.csv"
    # UTF-8 with BOM to help Excel detect encoding
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logging.info(f"[OK] Wrote summary → {out_csv}")

if __name__ == "__main__":
    main()
