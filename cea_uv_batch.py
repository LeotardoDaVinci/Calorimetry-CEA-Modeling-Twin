#!/usr/bin/env python3
"""
cea_uv_batch.py — Batch NASA CEA UV (constant-V, adiabatic) runs over many fuels,
and COLLECTS per-case .out listings by case id so you can inspect them.

CHANGES IN THIS VERSION
-----------------------
- Adds molecular weight (mw_g_mol) as a column in the results CSV (with UNITS row).

Overview
--------
- Input CSV (exact headers, case-sensitive): "Mass Fuel", "Po"
  * "Mass Fuel": grams (g)
  * "Po": MPa absolute
- Fuel catalog (JSON or YAML): list of fuel entries (see schema below).
- For each CSV row and each fuel:
  1) build UV deck
  2) run CEA
  3) parse peak pressure (MPa) from listing (stdout-first, then .OUT/.LIS/.LST)
  4) write one consolidated results CSV (first row = UNITS)
  5) collect each case’s listing as `<CASEID>.out` in a single folder for review

Output
------
1) Results CSV with columns + UNITS first row:
   fuel, mw_g_mol, mass_fuel_g, P0_MPa, hoc_kj_g, total_Q_kJ, P_peak_MPa
2) A folder (default: sibling to results CSV) named 'collected_listings'
   containing per-case listings named `<CASEID>.out` (sorted by case id).
   If CEA didn’t emit a native .OUT/.LIS/.LST, we save the stdout text
   as the listing so you can inspect what CEA printed.

Fuel Catalog Schema
-------------------
Each fuel is an object with the following fields:

Required:
- label: str                     # short id used in outputs (e.g., "paraffin", "EG")
- hoc_kj_g: float                # heat of combustion (kJ/g) used only for total_Q_kJ

Identity / amount (choose one path):
(1) Library species (preferred when available)
- cea_name: str                  # a CEA library token, e.g. "C2H6O2(L)"
- EITHER:
    - mw_g_mol: float            # MW (g/mol) to convert mass→moles
  OR
    - formula: str               # e.g., "C 25 H 52" — MW will be computed
- (optional) dhf_kj_mol: float   # usually NOT provided for library species

(2) Custom reactant (composition + enthalpy of formation)
- formula: str                   # "C a H b O c ..." (space-separated tokens)
- dhf_kj_mol: float              # required when formula is used for custom fuel
- (optional) cea_name: str       # label for the react line; 'label' is still used in outputs
- (optional) mw_g_mol: float     # if provided, overrides computed MW from formula

Notes:
- If "formula" is provided and "cea_name" is omitted, the deck defines a custom reactant line.
- If both "cea_name" and "formula" are given: treated as CUSTOM if "dhf_kj_mol" is present,
  otherwise treated as LIBRARY (fuel=cea_name) with formula used only to compute MW if mw_g_mol is missing.

CLI
---
cea_uv_batch.py INPUT_CSV --fuels FUELS.json --cea-cmd "C:\\CEA\\FCEA2.exe"
                   [--out results.csv] [--collect-dir <dir>]
                   [--volume-ml 240.0] [--t0-k 303.5]
                   [--workers 8] [--timeout-s 180]
                   [--clean-work]  # if set, deletes per-case work dirs at the end

(c) 2025
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import sys
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

# ----------------------------- Constants -----------------------------

R_J_PER_MOLK: float = 8.314462618
MW_O2_G_PER_MOL: float = 31.998
DEFAULT_VOLUME_ML: float = 240.0
DEFAULT_T0_K: float = 303.5
DEFAULT_WORKERS: int = 8
DEFAULT_TIMEOUT_S: int = 180

# Periodic table (extend as needed)
ATOMIC_WEIGHTS = {
    "H": 1.00794,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "Cl": 35.453,
    "S": 32.065,
    "P": 30.973762,
    "Na": 22.98976928,
    "K": 39.0983,
    "Al": 26.9815386,
    "Si": 28.0855,
    "B": 10.811,
}

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"

# ------------------------- Data Structures --------------------------

@dataclass(frozen=True)
class Fuel:
    """A single fuel definition from the catalog."""
    label: str
    hoc_kj_g: float
    cea_name: Optional[str] = None
    formula: Optional[str] = None
    dhf_kj_mol: Optional[float] = None
    mw_g_mol: Optional[float] = None

    def is_custom(self) -> bool:
        """Custom reactant if both formula and dhf_kj_mol are present."""
        return (self.formula is not None) and (self.dhf_kj_mol is not None)

# ------------------------- Utility Functions ------------------------

def _case_id(label: str) -> str:
    """
    Produce an <=8 char uppercase case id (CEA-safe) from an arbitrary label.
    3 letters from label stem + 5 hex chars of a hash.
    """
    stem = re.sub(r'[^A-Za-z0-9]', '', label).upper()
    if not stem:
        stem = "CASE"
    head = (stem[:3] if len(stem) >= 3 else (stem + "XXX")[:3])
    tail = hashlib.sha1(label.encode("utf-8")).hexdigest().upper()[:5]
    return (head + tail)[:8]

def load_fuels(path: Path) -> List[Fuel]:
    """Load fuel catalog from JSON or YAML (if available)."""
    if not path.exists():
        raise FileNotFoundError(f"Fuel catalog not found: {path}")

    ext = path.suffix.lower()
    if ext == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML fuel file provided but PyYAML is not installed.") from e
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Fuel catalog must be .json, .yaml, or .yml")

    if not isinstance(raw, list):
        raise ValueError("Fuel catalog must be a list of fuel entries.")

    fuels: List[Fuel] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Fuel entry #{i+1} must be an object, got {type(item)}")
        try:
            fuel = Fuel(
                label=str(item["label"]),
                hoc_kj_g=float(item["hoc_kj_g"]),
                cea_name=(str(item["cea_name"]) if item.get("cea_name") is not None else None),
                formula=(str(item["formula"]) if item.get("formula") is not None else None),
                dhf_kj_mol=(float(item["dhf_kj_mol"]) if item.get("dhf_kj_mol") is not None else None),
                mw_g_mol=(float(item["mw_g_mol"]) if item.get("mw_g_mol") is not None else None),
            )
        except KeyError as e:
            raise ValueError(f"Missing required key {e} in fuel entry #{i+1}") from e

        # Validate custom vs library rules
        if fuel.formula and (fuel.dhf_kj_mol is None) and (fuel.cea_name is None):
            raise ValueError(
                f"Fuel '{fuel.label}': 'formula' provided without 'dhf_kj_mol' (custom) "
                f"or 'cea_name' (library). Provide one."
            )

        fuels.append(fuel)

    return fuels

def parse_formula_to_atoms(formula: str) -> Dict[str, float]:
    """Parse a space-separated 'C 6 H 10 O 5' style formula into {element: count}."""
    toks = [t for t in formula.strip().split() if t]
    if len(toks) % 2 != 0:
        raise ValueError(f"Malformed formula '{formula}'. Expected pairs like 'C 6 H 10 O 5'.")
    atoms: Dict[str, float] = {}
    for sym, num in zip(toks[0::2], toks[1::2]):
        sym = sym.strip()
        if sym not in ATOMIC_WEIGHTS:
            sym_guess = sym[:1].upper() + sym[1:].lower()
            if sym_guess in ATOMIC_WEIGHTS:
                sym = sym_guess
            else:
                raise ValueError(f"Unknown element symbol '{sym}' in formula '{formula}'.")
        try:
            cnt = float(num)
        except Exception as e:
            raise ValueError(f"Non-numeric count '{num}' in formula '{formula}'.") from e
        atoms[sym] = atoms.get(sym, 0.0) + cnt
    return atoms

def compute_mw_from_formula(formula: str) -> float:
    """Compute molecular weight (g/mol) from a parsed 'C a H b O c ...' formula."""
    atoms = parse_formula_to_atoms(formula)
    mw = 0.0
    for sym, cnt in atoms.items():
        mw += ATOMIC_WEIGHTS[sym] * cnt
    return mw

def ensure_cea_exe(path_str: Optional[str]) -> Path:
    """Resolve the path to the CEA executable from CLI arg or env var."""
    ceap = path_str or os.environ.get("CEA_CMD")
    if not ceap:
        raise RuntimeError("Provide --cea-cmd or set environment variable CEA_CMD to point to FCEA2.exe.")
    exe = Path(ceap)
    if not exe.exists():
        raise FileNotFoundError(f"CEA executable not found: {exe}")
    return exe.resolve()

def read_input_csv_uniques(path: Path) -> Tuple[List[float], List[float]]:
    """
    Read 'Mass Fuel' (g) and 'Po' (MPa abs) from the CSV and return
    two **sorted unique** lists: (mass_values, po_values).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    mass_set, po_set = set(), set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        cols = [c.strip() for c in (rdr.fieldnames or [])]
        if "Mass Fuel" not in cols or "Po" not in cols:
            raise ValueError("Input CSV must have headers exactly: 'Mass Fuel', 'Po'.")
        for i, row in enumerate(rdr, start=2):
            ms = str(row.get("Mass Fuel", "")).strip()
            ps = str(row.get("Po", "")).strip()
            if not ms or not ps:
                # skip blank/partial rows silently
                continue
            try:
                mass = float(ms)
                po = float(ps)
            except Exception as e:
                raise ValueError(f"Row {i}: could not parse 'Mass Fuel' or 'Po' as float.") from e
            if not math.isfinite(mass) or not math.isfinite(po):
                continue
            if mass < 0 or po < 0:
                continue
            mass_set.add(mass)
            po_set.add(po)

    masses = sorted(mass_set)
    pressures = sorted(po_set)
    if not masses:
        raise ValueError("No valid 'Mass Fuel' values found.")
    if not pressures:
        raise ValueError("No valid 'Po' values found.")
    return masses, pressures

def read_input_csv_pairs(path: Path) -> List[Tuple[float, float]]:
    """
    Read 'Mass Fuel' (g) and 'Po' (MPa abs) from the CSV.
    Returns a list of (mass_g, P0_MPa).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    rows: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        cols = [c.strip() for c in (rdr.fieldnames or [])]
        if "Mass Fuel" not in cols or "Po" not in cols:
            raise ValueError("Input CSV must have headers exactly: 'Mass Fuel', 'Po'.")
        for i, row in enumerate(rdr, start=2):
            try:
                m = float(str(row["Mass Fuel"]).strip())
                p = float(str(row["Po"]).strip())
            except Exception as e:
                raise ValueError(f"Row {i}: could not parse 'Mass Fuel' or 'Po' as float.") from e
            rows.append((m, p))
    if not rows:
        raise ValueError("Input CSV contains no data rows.")
    return rows

def compute_case_inputs(
    mass_fuel_g: float,
    P0_MPa: float,
    V_m3: float,
    T0_K: float,
) -> Tuple[float, float]:
    """
    Compute moles of O2 (for the oxid line) and the specific volume denominator term from O2.
    Returns (n_O2_mol, m_O2_g).  v_spec = V_m3 / ((mass_fuel_g + m_O2_g)/1000).
    """
    P0_Pa = P0_MPa * 1e6
    n_O2 = (P0_Pa * V_m3) / (R_J_PER_MOLK * T0_K)  # mol
    m_O2_g = n_O2 * MW_O2_G_PER_MOL
    return n_O2, m_O2_g

def deck_text_uv(
    *,
    v_spec_m3_per_kg: float,
    T0_K: float,
    fuel: Fuel,
    n_fuel_mol: float,
    n_O2_mol: float,
) -> str:
    """Render a minimal UV deck for either a library or custom fuel."""
    lines: List[str] = []
    lines.append("problem")
    lines.append(f"  uv   v,m**3/kg={v_spec_m3_per_kg:.9g}")
    lines.append("react")
    if (fuel.formula is not None) and (fuel.dhf_kj_mol is not None):
        nm = (fuel.cea_name or fuel.label).strip()
        lines.append(f"  fuel={nm}   moles={n_fuel_mol:.9g}  t,k={T0_K:.4f}")
        lines.append(f"    h,kj/mol={fuel.dhf_kj_mol:.9g}  {fuel.formula}")
    else:
        if not fuel.cea_name:
            raise ValueError(
                f"Fuel '{fuel.label}' treated as library species but 'cea_name' missing. "
                f"Provide 'cea_name' or switch to custom (formula + dhf_kj_mol)."
            )
        lines.append(f"  fuel={fuel.cea_name}   moles={n_fuel_mol:.9g}  t,k={T0_K:.4f}")
    lines.append(f"  oxid=O2   moles={n_O2_mol:.9g}  t,k={T0_K:.4f}")
    lines.append("output")
    lines.append("  short")
    lines.append("end")
    lines.append("")
    return "\n".join(lines)

def ensure_libs_in_run_dir(run_dir: Path, cea_exe: Path) -> None:
    """Copy thermo.lib and trans.lib from the EXE's directory to the run directory if present."""
    src = cea_exe.parent
    for name in ("thermo.lib", "trans.lib"):
        p = src / name
        if p.exists():
            shutil.copy2(p, run_dir / name)

# ------------------------- Parsing helpers --------------------------

def parse_peak_pressure_MPa_text(listing_text: str) -> Optional[float]:
    """
    Parse final (peak) pressure in MPa from a CEA listing text (stdout or .OUT).
    Anchor to:
        THERMODYNAMIC EQUILIBRIUM COMBUSTION PROPERTIES AT ASSIGNED
                                         VOLUME
        ...
        THERMODYNAMIC PROPERTIES
        P, BAR   <value>
    Fallbacks: last 'THERMODYNAMIC PROPERTIES' block; last 'P, BAR' anywhere.
    """
    # 1) Prefer the LAST "ASSIGNED ... VOLUME" header
    m_hdrs = list(re.finditer(
        r"THERMODYNAMIC\s+EQUILIBRIUM\s+COMBUSTION\s+PROPERTIES\s+AT\s+ASSIGNED\s+\n\s*VOLUME",
        listing_text, flags=re.I
    ))

    def _scan_after(idx: int) -> Optional[float]:
        sub = listing_text[idx:]
        m_prop = re.search(r"THERMODYNAMIC\s+PROPERTIES", sub, flags=re.I)
        if not m_prop:
            return None
        window = sub[m_prop.end(): m_prop.end()+8000]
        m_p = re.search(r"^\s*P,\s*BAR\s*=?\s*(" + FLOAT_RE + r")", window, flags=re.M | re.I)
        if not m_p:
            return None
        try:
            return 0.1 * float(m_p.group(1))  # bar -> MPa
        except Exception:
            return None

    if m_hdrs:
        p = _scan_after(m_hdrs[-1].end())
        if p is not None:
            return p

    # 2) Fallback: last "THERMODYNAMIC PROPERTIES"
    props = list(re.finditer(r"THERMODYNAMIC\s+PROPERTIES", listing_text, flags=re.I))
    if props:
        tail = listing_text[props[-1].end():]
        m_p = re.search(r"^\s*P,\s*BAR\s*=?\s*(" + FLOAT_RE + r")", tail, flags=re.M | re.I)
        if m_p:
            try:
                return 0.1 * float(m_p.group(1))
            except Exception:
                pass

    # 3) Last 'P, BAR' anywhere
    matches = list(re.finditer(r"P,\s*BAR\s*=?\s*(" + FLOAT_RE + r")", listing_text, flags=re.I))
    if matches:
        try:
            return 0.1 * float(matches[-1].group(1))
        except Exception:
            return None

    return None

def parse_peak_pressure_MPa_from_file(path: Path) -> Optional[float]:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    return parse_peak_pressure_MPa_text(txt)

# ------------------------- Runner per case --------------------------

def run_cea_case(
    *,
    deck_str: str,
    human_label: str,
    case_id: str,
    cea_exe: Path,
    timeout_s: int,
    work_root: Path,
    collect_dir: Path,
) -> Tuple[str, str, Optional[float], Optional[str], Path]:
    """
    Run one CEA deck in an isolated work dir; return:
      (human_label, case_id, P_peak_MPa, error_str, run_dir)

    Side effects:
      - Writes deck, answers, stdout, stderr under run_dir
      - If a native listing (<CASEID>.out/.lis/.lst) exists, copy it into run_dir
      - Always writes a normalized listing into collect_dir as '<CASEID>.out':
          * if native listing exists -> copy it as <CASEID>.out
          * else fallback: copy stdout text as <CASEID>.out
    """
    run_dir = (work_root / f"{case_id}__{human_label}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write deck + libs + answers
        deck_path = run_dir / f"{case_id}.inp"
        deck_path.write_text(deck_str, encoding="utf-8")
        ensure_libs_in_run_dir(run_dir, cea_exe)

        answers = (case_id + "\r\n" + case_id + "\r\n" + case_id + "\r\n").encode("ascii", "ignore")
        (run_dir / "_answers.txt").write_bytes(answers)

        # Isolate HOME/USERPROFILE so CEA uses local libs and keeps outputs tidy
        env = os.environ.copy()
        rd = run_dir.resolve()
        drive, tail = os.path.splitdrive(str(rd))
        tail_win = "\\" + tail.replace("/", "\\").lstrip("\\")
        env["HOME"] = str(rd)
        env["USERPROFILE"] = str(rd)
        if drive:
            env["HOMEDRIVE"] = drive
        env["HOMEPATH"] = tail_win

        # Pipe case id into EXE
        is_windows = platform.system().lower().startswith("win")
        pipe_cmd = f'type "{(run_dir / "_answers.txt").name}" | "{cea_exe}"' if is_windows \
                   else f'cat "{(run_dir / "_answers.txt").name}" | "{cea_exe}"'

        import subprocess
        with (run_dir / "cea_stdout.txt").open("w", encoding="utf-8") as so, \
             (run_dir / "cea_stderr.txt").open("w", encoding="utf-8") as se:
            subprocess.run(
                pipe_cmd,
                cwd=run_dir,
                shell=True,
                env=env,
                stdout=so,
                stderr=se,
                timeout=max(10, timeout_s),
                check=True,
            )

        # Try to collect a native listing by case id
        native_listing: Optional[Path] = None
        for suf in (".out", ".OUT", ".lis", ".LIS", ".lst", ".LST"):
            src = cea_exe.parent / f"{case_id}{suf}"
            if src.exists() and src.stat().st_size > 0:
                dst = run_dir / src.name
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
                native_listing = dst
                break
        # Some builds write into CWD (run_dir)
        if native_listing is None:
            for suf in (".out", ".OUT", ".lis", ".LIS", ".lst", ".LST"):
                p = run_dir / f"{case_id}{suf}"
                if p.exists() and p.stat().st_size > 0:
                    native_listing = p
                    break

        # Normalize: ensure we create '<case_id>.out' in collect_dir
        collect_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = collect_dir / f"{case_id}.out"
        if native_listing is not None:
            # Copy as <case_id>.out (rename extension)
            try:
                txt = native_listing.read_text(encoding="utf-8", errors="ignore")
                normalized_path.write_text(txt, encoding="utf-8")
            except Exception:
                # If read fails, fallback to stdout
                stdout_txt = (run_dir / "cea_stdout.txt").read_text(encoding="utf-8", errors="ignore")
                normalized_path.write_text(stdout_txt, encoding="utf-8")
        else:
            # No native listing—save stdout as the listing for review
            stdout_txt = (run_dir / "cea_stdout.txt").read_text(encoding="utf-8", errors="ignore")
            normalized_path.write_text(stdout_txt, encoding="utf-8")

        # Parse peak P from the normalized content
        P_MPa = parse_peak_pressure_MPa_from_file(normalized_path)
        if P_MPa is None:
            return human_label, case_id, None, "No peak P found in listing", run_dir
        return human_label, case_id, P_MPa, None, run_dir

    except Exception as e:
        return human_label, case_id, None, f"{type(e).__name__}: {e}", run_dir

# ------------------------------ I/O ---------------------------------

def write_results_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write results with a first UNITS row.
    Columns:
      fuel, mw_g_mol, mass_fuel_g, P0_MPa, hoc_kj_g, total_Q_kJ, P_peak_MPa
    Units:
      "",    "g/mol",  "g",         "MPa",  "kJ/g",   "kJ",        "MPa"
    """
    cols = ["fuel", "mw_g_mol", "mass_fuel_g", "P0_MPa", "hoc_kj_g", "total_Q_kJ", "P_peak_MPa"]
    units = {
        "fuel": "",
        "mw_g_mol": "g/mol",
        "mass_fuel_g": "g",
        "P0_MPa": "MPa",
        "hoc_kj_g": "kJ/g",
        "total_Q_kJ": "kJ",
        "P_peak_MPa": "MPa",
    }

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        # UNITS row as first data row
        w.writerow({k: units.get(k, "") for k in cols})
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    tmp.replace(out_path)

def write_index_csv(index_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write a debug index for all cases so you can jump to artifacts quickly.
    Columns:
      case_id, human_label, run_dir, collected_out, had_native_listing, error
    """
    cols = ["case_id", "human_label", "run_dir", "collected_out", "had_native_listing", "error"]
    tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    tmp.replace(index_path)

# ------------------------------ Main --------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Batch UV CEA runner over many fuels and (mass,Po) cases; collects .out per case id.")
    ap.add_argument("input_csv", type=Path, help="CSV with headers: Mass Fuel, Po (g and MPa abs; headers have no units)")
    ap.add_argument("--fuels", type=Path, required=True, help="Fuel catalog (.json/.yaml) per schema in script docstring")
    ap.add_argument("--cea-cmd", type=str, default=None, help="Path to FCEA2.exe (or set env CEA_CMD)")
    ap.add_argument("--out", type=Path, default=None, help="Output results CSV (default: input_csv parent / results.csv)")
    ap.add_argument("--collect-dir", type=Path, default=None, help="Folder to place normalized <CASEID>.out files")
    ap.add_argument("--volume-ml", type=float, default=DEFAULT_VOLUME_ML, help="Bomb free volume in mL (default: 240.0)")
    ap.add_argument("--mode", choices=["grid", "pairs"], default="grid",
                    help="How to combine 'Mass Fuel' and 'Po'. 'grid' = Cartesian product of unique values "
                    "from each column; 'pairs' = row-wise pairs (legacy behavior).")
    ap.add_argument("--t0-k", type=float, default=DEFAULT_T0_K, help="Initial temperature, K (default: 303.5)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers (default: 8)")
    ap.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S, help="Timeout per case, seconds (default: 180)")
    ap.add_argument("--clean-work", action="store_true", help="If set, delete per-case work dirs after finishing")
    args = ap.parse_args()

    # Resolve paths & inputs
    cea_exe = ensure_cea_exe(args.cea_cmd)

    if args.mode == "grid":
        mass_values, po_values = read_input_csv_uniques(args.input_csv)
        print(f"[INFO] Grid mode: {len(mass_values)} masses × {len(po_values)} pressures "
              f"= {len(mass_values) * len(po_values)} combos per fuel.")
    else:
        mass_po_rows = read_input_csv_pairs(args.input_csv)
        print(f"[INFO] Pairs mode: {len(mass_po_rows)} row-wise combos per fuel.")

    fuels = load_fuels(args.fuels)

    # Resolve MW for each fuel (needed to convert mass → moles, and for CSV output)
    prepared: List[Fuel] = []
    for fu in fuels:
        mw = fu.mw_g_mol
        if mw is None and fu.formula:
            try:
                mw = compute_mw_from_formula(fu.formula)
            except Exception as e:
                raise ValueError(f"Fuel '{fu.label}': could not compute MW from formula: {e}") from e
        if mw is None:
            raise ValueError(f"Fuel '{fu.label}': missing MW. Provide 'mw_g_mol' or 'formula' to compute it.")
        prepared.append(Fuel(
            label=fu.label,
            hoc_kj_g=fu.hoc_kj_g,
            cea_name=fu.cea_name,
            formula=fu.formula,
            dhf_kj_mol=fu.dhf_kj_mol,
            mw_g_mol=mw,
        ))

    # Output targets
    out_csv = args.out or (args.input_csv.parent / "results.csv")
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    collect_dir = args.collect_dir or (out_csv.parent / "collected_listings")
    collect_dir = collect_dir.resolve()
    collect_dir.mkdir(parents=True, exist_ok=True)

    # Work root (we keep it unless --clean-work)
    work_root = Path(tempfile.mkdtemp(prefix="cea_batch_work_")).resolve()

    results_rows: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []

    V_m3 = (args.volume_ml * 1e-6)
    T0_K = float(args.t0_k)
    timeout_s = int(max(10, args.timeout_s))

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for fuel in prepared:
            if args.mode == "grid":
                combos = [(m, p) for m in mass_values for p in po_values]
            else:
                combos = mass_po_rows

            for (mass_g, P0_MPa) in combos:
                n_O2_mol, m_O2_g = compute_case_inputs(mass_g, P0_MPa, V_m3, T0_K)
                n_fuel_mol = mass_g / float(fuel.mw_g_mol)  # type: ignore[arg-type]
                m_total_kg = (mass_g + m_O2_g) / 1000.0
                if m_total_kg <= 0.0:
                    raise ValueError("Non-positive total mass in case; check inputs.")
                v_spec = V_m3 / m_total_kg

                human_label = f"{fuel.label}_{mass_g:g}g_{P0_MPa:g}MPa"
                case_id = _case_id(human_label)

                deck = deck_text_uv(
                    v_spec_m3_per_kg=v_spec,
                    T0_K=T0_K,
                    fuel=fuel,
                    n_fuel_mol=n_fuel_mol,
                    n_O2_mol=n_O2_mol,
                )

                fut = pool.submit(
                    run_cea_case,
                    deck_str=deck,
                    human_label=human_label,
                    case_id=case_id,
                    cea_exe=cea_exe,
                    timeout_s=timeout_s,
                    work_root=work_root,
                    collect_dir=collect_dir,
                )
                futures[fut] = (fuel, mass_g, P0_MPa, case_id, human_label)

        for fut in as_completed(futures):
            fuel, mass_g, P0_MPa, case_id, human_label = futures[fut]
            hoc = fuel.hoc_kj_g
            total_Q = mass_g * hoc
            try:
                human_label_rt, case_id_rt, P_peak, err, run_dir = fut.result()
                assert case_id_rt == case_id
                row = {
                    "fuel": fuel.label,
                    "mw_g_mol": float(fuel.mw_g_mol) if fuel.mw_g_mol is not None else "",
                    "mass_fuel_g": mass_g,
                    "P0_MPa": P0_MPa,
                    "hoc_kj_g": hoc,
                    "total_Q_kJ": total_Q,
                    "P_peak_MPa": ("" if P_peak is None else P_peak),
                }
                if err:
                    print(f"[WARN] {human_label_rt}: {err}")
                results_rows.append(row)

                collected_out = (collect_dir / f"{case_id}.out")
                had_native = "yes" if collected_out.exists() else "no"
                index_rows.append({
                    "case_id": case_id,
                    "human_label": human_label,
                    "run_dir": str(run_dir),
                    "collected_out": str(collected_out),
                    "had_native_listing": had_native,
                    "error": (err or ""),
                })
            except Exception as e:
                print(f"[ERR] {human_label} — {type(e).__name__}: {e}")
                results_rows.append({
                    "fuel": fuel.label,
                    "mw_g_mol": float(fuel.mw_g_mol) if fuel.mw_g_mol is not None else "",
                    "mass_fuel_g": mass_g,
                    "P0_MPa": P0_MPa,
                    "hoc_kj_g": hoc,
                    "total_Q_kJ": total_Q,
                    "P_peak_MPa": "",
                })
                index_rows.append({
                    "case_id": case_id,
                    "human_label": human_label,
                    "run_dir": "",
                    "collected_out": "",
                    "had_native_listing": "no",
                    "error": f"{type(e).__name__}: {e}",
                })

    # Write consolidated results
    write_results_csv(out_csv, results_rows)
    print(f"[OK] Results: {out_csv}")

    # Write an index so you can jump straight to the artifacts
    index_csv = out_csv.parent / "collected_listings_index.csv"
    write_index_csv(index_csv, index_rows)
    print(f"[OK] Listing index: {index_csv}")
    print(f"[OK] Listings folder (sorted by case id): {collect_dir}")

    # Cleanup work dirs only if requested
    if args.clean_work:
        try:
            shutil.rmtree(work_root, ignore_errors=True)
        except Exception:
            pass
    else:
        print(f"[INFO] Per-case work dirs preserved at: {work_root}")

if __name__ == "__main__":
    main()
