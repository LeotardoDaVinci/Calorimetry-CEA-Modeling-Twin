#!/usr/bin/env python3
r"""
manage_cea_pipeline.py

Orchestrates the full CEA workflow:

1) build_cea_uv_decks.py  <input_csv> <project_root>
2) run_cea_uv.py          <project_root> --cea-cmd ... [--workers N]
3) parse_cea_outputs.py   <project_root> [--manifest ...]
4) plot_cea_outputs.py    <project_root/results/summary.csv> --out-dir <plots_out>

Example (your EG project):
  python3 manage_cea_pipeline.py ^
    "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\Input-CSV\CEA-Input-Params-EG.csv" ^
    "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\EG" ^
    "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\EG\plots" ^
    --cea-cmd "C:\Users\joemi\OneDrive\Combustion_Lab\CEA\CEA\FCEA2.exe" --workers 8

Notes
- The first argument can be a CSV file OR a directory. If it's a directory, the script searches
  for a single CSV to use (prefers 'CEA-Input-Params-EG.csv' if present).
- The run step requires the CEA executable. Provide --cea-cmd or set the env var CEA_CMD.
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ---------- helpers ----------

def resolve_script(name: str) -> Path:
    here = Path(__file__).resolve().parent
    p = here / name
    if not p.exists():
        raise FileNotFoundError(f"Required script not found: {p}")
    return p

def pick_input_csv(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() == ".csv":
        return p.resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Input params path not found: {p}")

    # search recursively for CSVs
    candidates = list(p.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No .csv files found under: {p}")

    # prefer EG names if present
    preferred_names = {
        "cea-input-params-eg.csv",
        "cea_input_params_eg.csv",
        "cea-input-params.csv",
        "cea_input_params.csv",
    }
    preferred = [c for c in candidates if c.name.lower() in preferred_names]
    if len(preferred) == 1:
        return preferred[0].resolve()

    if len(candidates) == 1:
        return candidates[0].resolve()

    msg = "Multiple CSVs found under input params path; specify one explicitly:\n" + \
          "\n".join(f"  - {c}" for c in candidates)
    raise ValueError(msg)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def run_step(cmd: list[str], log_file: Path, cwd: Path | None = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n→ Running: {' '.join(map(str, cmd))}\n  • Log: {log_file}")
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd, text=True)
    if proc.returncode != 0:
        try:
            tail = log_file.read_text(encoding="utf-8").splitlines()[-50:]
            print("\n--- tail of log ---")
            print("\n".join(tail))
            print("--- end tail ---\n")
        except Exception:
            pass
        raise RuntimeError(f"Step failed (rc={proc.returncode}). See log: {log_file}")

# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Manage/run the entire CEA project pipeline.")
    ap.add_argument("input_params", type=str,
                    help="Path to input params CSV, or a directory containing it.")
    ap.add_argument("project_root", type=str,
                    help="Project root to build/run/parse under (will be created if missing).")
    ap.add_argument("plots_out", type=str,
                    help="Directory to write final plots (plot script controls clearing).")
    ap.add_argument("--cea-cmd", type=str, default=None,
                    help="Path to FCEA2.exe (defaults to env CEA_CMD).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers for run step (default 4).")
    ap.add_argument("--skip-build", action="store_true", help="Skip build step.")
    ap.add_argument("--skip-run", action="store_true", help="Skip run step.")
    ap.add_argument("--skip-parse", action="store_true", help="Skip parse step.")
    ap.add_argument("--skip-plot", action="store_true", help="Skip plot step.")
    args = ap.parse_args()

    # Resolve paths
    input_csv = pick_input_csv(args.input_params)
    project_root = ensure_dir(Path(args.project_root).resolve())
    plots_out = Path(args.plots_out).resolve()
    logs_dir = ensure_dir(project_root / "logs" / "pipeline")

    # Resolve scripts (expected in same folder as this manager)
    build_script = resolve_script("build_cea_uv_decks_starch.py")
    # build_script = resolve_script("build_cea_uv_decks_eg.py")
    run_script   = resolve_script("run_cea_uv.py")
    parse_script = resolve_script("parse_cea_outputs.py")
    plot_script  = resolve_script("plot_cea_outputs.py")

    # Resolve CEA exe
    cea_cmd = args.cea_cmd or os.environ.get("CEA_CMD")
    if not args.skip_run and not cea_cmd:
        raise ValueError("CEA executable not provided. Use --cea-cmd or set env var CEA_CMD.")

    # Timestamps for logs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) BUILD
    if not args.skip_build:
        run_step(
            cmd=[sys.executable, str(build_script), str(input_csv), str(project_root)],
            log_file=logs_dir / f"{stamp}_01_build.log",
        )
    else:
        print("… Skipping build step.")

    # 2) RUN
    if not args.skip_run:
        run_step(
            cmd=[sys.executable, str(run_script), str(project_root),
                 "--cea-cmd", str(cea_cmd), "--workers", str(args.workers)],
            log_file=logs_dir / f"{stamp}_02_run.log",
        )
    else:
        print("… Skipping run step.")

    # 3) PARSE
    if not args.skip_parse:
        manifest = project_root / "manifests" / "input_manifest.csv"
        cmd = [sys.executable, str(parse_script), str(project_root)]
        if manifest.exists():
            cmd += ["--manifest", str(manifest)]
        run_step(
            cmd=cmd,
            log_file=logs_dir / f"{stamp}_03_parse.log",
        )
    else:
        print("… Skipping parse step.")

    # 4) PLOT
    if not args.skip_plot:
        summary_csv = project_root / "results" / "summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"summary.csv not found at: {summary_csv}")
        run_step(
            cmd=[sys.executable, str(plot_script), str(summary_csv), "--out-dir", str(plots_out)],
            log_file=logs_dir / f"{stamp}_04_plot.log",
        )
    else:
        print("… Skipping plot step.")

    print("\n✅ Pipeline completed successfully.")

if __name__ == "__main__":
    main()
