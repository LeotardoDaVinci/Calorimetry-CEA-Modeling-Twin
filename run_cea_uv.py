#!/usr/bin/env python3
r"""
run_cea_exec_only.py  — EG project compatible

Execute all NASA CEA UV input decks without parsing or consolidation.

Per run:
  - Copies <project_root>/input_decks/<RUN>.inp → <project_root>/outputs/raw/<RUN>/
  - Copies thermo.lib / trans.lib from the FCEA2.exe folder into that run dir
  - Forces HOME/USERPROFILE to the run dir (so CEA won’t use C:\Users\<user>\thermo.lib)
  - Feeds the deck base name via a CRLF answers file:  type _answers.txt | FCEA2.exe
  - Captures stdout/stderr to cea_stdout.txt / cea_stderr.txt
  - Collects any listing/plot files (.out/.lis/.lst/.plt) that CEA may drop in the EXE folder
Notes:
  - This script is agnostic to the chemistry (EG-only or mixtures). It just executes decks.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------- config ----------------
@dataclass(frozen=True)
class ExecConfig:
    project_root: Path
    cea_cmd: Optional[str] = None   # path to FCEA2.exe (or env CEA_CMD)
    workers: int = 4
    timeout_s: int = 180
    copy_libs: bool = True          # copy thermo.lib/trans.lib into run dirs

# ------------- filesystem helpers -------------
def find_input_decks(root: Path) -> List[Path]:
    decks_dir = root / "input_decks"
    decks = sorted(decks_dir.glob("*.inp"))
    if not decks:
        raise FileNotFoundError(f"No .inp files found in {decks_dir}")
    return decks

def run_label_from_deck(deck: Path) -> str:
    return deck.stem

def prepare_run_dir(root: Path, label: str) -> Path:
    raw_dir = root / "outputs" / "raw" / label
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def resolve_cea_cmd(cea_cmd_arg: Optional[str]) -> Path:
    p = cea_cmd_arg or os.environ.get("CEA_CMD")
    if not p:
        raise RuntimeError("Set --cea-cmd to FCEA2.exe or define env CEA_CMD.")
    exe = Path(p)
    if not exe.exists():
        raise FileNotFoundError(f"CEA executable not found: {exe}")
    return exe

def ensure_cea_libs(run_dir: Path, cea_exe: Path, copy: bool = True) -> None:
    """
    Copy thermo.lib/trans.lib from the EXE folder into run_dir.
    CEA sometimes insists on finding these in %USERPROFILE% (we override HOME/USERPROFILE),
    so keeping local copies avoids surprises.
    """
    if not copy:
        return
    src = cea_exe.parent
    missing = []
    for lib in ("thermo.lib", "trans.lib"):
        s = src / lib
        d = run_dir / lib
        if s.exists():
            if not d.exists():
                shutil.copy2(s, d)
        else:
            missing.append(lib)
    if missing:
        raise FileNotFoundError(
            f"Missing required library file(s) next to {cea_exe.name}: {', '.join(missing)}"
        )

# ------------- execution helpers -------------
def _env_for_run(run_dir: Path) -> dict:
    """
    Force HOME/USERPROFILE to run_dir so CEA never touches user-home libs.
    Also nudge TMP/TEMP to the run dir (avoids write issues under OneDrive).
    """
    env = os.environ.copy()
    rd = run_dir.resolve()
    drive, tail = os.path.splitdrive(str(rd))
    tail = "\\" + tail.replace("/", "\\").lstrip("\\")
    env["HOME"] = str(rd)
    env["USERPROFILE"] = str(rd)
    env["HOMEDRIVE"] = drive if drive else os.environ.get("HOMEDRIVE", "")
    env["HOMEPATH"] = tail
    env["TMP"] = str(rd)
    env["TEMP"] = str(rd)
    return env

def exec_cea(run_dir: Path, inp_name: str, cea_exe: Path, timeout_s: int) -> None:
    """
    Feed base name to FCEA2 via a CRLF answers file and a shell pipe (Windows).
    CEA asks 3 times: input name (w/o .inp), listing name, plot name — we reuse the same base.
    """
    base = Path(inp_name).stem
    answers = run_dir / "_answers.txt"
    answers.write_bytes((base + "\r\n" + base + "\r\n" + base + "\r\n").encode("ascii", "ignore"))

    env = _env_for_run(run_dir)

    with open(run_dir / "cea_stdout.txt", "w", encoding="utf-8", newline="") as so, \
         open(run_dir / "cea_stderr.txt", "w", encoding="utf-8", newline="") as se:
        # 'type file | exe' keeps everything non-interactive on Windows
        cmd = f'type "{answers.name}" | "{cea_exe}"'
        subprocess.run(
            cmd,
            cwd=run_dir,
            shell=True,
            env=env,
            stdout=so,
            stderr=se,
            timeout=timeout_s,
            check=True
        )

def collect_outputs(run_dir: Path, cea_exe: Path, label: str) -> None:
    """
    Make sure the obvious artifacts are in the run dir.
    If a listing/plot was written in the EXE folder, copy it back here.
    """
    exedir = cea_exe.parent
    suffixes = (".out", ".OUT", ".lis", ".LIS", ".lst", ".LST", ".plt", ".PLT")
    # copy from EXE dir → run dir if matching label.*
    for suf in suffixes:
        src = exedir / f"{label}{suf}"
        if src.exists() and src.stat().st_size > 0:
            dst = run_dir / src.name
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
    # (If CEA already wrote into run_dir, those files are already there.)

# ------------- driver -------------
def process_one(deck: Path, cfg: ExecConfig) -> Tuple[str, Path]:
    label = run_label_from_deck(deck)
    run_dir = prepare_run_dir(cfg.project_root, label)

    # copy deck into run dir (isolated execution)
    local_inp = run_dir / f"{label}.inp"
    shutil.copy2(deck, local_inp)

    cea_exe = resolve_cea_cmd(cfg.cea_cmd)

    # ensure correct libs are present locally
    ensure_cea_libs(run_dir, cea_exe, copy=cfg.copy_libs)

    # execute
    exec_cea(run_dir, local_inp.name, cea_exe, cfg.timeout_s)

    # collect listing/plot if they landed in the EXE folder
    collect_outputs(run_dir, cea_exe, label)

    return label, run_dir

def main() -> None:
    ap = argparse.ArgumentParser(description="Execute NASA CEA decks without parsing/consolidation.")
    ap.add_argument("project_root", type=Path, help="Project root containing input_decks/")
    ap.add_argument("--cea-cmd", type=str, default=None, help="Path to FCEA2.exe (or env CEA_CMD)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    ap.add_argument("--timeout", type=int, default=180, help="Per-run timeout seconds (default 180)")
    ap.add_argument("--no-copy-libs", action="store_true", help="Do not copy thermo/trans libs into run dirs")
    args = ap.parse_args()

    cfg = ExecConfig(
        project_root=args.project_root.resolve(),
        cea_cmd=args.cea_cmd,
        workers=max(1, args.workers),
        timeout_s=max(10, args.timeout),
        copy_libs=not args.no_copy_libs
    )

    decks = find_input_decks(cfg.project_root)

    errors = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(process_one, d, cfg): d for d in decks}
        for fut in as_completed(futures):
            deck = futures[fut]
            label = deck.stem
            try:
                _, run_dir = fut.result()
                print(f"[OK] {label}: executed in {run_dir}")
            except Exception as e:
                errors.append((label, str(e)))
                print(f"[ERR] {label}: {e}")

    if errors:
        print("\n[REPORT] Some runs failed:")
        for lbl, msg in errors:
            print(f"  - {lbl}: {msg}")

if __name__ == "__main__":
    main()
