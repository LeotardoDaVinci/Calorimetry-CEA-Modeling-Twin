# NASA CEA UV Automation Pipeline (Boron/Starch Bomb Calorimeter)

This repository contains a **three-stage pipeline** to:

1) **Build** NASA CEA input decks from a CSV manifest  
2) **Run** CEA (**FCEA2.exe**) non-interactively and capture outputs (no parsing)  
3) **Parse** the listing files into per-run JSON and a **wide** `summary.csv`

All examples below use your current Windows paths.

---

## Current Example Terminal Calls for This Project: 
```powershell
python3 build_cea_uv_decks.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\Input-CSV\CEA-Input-Params-EG.csv" "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\EG"

python3 run_cea_uv.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\EG" --cea-cmd "C:\Users\joemi\OneDrive\Combustion_Lab\CEA\CEA\FCEA2.exe" --workers 8

python3 parse_cea_outputs.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\EG"

python plot_cea_outputs.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\EG\results\summary.csv" --out-dir "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\EG\plots"

```
## Full Terminal Call for This Project: 
# Ethylene Glycol:
```powershell
python3 manage_cea_pipeline.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\Input-CSV\CEA-Input-Params-EG.csv" "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\EG" "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\EG\plots" --cea-cmd "C:\Users\joemi\OneDrive\Combustion_Lab\CEA\CEA\FCEA2.exe" --workers 8
```

# PC:
```powershell
python3 manage_cea_pipeline.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\Input-CSV\CEA-Input-Params-PC.csv" "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\PC" "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Kyle-Gas-Phase-Energy\PC\plots" --cea-cmd "C:\Users\joemi\OneDrive\Combustion_Lab\CEA\CEA\FCEA2.exe" --workers 8
```

## Project structure

```
root/
  CEA_Input_Params.csv               # your input table (source for build stage)
  input_decks/                       # auto-generated decks
    <RUN>.inp
  outputs/
    raw/
      <RUN>/
        <RUN>.inp                    # copied here for isolated execution
        <RUN>.out / .lis / .lst      # listing (if produced)
        <RUN>.plt                    # plot file (if requested in deck)
        _answers.txt                 # CRLF-fed answers to FCEA2 prompts
        cea_stdout.txt               # full console transcript
        cea_stderr.txt               # errors, if any
        thermo.lib, trans.lib        # copied from CEA folder for stability
    parsed/
      <RUN>_parsed.json              # per-run parsed payload (from parser step)
  results/
    summary.csv                      # wide, one row per run (first row = UNITS)
  manifests/
    input_manifest.csv               # (auto-written by builder; the parser prefers this)
  logs/                              # optional, if you redirect stdout/stderr of scripts
  docs/
    README.md                        # (this file)
```

---

## Prerequisites

- **Windows + PowerShell**
- **Python 3.9+**
- **pandas**:  
  ```powershell
  python3 -m pip install --upgrade pip
  python3 -m pip install pandas
  ```
- NASA **CEA** (**FCEA2.exe**) is installed. Your path:  
  `C:\Users\joemi\OneDrive\Combustion_Lab\CEA\CEA\FCEA2.exe`

> Optional: set an environment variable so you don’t have to pass `--cea-cmd` each time:
> ```powershell
> setx CEA_CMD "C:\Users\joemi\OneDrive\Combustion_Lab\CEA\CEA\FCEA2.exe"
> ```
> (Open a new PowerShell after running `setx`.)

---

## The pipeline at a glance

**Inputs → Decks → Execution → Listings/Plots → Parsing → Summary**

1. **Build**: reads `CEA_Input_Params.csv`, creates `input_decks/<RUN>.inp`, and writes a normalized `manifests/input_manifest.csv` the parser can use later.
2. **Run**: executes each deck with **FCEA2.exe** in an isolated per-run directory under `outputs/raw/<RUN>/`, making sure the correct `thermo.lib` and `trans.lib` are visible. Captures stdout/stderr. **No parsing** here.
3. **Parse**: scans `outputs/raw/**` for listing files (`.out/.lis/.lst`), parses thermodynamic scalars and **MOLE FRACTIONS**, merges with input parameters from the manifest (or falls back to the `.inp`), writes per-run JSON, and builds a **wide** `results/summary.csv`.

---

## Stage 1 — Build input decks

**Script:** `build_cea_uv_decks.py`  
**Input:** `CEA_Input_Params.csv` (your experimental matrix)  
**Output:** `input_decks/<RUN>.inp` and `manifests/input_manifest.csv`

**Expected CSV columns (your current setup):**
```
Run,Mass PP,Mass B,Po
2115,0.1592,0.0227,3.10
```
- **Run** → deck/run label (e.g., `2115`)
- **Mass PP** (g) → starch (packing peanut) mass
- **Mass B** (g) → boron mass
- **Po** (MPa abs) → initial absolute pressure (pure O₂)

**Command (PowerShell):**
```powershell
python3 build_cea_uv_decks.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Boron\CEA\Input_CSV\CEA_Input_Params.csv" "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Boron\CEA\root"
```

**What it does**
- Validates/normalizes the CSV and writes `manifests/input_manifest.csv`
- Computes reactant **moles** from masses (and starch formula) and generates CEA UV problem decks
- **Tip:** The `output` block in each deck can include both listing and plot directives, e.g.:
  ```
  output
      massf  molef  trace
      plot   t p h s g cp
  end
  ```

---

## Stage 2 — Run CEA (execute-only)

**Script:** `run_cea_exec_only.py`  
**Input:** `input_decks/*.inp`  
**Output:** Executed runs under `outputs/raw/<RUN>/` (listing/plot/logs/libs)

**Command (PowerShell):**
```powershell
python3 run_cea_uv.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Boron\CEA\root" --cea-cmd "C:\Users\joemi\OneDrive\Combustion_Lab\CEA\CEA\FCEA2.exe" --workers 4
```

**What it does**
- For each `<RUN>.inp`:
  - Creates `outputs/raw/<RUN>/` and copies the deck there
  - Copies `thermo.lib` and `trans.lib` from the **FCEA2.exe** folder into the run folder
  - Forces `HOME`/`USERPROFILE` to the run folder to avoid `C:\Users\...` lib lookups
  - Non-interactively runs **FCEA2.exe** by piping `_answers.txt` (CRLF) via the shell
  - Captures `cea_stdout.txt` and `cea_stderr.txt`
  - If the listing is written to the **EXE** folder, copies it back into the run folder
- **Does not parse**; it just executes and organizes artifacts neatly.

**Knobs**
- `--workers 4` controls parallelism
- `--cea-cmd` can be omitted if you set `CEA_CMD` env var

---

## Stage 3 — Parse outputs

**Script:** `parse_cea_outputs.py`  
**Input:** `outputs/raw/<RUN>/*.out|*.lis|*.lst` (+ manifest or `.inp`)  
**Output:** `outputs/parsed/<RUN>_parsed.json` and `results/summary.csv`

**Command (PowerShell):**
```powershell
python3 parse_cea_outputs.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Boron\CEA\root"
```
(Optionally specify the manifest explicitly:)
```powershell
python3 parse_cea_outputs.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Boron\CEA\root" --manifest "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\Boron\CEA\root\manifests\input_manifest.csv"
```

**What it extracts**

From **manifest / `.inp`** (inputs):
- `P0_MPa` (initial MPa abs)
- `v_spec_m3_per_kg`
- `n_B_mol`, `n_PC_mol`, `n_O2_mol`
- `mass_B_g`, `mass_PC_g`, `mass_O2_g`, `total_mass_kg`
- `x0_B`, `x0_PC`, `x0_O2` (initial mole fractions)
- `w0_B`, `w0_PC`, `w0_O2` (initial mass fractions)
- **BLF** (Boron Loading Fraction): `mass_B_g * 1000 / mass_PC_g` (mg of B per g of starch)

From **listing** (first thermodynamic state):
- `P_MPa` (converted from `P, BAR` → MPa)  
- `delta_P_MPa` = `P_MPa - P0_MPa`
- `T_K`, `RHO_kg_m3`, `H_kJ_kg`, `U_kJ_kg`, `G_kJ_kg`, `S_kJ_kgK`
- `CP_kJ_kgK`, `GAMMA` (robust regex; case-insensitive)
- **MOLE FRACTIONS**: all species between `MOLE FRACTIONS` and the “PRODUCTS WHICH WERE CONSIDERED …” line  
  - Leading `*` is stripped (e.g., `*CO2` → `CO2`)  
  - Known bogus token `TO` is ignored  
  - Species like `B2O3(L)` are preserved as-is

**Outputs written**
- Per run: `outputs/parsed/<RUN>_parsed.json`  
  (includes `inputs`, `thermo`, and `species_mole_fractions`, plus a `units` map)
- Summary: `results/summary.csv`  
  - **Row 1** = **UNITS** (ASCII, Excel-friendly; file encoded **UTF-8 with BOM**)  
  - **One row per run**  
  - **One column per unique species** across all runs (blank if absent in a run)

**Knobs**
- `--fail-on-missing-listing` to stop if any run lacks a listing (default: skip with a warning)
- `--workers` (future-proof; current parser runs sequentially by default)

---

## Troubleshooting

### CEA waits at a prompt (“ENTER INPUT FILE NAME WITHOUT .inp EXTENSION.”)
- The run script feeds a CRLF `_answers.txt` via a shell pipe. If you still see this, check the run folder:
  - Confirm `_answers.txt` exists and contains the run base name on **three** lines.
  - Inspect `cea_stdout.txt` and `cea_stderr.txt` for details.

### Only `.plt` appears; no `.out`
- Some builds write the listing to the **EXE folder** or use `.lis`/`.lst`.  
  The run script auto-collects those back into the run folder. If absent, check:
  - The EXE folder
  - Windows VirtualStore: `%LOCALAPPDATA%\VirtualStore`

### `forrtl: severe (24): end-of-file during read, unit 14, file C:\Users\...\thermo.lib`
- Your CEA is reading a bad `thermo.lib` in your **home** directory. The run script:
  - Copies the correct `thermo.lib`/`trans.lib` from the EXE folder into each run dir.
  - Forces `HOME`/`USERPROFILE` to the run dir for CEA’s process.  
  If you still see this, remove/rename stray `C:\Users\<you>\thermo.lib` and `trans.lib`.

### Encoding or weird characters like `â€“` in CSV
- The parser writes `summary.csv` as **UTF-8 with BOM**. Excel should render units properly (Gamma’s unit is `dimensionless`).

### OneDrive / Controlled Folder Access
- If Windows blocks certain extensions or writes get redirected, try a local path outside OneDrive (e.g., `C:\CEA_test\root\`) to isolate the issue.

---

## Re-running / Cleaning

- **Re-build decks:** delete/overwrite `input_decks/*.inp` and rerun the **build** command.  
- **Re-run CEA:** re-run the **run** command; it executes every deck under `input_decks/`.  
- **Re-parse:** re-run the **parse** command; it scans `outputs/raw/*` and writes fresh JSON + CSV.  
- You can safely delete `outputs/parsed/*.json` and `results/summary.csv` to force clean regeneration.

---

## Notes

- The parser currently uses the **first** thermodynamic state block per listing (UV case).  
- The species section is delimited by `MOLE FRACTIONS` and `PRODUCTS WHICH WERE CONSIDERED …` and is robust to minor formatting differences.  
- If a run lacks a listing altogether, the parser **skips it with a warning** (no `.plt` fallback by design).

Happy modeling!
