# CEA Batch + Plot + ML Pipeline

This project automates NASA CEA (Chemical Equilibrium with Applications) simulations, visualization of combustion results, and data-driven modeling of peak pressure behavior. It includes **three core scripts**:

1. **`cea_uv_batch.py`** — Runs batch constant-volume adiabatic combustion (UV) CEA simulations for multiple fuels and initial conditions.
2. **`plot_outputs.py`** — Loads the consolidated results from CEA and generates high-quality 2D and 3D plots.
3. **`train_hoc_model.py`** — Trains a machine learning model (PyTorch MLP) to predict or invert heat of combustion (HOC) and peak pressure behavior.

---

## 🔧 Project Overview

The workflow proceeds in three stages:

1. **Run simulations** with `cea_uv_batch.py` using a JSON fuel catalog (`fuels.json`) and an input parameter CSV (`CEA-Input-Params.csv`).
2. **Visualize** the generated results (`output.csv`) using `plot_outputs.py`.
3. **Model** and optionally invert results using `train_hoc_model.py` to predict pressure behavior or estimate HOC.

---

## 📁 File Setup

### 1. `fuels.json`

This file defines the fuel catalog. Each entry describes one fuel with its formula, enthalpy of formation (if custom), and heat of combustion. Example:

```json
[
  {
    "label": "paraffin",
    "formula": "C 25 H 52",
    "dhf_kj_mol": -767.93,
    "hoc_kj_g": 46.77
  },
  {
    "label": "starch",
    "formula": "C 6 H 10 O 5",
    "dhf_kj_mol": -960.0,
    "hoc_kj_g": 17.0
  }
]
```

#### Required keys

| Key | Description |
|-----|--------------|
| `label` | Short fuel identifier (used in plots and outputs). |
| `hoc_kj_g` | Heat of combustion (kJ/g). |
| `formula` | Chemical composition in the format `"C a H b O c"`. |
| `dhf_kj_mol` | (Required for custom fuels) Enthalpy of formation, kJ/mol. |
| `cea_name` | (Optional) Library species name for CEA (e.g., `"JP-10(L)"`). |
| `mw_g_mol` | (Optional) Molecular weight in g/mol. |

Custom fuels **must** include both `formula` and `dhf_kj_mol`. Library fuels (CEA built-ins) can omit `dhf_kj_mol`.

---

### 2. `CEA-Input-Params.csv`

This file defines the simulation sweep parameters. Each row represents one simulation case.

**Headers must match exactly:**

| Column | Unit | Description |
|---------|------|-------------|
| `Mass Fuel` | g | Fuel mass for the case. |
| `Po` | MPa | Initial pressure in the bomb. |

Example:

```csv
Mass Fuel,Po
0.05,1.0
0.05,2.0
0.10,1.0
```

Each combination of mass and initial pressure will be run for every fuel in `fuels.json`.

---

## 🚀 Running the Batch Simulation

```bash
python cea_uv_batch.py CEA-Input-Params.csv --fuels fuels.json --cea-cmd "C:\CEA\FCEA2.exe" --out output.csv
```

### Optional Flags

| Flag | Description |
|------|--------------|
| `--collect-dir <dir>` | Folder for collected `.out` listings. |
| `--volume-ml` | Bomb free volume in mL (default 240). |
| `--t0-k` | Initial temperature in Kelvin (default 303.5 K). |
| `--workers` | Number of parallel workers (default 8). |
| `--timeout-s` | Per-case timeout in seconds (default 180 s). |
| `--clean-work` | Deletes temporary run directories after completion. |

### Outputs

- `output.csv` — Consolidated results with units in the first row.
- `collected_listings/` — Folder containing all normalized `.out` files for each run.

Each result row includes:

| Column | Description |
|---------|-------------|
| `fuel` | Fuel label from JSON. |
| `mw_g_mol` | Molecular weight (computed if missing). |
| `mass_fuel_g` | Fuel mass in grams. |
| `P0_MPa` | Initial pressure. |
| `hoc_kj_g` | Heat of combustion. |
| `total_Q_kJ` | Total heat release (kJ). |
| `P_peak_MPa` | Peak combustion pressure (MPa). |

---

## 📊 Plotting Results

To visualize results, run:

```bash
python plot_outputs.py output.csv --out-dir plots --plot2d --x P0_MPa --y P_peak_MPa
python plot_outputs.py output.csv --out-dir plots --plot3d --x total_Q_kJ --y P_peak_MPa --z P0_MPa
```

### Features

- 2D scatter plots (matplotlib) and 3D plots (Plotly).
- Automatically color-coded by fuel type.
- Automatically detects and removes the “UNITS” row from `output.csv`.
- Saves high-quality `.png` and `.html` plot files to the specified output directory.

---

## 🧠 Training the ML Model

`train_hoc_model.py` builds and trains a neural network model to predict `P_peak_MPa` or to perform inverse HOC estimation.

### Run Example

```bash
python train_hoc_model.py
```

All runtime options are defined in the `CONFIG` dictionary at the top of the file.

### Configuration Highlights

| Key | Description |
|------|-------------|
| `INPUT_CSV` | Path to consolidated output.csv from CEA. |
| `ARTIFACT_DIR` | Directory to save trained model artifacts. |
| `TRAIN_FUELS`, `VAL_FUELS`, `OMIT_FUELS` | Control dataset splits by fuel. |
| `FEAT_*` | Enable or disable input features such as P₀, m, Q, or molecular weight. |
| `TARGET_RATIO` | Use normalized target (P_peak/P0) or absolute P_peak. |
| `RUN_VAL_INVERSION` | Runs HOC inversion for validation fuels (estimates heat of combustion). |

### Outputs

| File | Description |
|------|-------------|
| `model.pt` | Trained PyTorch model weights. |
| `scalers.npz` | Saved input/output normalization parameters. |
| `metrics.json` | Training loss and accuracy metrics. |
| `external_val_metrics.csv` | Cross-fuel validation results. |
| `inversion_<fuel>.json` | Optional HOC inversion output per validation fuel. |

---

## 🧩 Recommended Workflow Summary

1. Edit **fuels.json** to include all fuels of interest.
2. Prepare **CEA-Input-Params.csv** with desired pressures and masses.
3. Run **cea_uv_batch.py** to generate results.
4. Plot with **plot_outputs.py**.
5. Train and evaluate the ML model with **train_hoc_model.py**.

---

## ⚙️ Environment Requirements

- **Python ≥ 3.9**
- **NASA CEA** (FCEA2.exe) — Windows executable required for simulation.
- **Python Libraries:**
  ```bash
  pip install numpy pandas torch matplotlib plotly
  ```

Optional (for YAML fuels):
```bash
pip install pyyaml
```

---

## 📚 Tips and Troubleshooting

- Ensure `CEA_CMD` environment variable or `--cea-cmd` flag points to your FCEA2.exe.
- If results seem truncated or empty, check the `collected_listings/` folder for detailed CEA output.
- Units row in the CSV must remain as the **first data row**; do not remove it.
- Use short labels for fuels (≤8 characters) to avoid issues with CEA filename limits.

---

**Author:**  
*(c) 2025 – NASA CEA Batch + ML Toolkit by J.L. Micus and contributors*  
**License:** MIT

