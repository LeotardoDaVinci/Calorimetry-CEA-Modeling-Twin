#!/usr/bin/env python3
"""
Forward model (P_peak) and configurable fuel-split + filtering + downsampling + (optional) HOC inversion.

What this script does
---------------------
1) Reads the consolidated CEA dataset (output.csv) with columns:
     fuel, mw_g_mol, mass_fuel_g, P0_MPa, hoc_kj_g, total_Q_kJ, P_peak_MPa
   (first data row may contain units; we skip non-numeric rows)
   - NOTE: 'mw_g_mol' is optional unless CONFIG["FEAT_MW"] = True.

2) Lets you choose *exactly* which fuels to train on, which fuels to validate on,
   and which fuels to omit entirely (not used at all). You can also randomly
   downsample per fuel and restrict rows by numeric bounds (e.g., total_Q_kJ in [1,5]).

3) Builds a forward model f: (P0, m, Q [, helpers, mw]) -> P_peak (or ratio y=P_peak/P0)
   using a compact MLP with standardization, early stopping, progress prints.

4) Reports metrics on:
   - Internal validation split within the TRAIN subset (for early stopping feedback).
   - External validation set composed of VAL_FUELS only (aggregate + per-fuel metrics).

5) (Optional) HOC inversion routine for a set of points per validation fuel:
   from measured {(P0_i, m_i, P_peak_meas_i)}, estimate HOC by minimizing
   sum_i ( P_meas_i - f(P0_i, m_i, Q_i = m_i * HOC) )^2 via golden-section search.

6) Saves artifacts:
   - model weights:        model.pt
   - scalers & config:     scalers.npz, config.json
   - training metrics:     metrics.json
   - external val metrics: external_val_metrics.csv / external_val_summary.json
   - optional inversion:   inversion_<fuel>.json and per-point CSVs

Dependencies
------------
- Python 3.9+
- numpy, pandas, torch (PyTorch)

(We intentionally avoid scikit-learn to keep the script portable and self-contained.)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ==========================
# =======  CONFIG  =========
# ==========================
CONFIG = {
    # ---- Paths ----
    "INPUT_CSV": r"C:\Users\joemi\OneDrive\Combustion_Lab\Projects\CEA\ML_Code\CEA_Output\output.csv",
    "ARTIFACT_DIR": r"C:\Users\joemi\OneDrive\Combustion_Lab\Projects\CEA\ML_Artifacts",

    # ---- Fuel selection ----
    # Any fuel listed here is removed before any other split.
    "OMIT_FUELS": [],      # e.g. ["acetone", "paraffin"]; [] means omit none
    # If TRAIN_FUELS is empty, we take "all remaining minus VAL_FUELS".
    "TRAIN_FUELS": [],              # e.g. ["ethanol", "methanol", "starch"]
    # These fuels are NOT used for training; we only evaluate on them externally.
    "VAL_FUELS": ["starch"],  # e.g. ["methanol","ethanol"]; [] means no external validation

    # ---- Downsampling (random per fuel) ----
    # If None or <=0, use all rows. Otherwise, cap to this many rows per fuel (random).
    "TRAIN_DOWNSAMPLE_PER_FUEL": None,  # e.g. 500
    "VAL_DOWNSAMPLE_PER_FUEL": 200,     # e.g. 200 rows per validation fuel

    # ---- Dataset filters / bounds (apply BEFORE splitting) ----
    # Map: column_name -> [min, max]; use None for open bounds
    # Example: only keep rows where 1.0 <= total_Q_kJ <= 5.0
    "FILTERS": {
        # "total_Q_kJ": [5, 34],
        # "P0_MPa": [0.5, 4.0],
        # "mw_g_mol": [10.0, 500.0],
    },

    # ---- Model target / features ----
    # Use normalized target y = P_peak / P0 (True) or raw P_peak (False)
    "TARGET_RATIO": True,

    # Which features to feed (booleans gate whether they are included)
    # Base features:
    "FEAT_P0": True,
    "FEAT_MASS": True,          # m (g)
    "FEAT_Q": True,             # total_Q_kJ (from CSV) — recommended
    "FEAT_MW": False,            # NEW: use molecular weight (mw_g_mol) as a feature
    # Optional helpers (set False to omit):
    "FEAT_vspec": False,         # requires bomb volume & T0 to compute n_O2 & v_spec
    "FEAT_q_per_total_mass": True,  # Q / (m_fuel + m_O2)

    # ---- Log-space transforms (optional) ----
    "LOG_SPACE": {
        "ENABLED": False,              # master switch
        "FEATURES": ["P0_MPa", "mass_fuel_g", "total_Q_kJ"],  # cols to log if present
        "TARGET": False,               # log-transform the training target (y)
        "EPS": 1e-6,                   # small offset to keep logs finite
    },

    # ---- Bomb constants for helpers (must match how you ran CEA) ----
    "BOMB_VOLUME_mL": 240.0,    # free volume in mL
    "T0_K": 303.5,              # initial temperature, K

    # ---- Train/val split (internal, within TRAIN subset) ----
    "VAL_SPLIT_FRACTION": 0.2,  # random holdout fraction (stratified by fuel)
    "RANDOM_SEED": 42,

    # ---- Model selector & CrossMLP hyperparams ----
    "MODEL": "cross_mlp",        # "mlp" (default behavior) or "cross_mlp"
    "CROSS_DEPTH": 3,            # number of cross layers (2–4 is typical)
    "DEEP_HIDDEN": 256,          # width of the deep MLP tower inside CrossMLP

    # ---- Model hyperparams ----
    "HIDDEN_SIZES": [64, 64],   # small MLP
    "DROPOUT_P": 0,
    "ACTIVATION": "silu",       # "relu" | "silu" | "tanh"
    "LEARNING_RATE":3e-3,
    "BATCH_SIZE": 256,
    "EPOCHS": 600,
    "WEIGHT_DECAY": 1e-6,
    "EARLY_STOP_PATIENCE": 30,  # epochs with no val improvement before early stop

    "EDGE_WEIGHTING": {
    "ENABLED": False,
    "Z_THRESH": 1.0,     # how “far” from mean (in stds) counts as edge
    "WEIGHT_EDGE": 2.0,  # multiply loss for edge points
    "COLUMNS": ["HOC_kJ_g"],  # which columns to consider for edge detection
    },

    # ---- Numeric stability ----
    "EPS": 1e-8,

    # ---- LOFO (leave-one-fuel-out) eval (optional) ----
    "RUN_LOFO": False,

    # ---- HOC inversion on external validation fuels (optional) ----
    "RUN_VAL_INVERSION": True,
    "VAL_INVERT_POINTS_PER_FUEL": 200,       # sample up to this many points per VAL fuel for inversion
    "HOC_SEARCH_BOUNDS": (10.0, 50.0),       # kJ/g lower/upper bounds for 1-D search
    "HOC_SEARCH_TOL": 1e-3,                  # tolerance for golden-section

    # ---- CUDA / dataloader ----
    "USE_AMP": True,          # Mixed precision on CUDA
    "NUM_WORKERS": 0,         # DataLoader workers (start with 0 on Windows)
    "PIN_MEMORY": True,       # Pin host memory for faster H2D copies (CUDA only)
}

def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using CUDA: {name}")
        torch.backends.cudnn.benchmark = True  # speeds up on constant shapes
        return torch.device("cuda")
    print("[INFO] Using CPU")
    return torch.device("cpu")


# Physical constants for helper features
R_J_PER_MOLK: float = 8.314462618
MW_O2_G_PER_MOL: float = 31.998

# ==========================
# ===== Data & Utils  ======
# ==========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the consolidated CSV and keep only rows with valid numerics.
    Expected columns (minimum):
      fuel, mass_fuel_g, P0_MPa, hoc_kj_g, total_Q_kJ, P_peak_MPa
    Optional:
      mw_g_mol   (required ONLY if CONFIG['FEAT_MW'] is True)

    The first data row may contain units (strings); we drop non-numeric rows for the numeric columns.
    """
    df = pd.read_csv(csv_path)
    required_min = ["fuel", "mass_fuel_g", "P0_MPa", "hoc_kj_g", "total_Q_kJ", "P_peak_MPa"]
    missing = [c for c in required_min if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce numeric columns; include mw_g_mol if present
    num_cols = ["mass_fuel_g", "P0_MPa", "hoc_kj_g", "total_Q_kJ", "P_peak_MPa"]
    if "mw_g_mol" in df.columns:
        num_cols.append("mw_g_mol")

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop non-numeric / NaN rows (for numeric cols)
    before = len(df)
    df = df.dropna(subset=["mass_fuel_g", "P0_MPa", "hoc_kj_g", "total_Q_kJ", "P_peak_MPa"])
    after = len(df)
    if after < before:
        print(f"[INFO] Dropped {before - after} non-numeric rows (likely the units row).")

    # Keep sensible values
    df = df[(df["mass_fuel_g"] > 0) & (df["P0_MPa"] > 0) & (df["P_peak_MPa"] > 0)]
    df["fuel"] = df["fuel"].astype(str).str.strip()
    df = df.reset_index(drop=True)
    return df

def apply_filters(df: pd.DataFrame, filters: Dict[str, Tuple[Optional[float], Optional[float]]]) -> pd.DataFrame:
    """
    Apply numeric bounds per column. filters[col] = (min, max); None means open bound.
    """
    if not filters:
        return df
    out = df.copy()
    kept = np.ones(len(out), dtype=bool)
    for col, rng in filters.items():
        if col not in out.columns:
            print(f"[WARN] FILTERS: column '{col}' not found; skipping.")
            continue
        mn, mx = None, None
        if rng is not None:
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                mn, mx = rng
            else:
                print(f"[WARN] FILTERS: value for '{col}' should be [min, max]; got {rng}. Skipping.")
                continue
        if mn is not None:
            kept &= (out[col] >= float(mn))
        if mx is not None:
            kept &= (out[col] <= float(mx))
    before = len(out)
    out = out[kept].reset_index(drop=True)
    print(f"[INFO] FILTERS kept {len(out)}/{before} rows.")
    return out

def downsample_per_fuel(df: pd.DataFrame, fuels: List[str], per_fuel_cap: Optional[int], seed: int) -> pd.DataFrame:
    """
    Randomly cap rows per fuel to 'per_fuel_cap' if given.
    """
    if per_fuel_cap is None or per_fuel_cap <= 0:
        return df
    rng = np.random.default_rng(seed)
    frames = []
    for f in fuels:
        sub = df[df["fuel"] == f]
        if len(sub) > per_fuel_cap:
            idx = rng.choice(sub.index.values, size=per_fuel_cap, replace=False)
            sub = sub.loc[idx]
        frames.append(sub)
    return pd.concat(frames, axis=0).reset_index(drop=True)

def compute_helpers(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Optionally augment df with helper features:
      v_spec (m^3/kg), q_per_total_mass (kJ/kg_total)
    Assumes constant bomb volume and T0 across dataset (as in your runs).
    """
    if not (cfg["FEAT_vspec"] or cfg["FEAT_q_per_total_mass"]):
        return df

    V_m3 = cfg["BOMB_VOLUME_mL"] * 1e-6
    T0 = cfg["T0_K"]

    # n_O2 depends only on P0 (if V and T0 fixed)
    P0_Pa = df["P0_MPa"].values * 1e6
    n_O2_mol = (P0_Pa * V_m3) / (R_J_PER_MOLK * T0)
    m_O2_g = n_O2_mol * MW_O2_G_PER_MOL

    if cfg["FEAT_vspec"]:
        m_total_kg = (df["mass_fuel_g"].values + m_O2_g) / 1000.0
        v_spec = V_m3 / np.maximum(m_total_kg, cfg["EPS"])
        df["v_spec_m3_per_kg"] = v_spec

    if cfg["FEAT_q_per_total_mass"]:
        m_total_kg = (df["mass_fuel_g"].values + m_O2_g) / 1000.0
        q_per_total_mass = df["total_Q_kJ"].values / np.maximum(m_total_kg, cfg["EPS"])
        df["q_per_total_mass_kJ_per_kg"] = q_per_total_mass

    return df

def build_feature_matrix(df: pd.DataFrame, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      X : (N, D) features (log-transformed per CONFIG['LOG_SPACE']['FEATURES'] if enabled)
      y : (N,) target in chosen space (ratio or raw), possibly log-transformed if CONFIG['LOG_SPACE']['TARGET']
      y_raw : (N,) P_peak (MPa) raw values (ALWAYS natural units, never log)
      feat_names: list of feature names used
    """
    log_cfg = cfg.get("LOG_SPACE", {})
    log_on  = bool(log_cfg.get("ENABLED", False))
    log_eps = float(log_cfg.get("EPS", 1e-6))
    log_feats = set(log_cfg.get("FEATURES", []))

    def maybe_log(name: str, arr: np.ndarray) -> np.ndarray:
        if log_on and (name in log_feats):
            return np.log(arr.astype(np.float32) + log_eps)
        return arr.astype(np.float32)

    feat = []
    names = []

    if cfg["FEAT_P0"]:
        if "P0_MPa" not in df.columns:
            raise ValueError("Column 'P0_MPa' missing from dataframe.")
        arr = maybe_log("P0_MPa", df["P0_MPa"].values)
        feat.append(arr.reshape(-1, 1))
        names.append("P0_MPa")

    if cfg["FEAT_MASS"]:
        if "mass_fuel_g" not in df.columns:
            raise ValueError("Column 'mass_fuel_g' missing from dataframe.")
        arr = maybe_log("mass_fuel_g", df["mass_fuel_g"].values)
        feat.append(arr.reshape(-1, 1))
        names.append("mass_fuel_g")

    if cfg["FEAT_Q"]:
        if "total_Q_kJ" not in df.columns:
            raise ValueError("Column 'total_Q_kJ' missing from dataframe.")
        arr = maybe_log("total_Q_kJ", df["total_Q_kJ"].values)
        feat.append(arr.reshape(-1, 1))
        names.append("total_Q_kJ")

    if cfg.get("FEAT_MW", False):
        if "mw_g_mol" not in df.columns:
            raise ValueError("CONFIG['FEAT_MW'] is True, but column 'mw_g_mol' is missing in the dataset.")
        arr = maybe_log("mw_g_mol", df["mw_g_mol"].values)
        feat.append(arr.reshape(-1, 1))
        names.append("mw_g_mol")

    if cfg["FEAT_vspec"] and ("v_spec_m3_per_kg" in df.columns):
        arr = maybe_log("v_spec_m3_per_kg", df["v_spec_m3_per_kg"].values)
        feat.append(arr.reshape(-1, 1))
        names.append("v_spec_m3_per_kg")

    if cfg["FEAT_q_per_total_mass"] and ("q_per_total_mass_kJ_per_kg" in df.columns):
        arr = maybe_log("q_per_total_mass_kJ_per_kg", df["q_per_total_mass_kJ_per_kg"].values)
        feat.append(arr.reshape(-1, 1))
        names.append("q_per_total_mass_kJ_per_kg")

    if not feat:
        raise ValueError("No features selected! Enable at least one FEAT_* in CONFIG.")

    X = np.concatenate(feat, axis=1).astype(np.float32)

    # Target construction
    y_raw = df["P_peak_MPa"].values.astype(np.float32)
    if cfg["TARGET_RATIO"]:
        y_nat = (y_raw / (df["P0_MPa"].values.astype(np.float32) + cfg["EPS"])).astype(np.float32)
    else:
        y_nat = y_raw

    # Optional log on target
    if log_on and bool(log_cfg.get("TARGET", False)):
        y = np.log(y_nat + log_eps).astype(np.float32)
    else:
        y = y_nat

    return X, y, y_raw, names

# Simple scaler (avoid sklearn)
@dataclass
class StandardScaler:
    """
    Simple sklearn-like scaler:
      - construct with no arguments
      - call .fit(X) to compute stats
      - .transform / .inverse_transform use stored stats
    """
    def __init__(self, eps: float = 1e-8):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, X: np.ndarray) -> "StandardScaler":
        m = X.mean(axis=0)
        s = X.std(axis=0)
        s = np.where(s < self.eps, 1.0, s)
        self.mean_ = m
        self.std_  = s
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler used before fit().")
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler used before fit().")
        return X * self.std_ + self.mean_

# ==========================
# =======  Model  ==========
# ==========================

def _act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation {name}")

class CrossLayer(nn.Module):
    """
    DCN-v2 style cross layer: x_{l+1} = x0 * (x_l @ w) + b + x_l
    """
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x0, x):
        # x, x0: (B, D). (x @ w) -> (B,)
        cross_term = (x @ self.w)[:, None]   # (B,1)
        return x0 * cross_term + self.b + x  # broadcast over D


class CrossMLP(nn.Module):
    """
    Combines a shallow 'cross' tower (explicit feature crosses) with a deep MLP tower.
    Concatenate their outputs and predict a scalar.
    """
    def __init__(self, in_dim: int, cross_depth: int = 3, hidden: int = 128, act: str = "silu", dropout_p: float = 0.0):
        super().__init__()
        A = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[act]

        # Cross tower
        self.cross = nn.ModuleList([CrossLayer(in_dim) for _ in range(cross_depth)])

        # Deep tower (very light)
        deep_layers: List[nn.Module] = [
            nn.LayerNorm(in_dim), A(), nn.Linear(in_dim, hidden), A()
        ]
        if dropout_p > 0:
            deep_layers.append(nn.Dropout(dropout_p))
        deep_layers += [nn.Linear(hidden, hidden), A()]
        self.deep = nn.Sequential(*deep_layers)

        # Head on concatenated [cross_out, deep_out]
        self.head = nn.Linear(in_dim + hidden, 1)

    def forward(self, x):
        # Cross tower
        x0 = x
        xc = x
        for layer in self.cross:
            xc = layer(x0, xc)
        # Deep tower
        h = self.deep(x)
        # Predict
        return self.head(torch.cat([xc, h], dim=-1)).squeeze(-1)

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout_p: float, act: str):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), _act(act)]
            if dropout_p > 0:
                layers += [nn.Dropout(dropout_p)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
    
class WeightedNumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
        self.X = X
        self.y = y
        if w is None:
            self.w = np.ones_like(y, dtype=np.float32)
        else:
            self.w = w.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # return weights as third item
        return self.X[idx], self.y[idx], self.w[idx]


def train_val_split_by_fuel(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split by fuel: pick a fraction of rows within each fuel for validation.
    Returns boolean masks (train_mask, val_mask)
    """
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    for fuel, sub in df.groupby("fuel"):
        idx = sub.index.values
        n = len(idx)
        n_val = max(1, int(round(val_frac * n)))
        val_idx = rng.choice(idx, size=n_val, replace=False)
        val_mask[val_idx] = True
    train_mask = ~val_mask
    return train_mask, val_mask

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: List[str],
    cfg: Dict,
    val_mask: Optional[np.ndarray] = None,
    df_rows_for_metrics: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Train an MLP with early stopping; print per-epoch progress.
    Returns a dict with model, scalers, metrics, etc.
    """
    torch.manual_seed(cfg["RANDOM_SEED"])
    device = get_device()
    use_amp = bool(cfg.get("USE_AMP", True) and device.type == "cuda")

    # ------------------------- Split --------------------------
    if val_mask is None:
        n = len(X)
        idx = np.arange(n)
        rng = np.random.default_rng(cfg["RANDOM_SEED"])
        val_size = int(round(cfg["VAL_SPLIT_FRACTION"] * n))
        # ensure at least 1 val sample when possible, but not all
        if n > 1:
            val_size = min(max(1, val_size), n - 1)
        else:
            val_size = 1
        val_sel = rng.choice(idx, size=val_size, replace=False)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_sel] = True
    train_mask = ~val_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    # ---------------------- Scaling ---------------------------
    xsc = StandardScaler(eps=cfg.get("EPS", 1e-8)).fit(X_train)
    ysc = StandardScaler(eps=cfg.get("EPS", 1e-8)).fit(y_train.reshape(-1, 1))

    X_train_s = xsc.transform(X_train).astype(np.float32)
    X_val_s   = xsc.transform(X_val).astype(np.float32)
    y_train_s = ysc.transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s   = ysc.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)

    # ------------------------------------------------------------
    # Edge weighting setup (optional)
    # ------------------------------------------------------------
    ew_cfg = cfg.get("EDGE_WEIGHTING", {})
    sample_w_train = None
    sample_w_val = None

    if ew_cfg.get("ENABLED", False):
        # Only use columns that exist in df_rows_for_metrics
        cols = [c for c in ew_cfg.get("COLUMNS", []) if c in df_rows_for_metrics.columns]
        if cols:
            # Training subset
            df_tr = df_rows_for_metrics.iloc[train_mask]

            # Compute mean and std for z-score normalization (avoid div by zero)
            mu = df_tr[cols].mean()
            sd = df_tr[cols].std().replace(0, 1.0)

            # Define function to compute weights for any dataframe split
            def weights(dfpart):
                # z-scores across specified columns
                z = ((dfpart[cols] - mu) / sd).abs()
                # "edge" = any column beyond threshold
                is_edge = (z > float(ew_cfg.get("Z_THRESH", 1.0))).any(axis=1)
                # Initialize weights = 1
                w = np.ones(len(dfpart), dtype=np.float32)
                # Heavier weight for edge samples
                w[is_edge.values] = float(ew_cfg.get("WEIGHT_EDGE", 2.0))
                return w

            # Compute weights for training and validation splits
            sample_w_train = weights(df_tr)
            sample_w_val = weights(df_rows_for_metrics.iloc[val_mask])

            # ------------------------------------------------------------
            # Diagnostic print: summarize how many samples are edge-weighted
            # ------------------------------------------------------------
            if sample_w_train is not None:
                frac_edge = (sample_w_train > 1.0).mean()
                mean_w = sample_w_train.mean()
                print(f"[EDGE] mean train weight = {mean_w:.3f} | fraction edge = {frac_edge:.3f}")

    

    # ---------------------- DataLoaders -----------------------
    pin = bool(cfg.get("PIN_MEMORY", True) and device.type == "cuda")
    nw  = int(cfg.get("NUM_WORKERS", 0))

    use_weights = (sample_w_train is not None)

    if use_weights:
        train_ds = WeightedNumpyDataset(X_train_s, y_train_s, sample_w_train)
        val_ds   = WeightedNumpyDataset(X_val_s,   y_val_s,   sample_w_val)
    else:
        train_ds = NumpyDataset(X_train_s, y_train_s)
        val_ds   = NumpyDataset(X_val_s,   y_val_s)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True,
        drop_last=False, pin_memory=pin, num_workers=nw
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False,
        drop_last=False, pin_memory=pin, num_workers=nw
    )


    # ----------------------- Model/Opt ------------------------
    if cfg.get("MODEL", "mlp").lower() == "cross_mlp":
        model = CrossMLP(
            in_dim=X.shape[1],
            cross_depth=int(cfg.get("CROSS_DEPTH", 3)),
            hidden=int(cfg.get("DEEP_HIDDEN", 128)),
            act=cfg.get("ACTIVATION", "silu"),
            dropout_p=float(cfg.get("DROPOUT_P", 0.0)),
        ).to(device)
    else:
        model = MLPRegressor(
            in_dim=X.shape[1],
            hidden=cfg["HIDDEN_SIZES"],
            dropout_p=cfg["DROPOUT_P"],
            act=cfg["ACTIVATION"],
        ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["LEARNING_RATE"],
        weight_decay=cfg["WEIGHT_DECAY"],
    )
    loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="none")  # Huber loss per sample
    # loss_fn = nn.MSELoss(reduction="none")  # per-sample loss; we’ll reduce manually
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_state = None
    patience = cfg["EARLY_STOP_PATIENCE"]
    epochs_no_improve = 0

    print(f"[INFO] Device: {device.type.upper()} | AMP: {use_amp}")
    print(f"[INFO] Training on {len(train_ds)} samples; validating on {len(val_ds)} samples.")

    # ===================== Train Loop =====================
    for epoch in range(1, cfg["EPOCHS"] + 1):

        # -------- TRAIN PHASE --------
        model.train()
        # We'll accumulate a *true* epoch loss:
        #   - If weighted: sum_i (w_i * mse_i) / sum_i (w_i)S
        #   - If unweighted: sum_i (mse_i) / N
        tr_num, tr_den = 0.0, 0.0

        for batch in train_loader:
            # Unpack batch; weighted loader yields (X, y, w), unweighted yields (X, y)
            if use_weights and len(batch) == 3:
                xb, yb, wb = batch
                wb = wb.to(device, non_blocking=True)
            else:
                xb, yb = batch
                wb = None

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(xb)                      # shape [B]
                per_sample = loss_fn(pred, yb)        # shape [B], MSE per item

                # Loss used for backprop (normalized so LR is stable)
                if wb is not None:
                    batch_den = wb.sum() + 1e-8
                    loss = (per_sample * wb).sum() / batch_den
                else:
                    loss = per_sample.mean()

            # Backward + step (AMP aware)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            # ---- Epoch loss accounting (for reporting only) ----
            if wb is not None:
                # Weighted numerator/denominator for epoch average
                tr_num += float((per_sample.detach() * wb.detach()).sum().item())
                tr_den += float(batch_den.item())
            else:
                tr_num += float(per_sample.detach().sum().item())
                tr_den += float(xb.shape[0])

        tr_loss = tr_num / max(tr_den, 1e-12)

        # -------- VALIDATION PHASE --------
        model.eval()
        va_num, va_den = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch; val loader mirrors train loader
                if use_weights and len(batch) == 3:
                    xb, yb, wb = batch
                    wb = wb.to(device, non_blocking=True)
                else:
                    xb, yb = batch
                    wb = None

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred = model(xb)
                    per_sample = loss_fn(pred, yb)  # shape [B]

                    if wb is not None:
                        batch_den = wb.sum() + 1e-8
                        loss = (per_sample * wb).sum() / batch_den
                    else:
                        loss = per_sample.mean()

                # ---- Epoch val loss accounting ----
                if wb is not None:
                    va_num += float((per_sample * wb).sum().item())
                    va_den += float(batch_den.item())
                else:
                    va_num += float(per_sample.sum().item())
                    va_den += float(xb.shape[0])

        va_loss = va_num / max(va_den, 1e-12)

        # -------- LOGGING & EARLY STOP --------
        print(f"Epoch {epoch:4d} | train loss: {tr_loss:.6f} | val loss: {va_loss:.6f}")

        if va_loss + 1e-12 < best_val:
            best_val = va_loss
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val_mse": best_val,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EARLY STOP] No val improvement for {patience} epochs (best val MSE={best_val:.6f}).")
                break

    # -------------------- Load best weights -------------------
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # ===== Final predictions on internal split (for reporting) =====
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_train_s, dtype=torch.float32, device=device)
        yt_pred_s = model(xt).cpu().numpy()
        xv = torch.tensor(X_val_s, dtype=torch.float32, device=device)
        yv_pred_s = model(xv).cpu().numpy()

    # Use the *fitted* ysc from above—do not re-create it.
    yt_pred = ysc.inverse_transform(yt_pred_s.reshape(-1, 1)).ravel()
    yv_pred = ysc.inverse_transform(yv_pred_s.reshape(-1, 1)).ravel()

    # If target was trained in log-space, un-log here
    log_cfg = cfg.get("LOG_SPACE", {})
    if bool(log_cfg.get("ENABLED", False)) and bool(log_cfg.get("TARGET", False)):
        log_eps = float(log_cfg.get("EPS", 1e-6))
        yt_pred = np.exp(yt_pred) - log_eps
        yv_pred = np.exp(yv_pred) - log_eps
        # numerical safety
        yt_pred = np.maximum(yt_pred, 0.0)
        yv_pred = np.maximum(yv_pred, 0.0)

    # -------------------- Human metrics (MPa) ------------------
    if df_rows_for_metrics is None:
        raise RuntimeError("df_rows_for_metrics must be provided to compute human-readable metrics.")
    df_used = df_rows_for_metrics

    idx_all = np.arange(len(df_used))
    idx_val = idx_all[val_mask]
    idx_tr  = idx_all[~val_mask]
    train_rows = df_used.iloc[idx_tr].reset_index(drop=True)
    val_rows   = df_used.iloc[idx_val].reset_index(drop=True)

    def to_p_peak(y_like: np.ndarray, df_rows: pd.DataFrame) -> np.ndarray:
        if cfg["TARGET_RATIO"]:
            return y_like * df_rows["P0_MPa"].values
        else:
            return y_like

    p_train_pred = to_p_peak(yt_pred, train_rows)
    p_train_true = train_rows["P_peak_MPa"].values
    p_val_pred   = to_p_peak(yv_pred, val_rows)
    p_val_true   = val_rows["P_peak_MPa"].values

    metrics = {
        "train_RMSE_P_MPa": rmse(p_train_true, p_train_pred),
        "train_MAE_P_MPa":  mae(p_train_true, p_train_pred),
        "val_RMSE_P_MPa":   rmse(p_val_true, p_val_pred),
        "val_MAE_P_MPa":    mae(p_val_true, p_val_pred),
        "best_val_MSE_target_space": best_val,
        "epochs_trained": epoch,
        "best_epoch": (best_state or {}).get("epoch", epoch),
    }
    print("[RESULT] Internal (TRAIN subset) metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Store the *fitted* scalers
    bundle = {
        "model": model,
        "xscaler": xsc,
        "yscaler": ysc,
        "feat_names": feat_names,
        "metrics": metrics,
        "device": device.type,
        "best_state": best_state,
        "log_space": cfg.get("LOG_SPACE", {}).copy(),
    }
    return bundle


def predict_p_peak_MPa(model_bundle: Dict, X_raw: np.ndarray, df_rows: pd.DataFrame) -> np.ndarray:
    """
    Predict P_peak (MPa) for given raw feature matrix rows (unscaled).
    """
    model: MLPRegressor = model_bundle["model"]
    xsc: StandardScaler = model_bundle["xscaler"]
    ysc: StandardScaler = model_bundle["yscaler"]
    device = torch.device(model_bundle["device"])

    # Safety: ensure scalers are fitted
    if xsc.mean_ is None or ysc.mean_ is None:
        raise RuntimeError("Scalers in model_bundle are not fitted. Re-train or rebuild the bundle.")

    Xs = xsc.transform(X_raw).astype(np.float32)
    with torch.no_grad():
        xt = torch.tensor(Xs, dtype=torch.float32, device=device)
        yhat_s = model(xt).cpu().numpy().ravel()

    yhat = ysc.inverse_transform(yhat_s.reshape(-1, 1)).ravel()

    # If target was trained in log-space, un-log it
    log_cfg = model_bundle.get("log_space", {})
    if bool(log_cfg.get("ENABLED", False)) and bool(log_cfg.get("TARGET", False)):
        log_eps = float(log_cfg.get("EPS", 1e-6))
        yhat = np.exp(yhat) - log_eps
        yhat = np.maximum(yhat, 0.0)

    if CONFIG["TARGET_RATIO"]:
        P0 = df_rows["P0_MPa"].values.astype(np.float32)
        return yhat * P0
    else:
        return yhat


# ==========================
# === Leave-one-fuel-out ===
# ==========================

def leave_one_fuel_out_eval(df: pd.DataFrame, cfg: Dict, feat_names: List[str]) -> pd.DataFrame:
    fuels = sorted(df["fuel"].unique())
    rows = []
    for f in fuels:
        test_idx = (df["fuel"].values == f)
        train_idx = ~test_idx
        if train_idx.sum() < 10 or test_idx.sum() < 3:
            continue
        X_all, y_all, _, _ = build_feature_matrix(df, cfg)
        val_mask = test_idx  # treat held-out fuel as "validation"
        bundle = train_model(X_all, y_all, feat_names, cfg, val_mask=val_mask, df_rows_for_metrics=df)
        val_rows = df.iloc[val_mask].reset_index(drop=True)
        X_val = X_all[val_mask]
        p_pred = predict_p_peak_MPa(bundle, X_val, val_rows)
        p_true = val_rows["P_peak_MPa"].values
        rows.append({
            "fuel": f,
            "n_test": int(val_mask.sum()),
            "RMSE_P_MPa": rmse(p_true, p_pred),
            "MAE_P_MPa": mae(p_true, p_pred),
        })
        print(f"[LOFO] fuel={f:20s} | n={int(val_mask.sum()):4d} | RMSE={rows[-1]['RMSE_P_MPa']:.4f} MPa | MAE={rows[-1]['MAE_P_MPa']:.4f} MPa")
    return pd.DataFrame(rows)

# ==========================
# ====== HOC Inversion =====
# ==========================

def golden_section_minimize(func, a, b, tol=1e-3, max_iter=200):
    """
    Minimize a 1-D function on [a, b] with golden-section search.
    Returns (x_min, f_min, iters)
    """
    gr = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = func(c)
    fd = func(d)
    it = 0
    while abs(b - a) > tol and it < max_iter:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) / gr
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) / gr
            fd = func(d)
        it += 1
    x_min = (a + b) / 2
    f_min = func(x_min)
    return x_min, f_min, it

def estimate_hoc_for_points(model_bundle: Dict, df_pts: pd.DataFrame, cfg: Dict) -> Tuple[float, float]:
    """
    Estimate HOC (kJ/g) from df_pts rows having columns P0_MPa, mass_fuel_g, P_peak_MPa (measured).
    If CONFIG['FEAT_MW'] is True, df_pts MUST include 'mw_g_mol' (it will be passed through as a feature).
    """
    if len(df_pts) == 0:
        raise ValueError("No points provided for inversion.")

    # Minimal temporary frame for features; Q is set via candidate HOC in the objective.
    needed = ["P0_MPa", "mass_fuel_g", "P_peak_MPa"]
    if cfg.get("FEAT_MW", False):
        if "mw_g_mol" not in df_pts.columns:
            raise ValueError("CONFIG['FEAT_MW']=True but df_pts does not contain 'mw_g_mol' for inversion.")
        needed.append("mw_g_mol")

    tmp = df_pts[needed].copy()
    tmp["fuel"] = "UNKNOWN"
    tmp["hoc_kj_g"] = 0.0
    tmp["total_Q_kJ"] = 0.0

    V_m3 = cfg["BOMB_VOLUME_mL"] * 1e-6
    T0 = cfg["T0_K"]
    P0_Pa = tmp["P0_MPa"].values * 1e6
    n_O2_mol = (P0_Pa * V_m3) / (R_J_PER_MOLK * T0)
    m_O2_g = n_O2_mol * MW_O2_G_PER_MOL

    v_spec_const = None
    if cfg["FEAT_vspec"]:
        m_total_kg_base = (tmp["mass_fuel_g"].values + m_O2_g) / 1000.0
        v_spec_const = (V_m3 / np.maximum(m_total_kg_base, cfg["EPS"])).astype(np.float32)

    def obj(hoc_kj_g: float) -> float:
        df2 = tmp.copy()
        total_Q = df2["mass_fuel_g"].values * hoc_kj_g
        df2["total_Q_kJ"] = total_Q
        if cfg["FEAT_q_per_total_mass"]:
            m_total_kg = (df2["mass_fuel_g"].values + m_O2_g) / 1000.0
            df2["q_per_total_mass_kJ_per_kg"] = total_Q / np.maximum(m_total_kg, cfg["EPS"])
        if cfg["FEAT_vspec"]:
            df2["v_spec_m3_per_kg"] = v_spec_const
        Xp, _, _, _ = build_feature_matrix(df2, cfg)
        p_pred = predict_p_peak_MPa(model_bundle, Xp, df2)
        p_meas = df_pts["P_peak_MPa"].values
        return float(np.mean((p_meas - p_pred) ** 2))

    lo, hi = cfg["HOC_SEARCH_BOUNDS"]
    hoc_hat, fmin, iters = golden_section_minimize(obj, lo, hi, tol=cfg["HOC_SEARCH_TOL"])
    print(f"[INVERT] HOC* = {hoc_hat:.4f} kJ/g | objective={fmin:.6f} | iters={iters}")
    return hoc_hat, fmin

# ==========================
# ========= Main ===========
# ==========================

if __name__ == "__main__":
    cfg = CONFIG
    ensure_dir(cfg["ARTIFACT_DIR"])

   # 1) Load + FILTER
    df = load_dataset(cfg["INPUT_CSV"])

    # If FEAT_MW is requested, enforce presence now (clear error early)
    if cfg.get("FEAT_MW", False) and "mw_g_mol" not in df.columns:
        raise ValueError("CONFIG['FEAT_MW'] is True, but 'mw_g_mol' column is missing from INPUT_CSV.")

    df = apply_filters(df, cfg.get("FILTERS", {}))

    # # Convert P_peak_MPa to overpressure (ΔP = P_peak - P0)
    # if "P_peak_MPa" in df.columns and "P0_MPa" in df.columns:
    #     df["P_peak_MPa"] = df["P_peak_MPa"] - df["P0_MPa"]
    #     print("[INFO] Converted P_peak_MPa to overpressure (ΔP = P_peak - P0).")

    print(f"[INFO] Loaded/filtered: {len(df)} rows, {df['fuel'].nunique()} fuels.")

    # 2) Apply helper features
    df = compute_helpers(df, cfg)

    # 3) Assemble fuel sets
    all_fuels = sorted(df["fuel"].unique().tolist())

    omit = set([f for f in cfg.get("OMIT_FUELS", []) if f in all_fuels])
    val_fuels = set([f for f in cfg.get("VAL_FUELS", []) if f in all_fuels and f not in omit])

    if cfg.get("TRAIN_FUELS"):
        train_fuels = set([f for f in cfg["TRAIN_FUELS"] if f in all_fuels and f not in omit and f not in val_fuels])
    else:
        train_fuels = set([f for f in all_fuels if f not in omit and f not in val_fuels])

    print(f"[SPLIT] OMIT_FUELS: {sorted(list(omit))}")
    print(f"[SPLIT] TRAIN_FUELS: {sorted(list(train_fuels))}")
    print(f"[SPLIT] VAL_FUELS (external): {sorted(list(val_fuels))}")

    # 4) Slice dataframes
    df_train_full = df[df["fuel"].isin(train_fuels)].reset_index(drop=True)
    df_valext_full = df[df["fuel"].isin(val_fuels)].reset_index(drop=True)

    # 5) Downsample if requested
    if cfg.get("TRAIN_DOWNSAMPLE_PER_FUEL"):
        df_train_full = downsample_per_fuel(df_train_full, sorted(list(train_fuels)), int(cfg["TRAIN_DOWNSAMPLE_PER_FUEL"]), cfg["RANDOM_SEED"])
        print(f"[INFO] Train downsampled to {len(df_train_full)} rows total.")
    if cfg.get("VAL_DOWNSAMPLE_PER_FUEL"):
        df_valext_full = downsample_per_fuel(df_valext_full, sorted(list(val_fuels)), int(cfg["VAL_DOWNSAMPLE_PER_FUEL"]), cfg["RANDOM_SEED"])
        print(f"[INFO] External val downsampled to {len(df_valext_full)} rows total.")

    # If no training rows, bail early
    if len(df_train_full) < 5:
        raise RuntimeError("No training rows after filtering/selection. Adjust CONFIG.")

    # 6) Build training features and internal split
    X_train_all, y_train_all, _, feat_names = build_feature_matrix(df_train_full, cfg)
    print(f"[INFO] Features: {feat_names} (D={X_train_all.shape[1]}); target={'ratio (P/P0)' if cfg['TARGET_RATIO'] else 'P_peak (MPa)'} | training rows: {len(df_train_full)}")

    train_mask, val_mask = train_val_split_by_fuel(df_train_full, cfg["VAL_SPLIT_FRACTION"], cfg["RANDOM_SEED"])
    bundle = train_model(X_train_all, y_train_all, feat_names, cfg, val_mask=val_mask, df_rows_for_metrics=df_train_full)

    # 7) Save core artifacts
    torch.save(bundle["model"].state_dict(), os.path.join(cfg["ARTIFACT_DIR"], "model.pt"))
    np.savez(os.path.join(cfg["ARTIFACT_DIR"], "scalers.npz"),
             x_mean=bundle["xscaler"].mean_, x_std=bundle["xscaler"].std_,
             y_mean=bundle["yscaler"].mean_, y_std=bundle["yscaler"].std_)
    with open(os.path.join(cfg["ARTIFACT_DIR"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(cfg["ARTIFACT_DIR"], "metrics.json"), "w") as f:
        json.dump(bundle["metrics"], f, indent=2)
    print(f"[OK] Saved artifacts to {cfg['ARTIFACT_DIR']}")

    # 8) Optional LOFO (on the POST-FILTERED df)
    if cfg.get("RUN_LOFO", False):
        print("[INFO] Running Leave-One-Fuel-Out evaluation (on filtered data)...")
        lofo_df = leave_one_fuel_out_eval(df, cfg, feat_names)
        lofo_path = os.path.join(cfg["ARTIFACT_DIR"], "lofo_metrics.csv")
        lofo_df.to_csv(lofo_path, index=False)
        if len(lofo_df):
            print(f"[OK] LOFO summary saved: {lofo_path}")
            print(f"       Mean RMSE (MPa): {lofo_df['RMSE_P_MPa'].mean():.4f} | Mean MAE (MPa): {lofo_df['MAE_P_MPa'].mean():.4f}")
        else:
            print("[WARN] LOFO skipped or not enough data per fuel.")

    # 9) External validation on VAL_FUELS (pressure metrics)
    if len(df_valext_full) > 0:
        print("[INFO] Computing external validation metrics on VAL_FUELS...")
        X_valext, _, _, _ = build_feature_matrix(df_valext_full, cfg)
        p_pred = predict_p_peak_MPa(bundle, X_valext, df_valext_full)
        p_true = df_valext_full["P_peak_MPa"].values
        rows = []
        rows.append({"fuel": "__ALL__", "n": int(len(df_valext_full)),
                     "RMSE_P_MPa": rmse(p_true, p_pred), "MAE_P_MPa": mae(p_true, p_pred)})
        for f in sorted(list(val_fuels)):
            sub = df_valext_full[df_valext_full["fuel"] == f]
            if len(sub) == 0:
                continue
            Xs, _, _, _ = build_feature_matrix(sub, cfg)
            pp = predict_p_peak_MPa(bundle, Xs, sub)
            pt = sub["P_peak_MPa"].values
            rows.append({"fuel": f, "n": int(len(sub)),
                         "RMSE_P_MPa": rmse(pt, pp), "MAE_P_MPa": mae(pt, pp)})
            print(f"[VAL-EXT] fuel={f:20s} | n={len(sub):4d} | RMSE={rows[-1]['RMSE_P_MPa']:.4f} MPa | MAE={rows[-1]['MAE_P_MPa']:.4f} MPa")

        df_ext = pd.DataFrame(rows)
        ext_csv = os.path.join(cfg["ARTIFACT_DIR"], "external_val_metrics.csv")
        df_ext.to_csv(ext_csv, index=False)
        with open(os.path.join(cfg["ARTIFACT_DIR"], "external_val_summary.json"), "w") as f:
            json.dump({
                "aggregate": {"RMSE_P_MPa": float(df_ext.loc[df_ext["fuel"]=="__ALL__", "RMSE_P_MPa"].values[0]),
                              "MAE_P_MPa": float(df_ext.loc[df_ext["fuel"]=="__ALL__", "MAE_P_MPa"].values[0])},
                "per_fuel": df_ext[df_ext["fuel"] != "__ALL__"].to_dict(orient="records")
            }, f, indent=2)
        print(f"[OK] External validation metrics saved: {ext_csv}")

    # 10) Optional: HOC inversion for each validation fuel
    if cfg.get("RUN_VAL_INVERSION", False) and len(df_valext_full) > 0:
        print("[INFO] Running HOC inversion on external validation fuels...")
        rng = np.random.default_rng(cfg["RANDOM_SEED"])
        k_per = int(cfg.get("VAL_INVERT_POINTS_PER_FUEL", 200))
        for f in sorted(list(val_fuels)):
            sub = df_valext_full[df_valext_full["fuel"] == f].reset_index(drop=True)
            if len(sub) == 0:
                continue
            k = min(k_per, len(sub))
            idx = rng.choice(sub.index.values, size=k, replace=False)
            cols = ["P0_MPa", "mass_fuel_g", "P_peak_MPa", "hoc_kj_g"]
            if cfg.get("FEAT_MW", False):
                cols.append("mw_g_mol")
            df_pts = sub.loc[idx, cols].reset_index(drop=True)
            hoc_true = float(np.median(sub["hoc_kj_g"].values)) if len(sub["hoc_kj_g"].dropna()) else float("nan")

            hoc_hat, obj_min = estimate_hoc_for_points(bundle, df_pts, cfg)
            abs_err = abs(hoc_hat - hoc_true) if np.isfinite(hoc_true) else float("nan")
            rel_err = abs_err / abs(hoc_true) if np.isfinite(hoc_true) and abs(hoc_true) > 1e-12 else float("nan")
            print(f"[INV] fuel={f:20s} | n={k:4d} | HOC*={hoc_hat:.4f} kJ/g | true≈{hoc_true:.4f} | abs={abs_err:.4f} | rel={rel_err:.2%} | obj={obj_min:.6f}")

            # Save per-fuel inversion summary
            with open(os.path.join(cfg["ARTIFACT_DIR"], f"inversion_{f}.json"), "w") as fp:
                json.dump({
                    "fuel": f,
                    "n_points": int(k),
                    "hoc_est_kJ_g": float(hoc_hat),
                    "hoc_true_kJ_g": float(hoc_true),
                    "hoc_abs_err_kJ_g": float(abs_err),
                    "hoc_rel_err": float(rel_err),
                    "objective_min": float(obj_min),
                }, fp, indent=2)

    print("[DONE]")
