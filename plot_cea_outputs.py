#!/usr/bin/env python3
r"""
plot_cea_outputs.py

- Loads summary.csv (units row aware), builds publication-quality plots.
- Scatter plots of thermodynamic properties and species vs fuel mass.
- Grouped plots by initial pressure (P0_MPa).
- Extra: Overpressure vs Initial Pressure (scatter + grouped by fuel mass).

Usage:
  python3 plot_cea_outputs.py "C:\path\to\results\summary.csv" --out-dir "C:\path\to\plots"
"""

from __future__ import annotations
from dataclasses import dataclass

import argparse
import math
import re
import os, stat, time, shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


# ------------------------ Units detection ------------------------

_UNIT_HINTS = (
    "mpa", "kpa", "bar", "atm",
    "kJ/(kg*k)", "kJ/(kg·k)", "kJ/kg", "kJ mol", "kj", "j/",
    "kg/m^3", "kg/m3", "g", "kg", "m^3/kg", "m3/kg",
    "mole frac", "mass frac", "dimensionless",
)
_UNIT_CHARS = set("/()*^[]·_- ")

def _looks_like_unit_token(s: str) -> bool:
    if s is None:
        return True
    s = str(s).strip()
    if not s:
        return True
    low = s.lower()
    if any(h in low for h in _UNIT_HINTS):
        return True
    if any(c.isalpha() for c in low) and any(c in _UNIT_CHARS for c in low):
        return True
    if low.isalpha():
        return True
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s):
        return False
    return False

def _detect_units_row(raw: pd.DataFrame) -> Tuple[Optional[int], Dict[str, str]]:
    if raw.empty:
        return None, {}
    rl = raw.columns[0] if "run_label" not in raw.columns else "run_label"
    if rl in raw.columns:
        first_cell = str(raw.iloc[0].get(rl, "")).strip()
        if first_cell.upper() == "UNITS":
            units = raw.iloc[0].fillna("").to_dict()
            return 0, {k: ("" if (isinstance(v, float) and math.isnan(v)) else str(v)) for k, v in units.items()}
    r0 = raw.iloc[0]
    cols = list(raw.columns)
    if rl in cols:
        cols = [c for c in cols if c != rl]
    votes = 0
    total = 0
    for c in cols:
        val = r0.get(c, "")
        total += 1
        if _looks_like_unit_token(val):
            votes += 1
    if total > 0 and votes / total >= 0.6:
        units = r0.fillna("").to_dict()
        return 0, {k: ("" if (isinstance(v, float) and math.isnan(v)) else str(v)) for k, v in units.items()}
    return None, {}

# ------------------------ Plotting helpers & loader ------------------------

def _to_numpy(arr):
    if arr is None:
        return None
    if isinstance(arr, pd.Series):
        return arr.to_numpy()
    return np.asarray(arr)

def _compose_label(col_name: Optional[str], units: Dict[str, str], override: Optional[str], fallback: Optional[str]) -> str:
    if override:
        return override
    if col_name:
        u = (units or {}).get(col_name, "").strip()
        return rf"{col_name} ({u})" if u else col_name
    return fallback or ""

def load_summary_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"summary.csv not found: {path}")
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    units_row_idx, units = _detect_units_row(raw)
    if units_row_idx is not None:
        df = raw.iloc[units_row_idx + 1 :].copy()
    else:
        print("[WARN] No units row detected; proceeding without units.")
        df = raw.copy()
    if "run_label" in df.columns:
        df["run_label"] = df["run_label"].astype(str).str.strip()
        df["run_label"] = df["run_label"].str.replace(r"\.0$", "", regex=True)
    for col in df.columns:
        if col == "run_label":
            continue
        series = df[col].astype(str).replace({"—": "", "–": "", "nan": "", "None": "", "": np.nan})
        numeric = pd.to_numeric(series, errors="coerce")
        df[col] = numeric if numeric.notna().any() else series.where(series.notna(), None)
    df = df.reset_index(drop=True)
    units = {c: str(units.get(c, "")).strip() for c in df.columns}
    return df, units

# ------------------------ Parameter store ------------------------

class ParameterStore:
    def __init__(self, df: pd.DataFrame, units: Dict[str, str]):
        self._df = df
        self._units = units
        self._columns = list(df.columns)
        self._colmap: Dict[str, Any] = {c: df[c] for c in self._columns}
    def __getattr__(self, name: str):
        if name in self._colmap:
            return self._colmap[name]
        raise AttributeError(f"No parameter named '{name}'. Use .columns to list available names.")
    def __getitem__(self, name: str):
        return self._colmap[name]
    @property
    def df(self) -> pd.DataFrame: return self._df
    @property
    def units(self) -> Dict[str, str]: return self._units
    @property
    def columns(self) -> List[str]: return self._columns

# ------------------------ Pretty print ------------------------

def _is_numeric_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def simple_stats(series: pd.Series) -> Dict[str, Any]:
    if _is_numeric_dtype(series):
        s = series.astype(float)
        non_null = int(s.notna().sum())
        return {"non_null": non_null, "null": int(s.isna().sum()),
                "min": float(np.nanmin(s)) if non_null else None,
                "mean": float(np.nanmean(s)) if non_null else None,
                "max": float(np.nanmax(s)) if non_null else None}
    else:
        s = series.astype(str)
        non_null = int((s != "nan").sum())
        try:
            nunique = int(series.nunique(dropna=True))
            top = series.value_counts(dropna=True).head(3).index.tolist()
        except Exception:
            nunique, top = 0, []
        return {"non_null": non_null, "null": int(series.isna().sum()),
                "unique": nunique, "top_values": top}

def print_catalog(params: ParameterStore, only_cols: Optional[List[str]] = None) -> None:
    cols = params.columns if not only_cols else [c for c in only_cols if c in params.columns]
    header = f"{'parameter':<28} | {'unit':<16} | {'dtype':<10} | details"
    print(header); print("-" * len(header))
    for c in cols:
        unit = params.units.get(c, ""); dtype = str(params.df[c].dtype)
        stats = simple_stats(params.df[c])
        print(f"{c:<28} | {unit:<16} | {dtype:<10} | ", end="")
        if _is_numeric_dtype(params.df[c]):
            print(f"non-null={stats['non_null']}, null={stats['null']}, "
                  f"min={stats['min']}, mean={stats['mean']}, max={stats['max']}")
        else:
            top = ", ".join(map(str, stats.get("top_values", [])))
            print(f"non-null={stats['non_null']}, null={stats['null']}, unique={stats['unique']}"
                  + (f", top=[{top}]" if top else ""))

# ------------------------ Plotting core ------------------------

@dataclass
class PubStyle:
    font_family: str = "serif"
    font_serif: List[str] = ("Times New Roman", "Times", "DejaVu Serif")
    font_size: float = 8
    axis_label_font_size: float = 8
    axis_tick_font_size: float = 6
    legend_font_size: float = 7
    legend_handle_length: float = 2.0
    legend_handle_textpad: float = 0.6
    legend_borderpad: float = 0.3
    legend_markerscale: float = 1.0
    legend_handle_linewidth: float = 1.0
    line_width: float = 2.0
    axis_line_width: float = 1.2
    grid_line_width: float = 0.7
    scatter_size: float = 35.0
    column_width_mm: float = 90.0
    aspect_ratio: float = 4.0 / 3.0
    figure_dpi: int = 600
    extension: str = "tif"
    pad_inches: float = 0.05
    colormap_name: str = "viridis"
    @property
    def fig_w(self) -> float: return self.column_width_mm / 25.4
    @property
    def fig_h(self) -> float: return self.fig_w / self.aspect_ratio

def apply_pub_style(cfg: PubStyle) -> None:
    mpl.rcParams.update({
        "font.family": cfg.font_family,
        "font.serif": list(cfg.font_serif),
        "font.size": cfg.font_size,
        "axes.linewidth": cfg.axis_line_width,
        "axes.labelsize": cfg.axis_label_font_size,
        "legend.fontsize": cfg.legend_font_size,
        "xtick.labelsize": cfg.axis_tick_font_size,
        "ytick.labelsize": cfg.axis_tick_font_size,
        "grid.linewidth": cfg.grid_line_width,
        "text.usetex": True,
    })

def _discrete_colors(n: int, cmap_name: str):
    cmap = mpl.colormaps.get_cmap(cmap_name)
    if n <= 1: return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]

def _remove_readonly(func, path, exc):
    excvalue = exc[1]
    if isinstance(excvalue, PermissionError):
        try: os.chmod(path, stat.S_IWRITE)
        except Exception: pass
        func(path)
    else:
        raise

def ensure_clean_dir(out_dir: Path, clear: bool = False, retries: int = 8, wait: float = 0.35) -> Path:
    out_dir = Path(out_dir).resolve()
    if clear and out_dir.exists():
        for i in range(retries):
            try:
                shutil.rmtree(out_dir, onerror=_remove_readonly); break
            except PermissionError:
                time.sleep(wait * (2 ** i))
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = out_dir / f"session_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _axis_label(col: str, units: Dict[str, str], override_label: Optional[str]) -> str:
    if override_label: return override_label
    u = (units or {}).get(col, "").strip()
    return rf"{col} ({u})" if u else col

def _sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

def _format_group_value(gval) -> str:
    try:
        gv = float(gval)
        if abs(gv) >= 1e3 or (abs(gv) < 1e-2 and gv != 0): return f"{gv:.3e}"
        return f"{gv:g}"
    except Exception:
        return str(gval)

# -------- grouped lines (supports DF or direct arrays) --------

def plot_grouped_lines(
    params,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    group_col: Optional[str] = None,
    out_dir: Path = Path("."),
    *,
    x_data: Optional[object] = None,
    y_data: Optional[object] = None,
    group_data: Optional[object] = None,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    legend_title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    dpi: Optional[int] = None,
    ext: Optional[str] = None,
    cfg: Optional[PubStyle] = None,
    omit_col: Optional[str] = None,
    omit_series: Optional[object] = None,
    omit_values: Optional[List[object]] = None,
    omit_epsilon: float = 1e-12,
) -> Path:
    cfg = cfg or PubStyle(); apply_pub_style(cfg)
    ext = (ext or cfg.extension).lstrip("."); dpi = dpi or cfg.figure_dpi
    out_dir = Path(out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    using_arrays = (x_data is not None) or (y_data is not None) or (group_data is not None)

    if using_arrays:
        x_arr = _to_numpy(x_data) if x_data is not None else params.df[x_col].to_numpy()
        y_arr = _to_numpy(y_data) if y_data is not None else params.df[y_col].to_numpy()
        g_arr = _to_numpy(group_data) if group_data is not None else params.df[group_col].to_numpy()
        df_plot = pd.DataFrame({"x": x_arr, "y": y_arr, "g": g_arr})
        x_id = x_col or (x_name or "x"); y_id = y_col or (y_name or "y"); g_id = group_col or "group"
    else:
        assert params is not None and x_col in params.columns and y_col in params.columns and group_col in params.columns, "Missing columns."
        df_plot = params.df[[x_col, y_col, group_col]].rename(columns={x_col:"x", y_col:"y", group_col:"g"})
        x_id, y_id, g_id = x_col, y_col, group_col

    df_plot = df_plot.dropna(subset=["x","y","g"])

    if omit_values is not None:
        if omit_series is not None:
            base = _to_numpy(omit_series)
            if base is None or len(base) != len(df_plot): raise ValueError("omit_series must be parallel to data.")
            series = pd.Series(base, index=df_plot.index)
        elif omit_col and (params is not None) and (omit_col in params.columns):
            series = params.df[omit_col].reindex(df_plot.index)
        else:
            series = None
        if series is not None:
            try:
                col_num = pd.to_numeric(series, errors="coerce")
                num_vals, str_vals = [], []
                for v in (omit_values if isinstance(omit_values,(list,tuple,set)) else [omit_values]):
                    try: num_vals.append(float(v))
                    except Exception: str_vals.append(str(v))
                num_mask = np.zeros(len(df_plot), dtype=bool)
                if num_vals and col_num.notna().any():
                    arr = col_num.to_numpy(dtype=float, copy=False)
                    for ov in num_vals:
                        num_mask |= np.isclose(arr, ov, rtol=0.0, atol=omit_epsilon, equal_nan=False)
                str_mask = np.zeros(len(df_plot), dtype=bool)
                if str_vals:
                    str_mask = series.astype(str).isin(set(str_vals)).to_numpy()
                df_plot = df_plot.loc[~(num_mask | str_mask)]
            except Exception:
                df_plot = df_plot.loc[~series.astype(str).isin(set(map(str, omit_values)))]

    groups = sorted(df_plot["g"].dropna().unique(), key=lambda v: (str(type(v)), v)) if len(df_plot) else []
    colors = _discrete_colors(len(groups), cfg.colormap_name)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h), dpi=dpi)
    for i, gval in enumerate(groups):
        sub = df_plot[df_plot["g"] == gval].sort_values(by="x")
        if sub.empty: continue
        color = colors[i]
        ax.plot(
            sub["x"].to_numpy(),
            sub["y"].to_numpy(),
            label=_format_group_value(gval),
            lw=cfg.line_width,
            marker="o",
            ms=np.sqrt(cfg.scatter_size),
            mfc="white", mec=color, color=color,
        )

    units = getattr(params, "units", {}) if params is not None else {}
    ax.set_xlabel(_compose_label(x_col, units, x_label, x_name))
    ax.set_ylabel(_compose_label(y_col, units, y_label, y_name))
    if xlim and len(xlim)==2: ax.set_xlim(xlim[0], xlim[1])
    if ylim and len(ylim)==2: ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, which="major", alpha=0.25); ax.grid(True, which="minor", alpha=0.10); ax.minorticks_on()

    leg = ax.legend(title=legend_title, frameon=False, ncols=1,
                    handlelength=cfg.legend_handle_length, handletextpad=cfg.legend_handle_textpad,
                    borderpad=cfg.legend_borderpad, prop={"size": cfg.legend_font_size},
                    markerscale=cfg.legend_markerscale)
    if leg is not None:
        leg.get_title().set_fontsize(cfg.legend_font_size)
        handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", None) or leg.get_lines()
        for h in handles:
            if hasattr(h, "set_linewidth"): h.set_linewidth(cfg.legend_handle_linewidth)

    fname = f"grouped_{_sanitize_filename(y_id)}_vs_{_sanitize_filename(x_id)}_by_{_sanitize_filename(g_id)}.{ext}"
    fpath = out_dir / fname
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight", pad_inches=cfg.pad_inches)
    plt.close(fig)
    return fpath

# -------- scatter (supports DF or direct arrays) --------

def plot_scatter_xy(
    params,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    out_dir: Path = Path("."),
    *,
    x_data: Optional[object] = None,
    y_data: Optional[object] = None,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    dpi: Optional[int] = None,
    ext: Optional[str] = None,
    cfg: Optional[PubStyle] = None,
    omit_col: Optional[str] = None,
    omit_series: Optional[object] = None,
    omit_values: Optional[List[object]] = None,
    omit_epsilon: float = 1e-12,
    color: Optional[tuple] = None,
    marker: str = "o",
    edgewidth: float = 0.8,
    alpha: float = 1.0,
) -> Path:
    cfg = cfg or PubStyle(); apply_pub_style(cfg)
    ext = (ext or cfg.extension).lstrip("."); dpi = dpi or cfg.figure_dpi
    out_dir = Path(out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    using_arrays = (x_data is not None) or (y_data is not None)
    if using_arrays:
        x_arr = _to_numpy(x_data) if x_data is not None else params.df[x_col].to_numpy()
        y_arr = _to_numpy(y_data) if y_data is not None else params.df[y_col].to_numpy()
        df_plot = pd.DataFrame({"x": x_arr, "y": y_arr})
        x_id = x_col or (x_name or "x"); y_id = y_col or (y_name or "y")
    else:
        assert params is not None and x_col in params.columns and y_col in params.columns, "Missing columns."
        df_plot = params.df[[x_col, y_col]].rename(columns={x_col:"x", y_col:"y"})
        x_id, y_id = x_col, y_col

    df_plot = df_plot.dropna(subset=["x","y"])

    if omit_values is not None:
        if omit_series is not None:
            base = _to_numpy(omit_series)
            if base is None or len(base) != len(df_plot): raise ValueError("omit_series must be parallel to data.")
            series = pd.Series(base, index=df_plot.index)
        elif (omit_col is not None) and (params is not None) and (omit_col in params.columns):
            series = params.df[omit_col].reindex(df_plot.index)
        else:
            series = None
        if series is not None:
            try:
                col_num = pd.to_numeric(series, errors="coerce")
                num_vals, str_vals = [], []
                for v in (omit_values if isinstance(omit_values,(list,tuple,set)) else [omit_values]):
                    try: num_vals.append(float(v))
                    except Exception: str_vals.append(str(v))
                num_mask = np.zeros(len(df_plot), dtype=bool)
                if num_vals and col_num.notna().any():
                    arr = col_num.to_numpy(dtype=float, copy=False)
                    for ov in num_vals:
                        num_mask |= np.isclose(arr, ov, rtol=0.0, atol=omit_epsilon, equal_nan=False)
                str_mask = np.zeros(len(df_plot), dtype=bool)
                if str_vals:
                    str_mask = series.astype(str).isin(set(str_vals)).to_numpy()
                df_plot = df_plot.loc[~(num_mask | str_mask)]
            except Exception:
                df_plot = df_plot.loc[~series.astype(str).isin(set(map(str, omit_values)))]

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h), dpi=dpi)
    ax.scatter(
        df_plot["x"].to_numpy(), df_plot["y"].to_numpy(),
        s=cfg.scatter_size, marker=marker,
        facecolors=color if color is not None else "#36454F",
        edgecolors="black", linewidths=edgewidth, alpha=alpha,
    )

    units = getattr(params, "units", {}) if params is not None else {}
    ax.set_xlabel(_compose_label(x_col, units, x_label, x_name))
    ax.set_ylabel(_compose_label(y_col, units, y_label, y_name))
    if xlim and len(xlim)==2: ax.set_xlim(xlim[0], xlim[1])
    if ylim and len(ylim)==2: ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, which="major", alpha=0.25); ax.grid(True, which="minor", alpha=0.10); ax.minorticks_on()

    fname = f"scatter_{_sanitize_filename(y_id)}_vs_{_sanitize_filename(x_id)}.{ext}"
    fpath = out_dir / fname
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight", pad_inches=cfg.pad_inches)
    plt.close(fig)
    return fpath

# ------------------------ CLI driver ------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CEA EG results.")
    parser.add_argument("summary_csv", type=Path, help="Path to results/summary.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write plots (will be cleared)")
    args = parser.parse_args()

    df, units = load_summary_csv(args.summary_csv)
    params = ParameterStore(df, units)

    print("\nLoaded summary:")
    print(f"- file: {args.summary_csv}")
    print(f"- rows (excluding UNITS): {len(df)}")
    print(f"- columns: {len(params.columns)}\n")
    print("Available parameters (catalog):")
    print_catalog(params)

    out_base = ensure_clean_dir(args.out_dir, clear=True)

    style = PubStyle(
        column_width_mm=90.0, aspect_ratio=4/3,
        figure_dpi=600, extension="tif",
        font_family="serif", font_serif=("Times New Roman","Times","DejaVu Serif"),
        font_size=8, axis_label_font_size=8, axis_tick_font_size=6,
        line_width=2.0, axis_line_width=1.2, grid_line_width=0.7,
        scatter_size=35.0, colormap_name="viridis",
        legend_font_size=5, legend_handle_length=2.0, legend_handle_textpad=0.6,
        legend_borderpad=0.1, legend_markerscale=0.5, legend_handle_linewidth=0.5,
        pad_inches=0.05,
    )

    ext = style.extension.lstrip(".")

    # ---------------- Thermodynamic properties vs fuel mass ----------------
    thermo_targets = [
        ("T_K",         r"T (K)"),
        ("delta_P_MPa", r"Overpressure (MPa)"),
        ("H_kJ_kg",     r"H (kJ/kg)"),
    ]

    # 1) Scatter (no grouping), x = mass_fuel_g
    scatter_paths = []
    for y_col, y_lab in thermo_targets:
        fpath = plot_scatter_xy(
            params=params,
            x_col="mass_fuel_g", y_col=y_col,
            out_dir=out_base,
            x_label=None, y_label=y_lab,
            xlim=None, ylim=None,
            dpi=style.figure_dpi, ext=style.extension, cfg=style,
            omit_col="mass_fuel_g", omit_values=[0],
            marker="o", edgewidth=0.8, alpha=1.0,
        )
        scatter_paths.append(fpath)
    dest_scatter = out_base / "thermodynamic_outputs_scatter"
    dest_scatter.mkdir(parents=True, exist_ok=True)
    for f in scatter_paths:
        if f.exists(): shutil.move(str(f), str(dest_scatter / f.name))
        else: print(f"[WARN] Missing scatter plot: {f.name}")

    # 2) Grouped by initial pressure, x = mass_fuel_g
    grouped_paths = []
    for y_col, y_lab in thermo_targets:
        fpath = plot_grouped_lines(
            params=params,
            x_col="mass_fuel_g", y_col=y_col, group_col="P0_MPa",
            out_dir=out_base,
            legend_title=r"Initial Pressure (MPa)",
            x_label=None, y_label=y_lab,
            xlim=None, ylim=None,
            dpi=style.figure_dpi, ext=style.extension, cfg=style,
            omit_col="mass_fuel_g", omit_values=[0],
        )
        grouped_paths.append(fpath)
    dest_grouped = out_base / "thermodynamic_outputs_grouped_by_P0"
    dest_grouped.mkdir(parents=True, exist_ok=True)
    for f in grouped_paths:
        if f.exists(): shutil.move(str(f), str(dest_grouped / f.name))
        else: print(f"[WARN] Missing grouped plot: {f.name}")

    # ---------------- Species mole fractions vs fuel mass ----------------
    # Discover species: everything AFTER 'GAMMA', numeric & any non-empty
    all_cols = list(params.columns)
    gamma_idx = next((i for i, c in enumerate(all_cols) if str(c).strip().lower() == "gamma"), None)
    if gamma_idx is None:
        raise KeyError("Could not find a 'GAMMA' column to anchor species selection.")
    species = [
        c for c in all_cols[gamma_idx + 1 :]
        if pd.api.types.is_numeric_dtype(params.df[c]) and params.df[c].notna().any()
    ]

    # Species scatter
    sp_scatter = []
    for y in species:
        fpath = plot_scatter_xy(
            params=params,
            x_col="mass_fuel_g", y_col=y,
            out_dir=out_base,
            x_label=None, y_label=f"{y} (mole frac)",
            xlim=None, ylim=None,
            dpi=style.figure_dpi, ext=style.extension, cfg=style,
            omit_col="mass_fuel_g", omit_values=[0],
            marker="o", edgewidth=0.8, alpha=1.0,
        )
        sp_scatter.append(fpath)
    dest_sp_scatter = out_base / "mole_fractions_scatter"
    dest_sp_scatter.mkdir(parents=True, exist_ok=True)
    for f in sp_scatter:
        if f.exists(): shutil.move(str(f), str(dest_sp_scatter / f.name))
        else: print(f"[WARN] Missing species scatter: {f.name}")

    # Species grouped by initial pressure
    sp_grouped = []
    for y in species:
        fpath = plot_grouped_lines(
            params=params,
            x_col="mass_fuel_g", y_col=y, group_col="P0_MPa",
            out_dir=out_base,
            legend_title=r"Initial Pressure (MPa)",
            x_label=None, y_label=f"{y} (mole frac)",
            xlim=None, ylim=None,
            dpi=style.figure_dpi, ext=style.extension, cfg=style,
            omit_col="mass_fuel_g", omit_values=[0],
        )
        sp_grouped.append(fpath)
    dest_sp_grouped = out_base / "mole_fractions_grouped_by_P0"
    dest_sp_grouped.mkdir(parents=True, exist_ok=True)
    for f in sp_grouped:
        if f.exists(): shutil.move(str(f), str(dest_sp_grouped / f.name))
        else: print(f"[WARN] Missing species grouped plot: {f.name}")

    # ---------------- Extra: Overpressure vs Initial Pressure ----------------
    # A) Plain scatter: y = delta_P_MPa, x = P0_MPa
    op_scatter = plot_scatter_xy(
        params=params,
        x_col="P0_MPa", y_col="delta_P_MPa",
        out_dir=out_base,
        x_label=None, y_label=r"Overpressure (MPa)",
        xlim=None, ylim=None,
        dpi=style.figure_dpi, ext=style.extension, cfg=style,
        omit_col="mass_fuel_g", omit_values=[0],
        marker="o", edgewidth=0.8, alpha=1.0,
    )
    dest_op_scatter = out_base / "overpressure_vs_initial_scatter"
    dest_op_scatter.mkdir(parents=True, exist_ok=True)
    if Path(op_scatter).exists():
        shutil.move(str(op_scatter), str(dest_op_scatter / Path(op_scatter).name))
    else:
        print(f"[WARN] Missing overpressure scatter plot: {Path(op_scatter).name}")

    # B) Grouped by fuel mass: group_col = mass_fuel_g, x = P0_MPa, y = delta_P_MPa
    op_grouped = plot_grouped_lines(
        params=params,
        x_col="P0_MPa", y_col="delta_P_MPa", group_col="mass_fuel_g",
        out_dir=out_base,
        legend_title=r"Fuel mass (g)",
        x_label=None, y_label=r"Overpressure (MPa)",
        xlim=None, ylim=None,
        dpi=style.figure_dpi, ext=style.extension, cfg=style,
        omit_col="mass_fuel_g", omit_values=[0],
    )
    dest_op_grouped = out_base / "overpressure_vs_initial_grouped_by_fuel_mass"
    dest_op_grouped.mkdir(parents=True, exist_ok=True)
    if Path(op_grouped).exists():
        shutil.move(str(op_grouped), str(dest_op_grouped / Path(op_grouped).name))
    else:
        print(f"[WARN] Missing overpressure grouped plot: {Path(op_grouped).name}")

if __name__ == "__main__":
    main()
