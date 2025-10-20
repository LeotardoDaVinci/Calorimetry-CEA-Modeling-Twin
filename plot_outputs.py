#!/usr/bin/env python3
r"""
plot_cea_results.py

Load consolidated CEA results (with a UNITS row), clean them, and provide
publication-ready plotting utilities:

- 2D scatter plots (matplotlib)
- 3D scatter plots (Plotly)

Grouping:
- Points are grouped and colored by the 'fuel' column (first column).

Input CSV (first row after header is UNITS):
  Columns (expected at minimum):
    fuel, mass_fuel_g, P0_MPa, hoc_kj_g, total_Q_kJ, P_peak_MPa
  Optional (will be parsed if present):
    mw_g_mol

Examples
--------
# Example 2D: P0_MPa vs P_peak_MPa
python plot_cea_results.py "C:\path\to\output.csv" --out-dir "C:\plots" \
  --plot2d --x P0_MPa --y P_peak_MPa

# Example 3D: total_Q_kJ vs P_peak_MPa vs P0_MPa
python plot_cea_results.py "C:\path\to\output.csv" --out-dir "C:\plots" \
  --plot3d --x total_Q_kJ --y P_peak_MPa --z P0_MPa

You can also generate both in one go by providing both --plot2d and --plot3d.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot as plotly_save

import plotly.io as pio
pio.renderers.default = "browser"


# ---------------------- Loading & Cleaning ----------------------

EXPECTED_MIN_COLUMNS = [
    "fuel", "mass_fuel_g", "P0_MPa", "hoc_kj_g", "total_Q_kJ", "P_peak_MPa"
]
OPTIONAL_COLUMNS = ["mw_g_mol"]  # NEW: molecular weight is optional

@dataclass
class ResultsData:
    df: pd.DataFrame
    units: Dict[str, str]  # best-effort units map (may be empty strings)
    fuels: List[str]       # sorted unique fuel names


def _looks_like_units_row(row: pd.Series) -> bool:
    """
    Heuristic: the UNITS row usually contains strings like 'MPa', 'g', 'kJ/g', 'g/mol', etc.,
    or empty strings; almost never pure numbers across the board.
    """
    tokens = [str(v).strip().lower() for v in row.values]
    blob = " ".join(tokens)
    # expanded hints to include g/mol for MW column
    hints = ("mpa", "g", "kj", "kJ", "kJ/g", "kj/g", "m^3/kg", "g/mol", "dimensionless")
    if any(h.lower() in blob for h in hints):
        return True
    # If most cells are non-numeric text, likely units
    numeric_mask = []
    for v in row.values:
        try:
            float(str(v))
            numeric_mask.append(True)
        except Exception:
            numeric_mask.append(False)
    return sum(numeric_mask) <= (len(numeric_mask) // 3)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns that look numeric into float; leave others as-is.
    """
    out = df.copy()
    for c in out.columns:
        if c == "fuel":
            continue
        series = out[c]
        as_num = pd.to_numeric(series, errors="coerce")
        # keep numeric if we have at least one non-NaN numeric value
        if as_num.notna().any():
            out[c] = as_num
    return out


def load_results_csv(path: Path) -> ResultsData:
    """
    Load the results CSV written by the batch CEA code (with UNITS row).
    Returns a typed container with the cleaned DataFrame, units map, and fuel list.
    MW is optional; if present, it will be parsed and available for plotting.
    """
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if raw.empty:
        raise ValueError("Results CSV is empty.")

    # First data row after header is UNITS (usually). Detect it heuristically.
    units_row_idx = None
    if len(raw) >= 1 and _looks_like_units_row(raw.iloc[0]):
        units_row_idx = 0

    units: Dict[str, str] = {}
    if units_row_idx is not None:
        for c in raw.columns:
            units[c] = str(raw.iloc[units_row_idx].get(c, "") or "").strip()
        df = raw.iloc[units_row_idx + 1 :].copy()
    else:
        df = raw.copy()
        units = {c: "" for c in df.columns}

    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure required minimum columns exist; allow optional extras (like mw_g_mol)
    for c in EXPECTED_MIN_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found in CSV.")

    # Strip whitespace, coerce numerics (this will also coerce mw_g_mol if present)
    df["fuel"] = df["fuel"].astype(str).str.strip()
    df = _coerce_numeric_columns(df)
    df = df.reset_index(drop=True)

    # Clean obvious non-physical rows (don’t require mw_g_mol to exist)
    df = df.dropna(subset=["mass_fuel_g", "P0_MPa", "P_peak_MPa"])
    df = df[(df["mass_fuel_g"] > 0) & (df["P0_MPa"] > 0) & (df["P_peak_MPa"] > 0)].reset_index(drop=True)

    # Convert P_peak_MPa to overpressure (ΔP = P_peak - P0) without renaming the column
    if "P_peak_MPa" in df.columns and "P0_MPa" in df.columns:
        df["P_peak_MPa"] = df["P_peak_MPa"] - df["P0_MPa"]
        # (Optional) clarify units label if present
        if "P_peak_MPa" in units:
            units["P_peak_MPa"] = (units.get("P_peak_MPa") or "MPa").strip()
            # Keep unit as MPa; it's now overpressure in MPa

    fuels = sorted([f for f in df["fuel"].dropna().unique().tolist() if f != ""])
    return ResultsData(df=df, units=units, fuels=fuels)


# ---------------------- Styling & Colors ----------------------

@dataclass
class PubStyle:
    font_family: str = "serif"
    font_serif: Tuple[str, ...] = ("Times New Roman", "Times", "DejaVu Serif")
    base_font_size: float = 10.0
    axis_label_font_size: float = 10.0
    tick_font_size: float = 8.0
    legend_font_size: float = 8.0
    axis_line_width: float = 1.2
    grid_line_width: float = 0.6
    marker_size: float = 45.0  # points^2 in matplotlib
    figure_dpi: int = 300

    def apply_matplotlib(self) -> None:
        mpl.rcParams.update({
            "font.family": self.font_family,
            "font.serif": list(self.font_serif),
            "font.size": self.base_font_size,
            "axes.linewidth": self.axis_line_width,
            "axes.labelsize": self.axis_label_font_size,
            "xtick.labelsize": self.tick_font_size,
            "ytick.labelsize": self.tick_font_size,
            "legend.fontsize": self.legend_font_size,
            "grid.linewidth": self.grid_line_width,
            "figure.dpi": self.figure_dpi,
        })


def assign_fuel_colors(fuels: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    """
    Deterministic color assignment for fuels using matplotlib tab palette(s).
    """
    cmap = plt.get_cmap("tab20")
    colors: Dict[str, Tuple[float, float, float]] = {}
    for i, fuel in enumerate(fuels):
        colors[fuel] = cmap(i % cmap.N)
    return colors


# ---------------------- Plotting Functions ----------------------

def make_axis_label(name: str, units: Dict[str, str]) -> str:
    u = (units or {}).get(name, "").strip()
    return f"{name} ({u})" if u else name


def plot_scatter_2d(
    data: ResultsData,
    x: str,
    y: str,
    out_path: Path,
    *,
    title: Optional[str] = None,
    style: Optional[PubStyle] = None,
    alpha: float = 0.95,
    edgewidth: float = 0.6,
    figsize: Tuple[float, float] = (4.0, 3.0),
) -> Path:
    """
    2D scatter, grouped by 'fuel'. Saves to out_path (PNG).
    MW is supported if chosen as x or y; parsing stays robust either way.
    """
    style = style or PubStyle()
    style.apply_matplotlib()
    colors = assign_fuel_colors(data.fuels)

    # Drop rows with missing coords
    dfp = data.df.dropna(subset=[x, y, "fuel"]).copy()

    fig, ax = plt.subplots(figsize=figsize)
    for fuel in data.fuels:
        sub = dfp[dfp["fuel"] == fuel]
        if sub.empty:
            continue
        ax.scatter(
            sub[x].to_numpy(),
            sub[y].to_numpy(),
            s=style.marker_size,
            facecolors=colors[fuel],
            edgecolors="black",
            linewidths=edgewidth,
            alpha=alpha,
            label=fuel,
        )

    ax.set_xlabel(make_axis_label(x, data.units))
    ax.set_ylabel(make_axis_label(y, data.units))
    if title:
        ax.set_title(title, pad=8)
    ax.grid(True, which="major", alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12)
    leg = ax.legend(title="Fuel", frameon=False, ncols=1)
    if leg is not None and leg.get_title():
        leg.get_title().set_fontsize(style.legend_font_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return out_path


def plot_scatter_3d(
    data: ResultsData,
    x: str,
    y: str,
    z: str,
    out_html: Path,
    *,
    title: Optional[str] = None,
    marker_size: int = 6,
    opacity: float = 0.95,
) -> Path:
    """3D scatter with Plotly. Saves and shows interactively.
    MW is supported if used as any axis; parsing stays robust either way.
    """
    colors = assign_fuel_colors(data.fuels)

    def rgba_to_hex(rgba: Tuple[float, float, float, float] | Tuple[float, float, float]) -> str:
        r, g, b = rgba[:3]
        return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"

    dfp = data.df.dropna(subset=[x, y, z, "fuel"]).copy()
    traces = []
    for fuel in data.fuels:
        sub = dfp[dfp["fuel"] == fuel]
        if sub.empty:
            continue
        color = rgba_to_hex(colors[fuel])
        traces.append(go.Scatter3d(
            x=sub[x],
            y=sub[y],
            z=sub[z],
            mode="markers",
            name=fuel,
            marker=dict(size=marker_size, opacity=opacity, line=dict(width=0.5, color="#222"), color=color),
            text=sub["fuel"],
        ))

    layout = go.Layout(
        title=title or "",
        scene=dict(
            xaxis=dict(title=make_axis_label(x, data.units)),
            yaxis=dict(title=make_axis_label(y, data.units)),
            zaxis=dict(title=make_axis_label(z, data.units)),
        ),
        legend=dict(title="Fuel"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig = go.Figure(data=traces, layout=layout)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    plotly_save(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")
    fig.show()
    return out_html


# ---------------------- CLI Driver ----------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Plot CEA results (2D matplotlib, 3D Plotly), grouped by fuel.")
    p.add_argument("results_csv", type=Path, help="Path to output.csv created by the batch runner.")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory to save plots.")
    p.add_argument("--plot2d", action="store_true", help="Create a 2D scatter.")
    p.add_argument("--plot3d", action="store_true", help="Create a 3D scatter.")
    p.add_argument("--x", type=str, default=None, help="X-axis column name.")
    p.add_argument("--y", type=str, default=None, help="Y-axis column name (2D/3D).")
    p.add_argument("--z", type=str, default=None, help="Z-axis column name (3D).")
    p.add_argument("--title", type=str, default=None, help="Optional plot title.")
    args = p.parse_args()

    data = load_results_csv(args.results_csv)

    # Provide defaults if user just wants quick examples
    x = args.x
    y = args.y
    z = args.z

    if args.plot2d and (x is None or y is None):
        # Example 2D default: P0_MPa vs P_peak_MPa
        x = x or "P0_MPa"
        y = y or "P_peak_MPa"

    if args.plot3d and (x is None or y is None or z is None):
        # Example 3D default: total_Q_kJ vs P_peak_MPa vs P0_MPa
        x = x or "total_Q_kJ"
        y = y or "P_peak_MPa"
        z = z or "P0_MPa"

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot2d:
        png_path = out_dir / f"scatter2d_{x}_vs_{y}.png"
        plot_scatter_2d(
            data,
            x=x, y=y,
            out_path=png_path,
            title=args.title or f"{y} vs {x}",
        )
        print(f"[OK] 2D plot: {png_path}")

    if args.plot3d:
        html_path = out_dir / f"scatter3d_{x}_vs_{y}_vs_{z}.html"
        plot_scatter_3d(
            data,
            x=x, y=y, z=z,
            out_html=html_path,
            title=args.title or f"{y} vs {x} vs {z}",
        )
        print(f"[OK] 3D plot: {html_path}")

    if not args.plot2d and not args.plot3d:
        # If nothing requested, produce both defaults.
        png_path = out_dir / "scatter2d_P_peak_MPa_vs_P0_MPa.png"
        plot_scatter_2d(
            data,
            x="P0_MPa", y="P_peak_MPa",
            out_path=png_path,
            title="P_peak_MPa vs P0_MPa",
        )
        print(f"[OK] 2D plot (default): {png_path}")

        html_path = out_dir / "scatter3d_total_Q_kJ_vs_P_peak_MPa_vs_P0_MPa.html"
        plot_scatter_3d(
            data,
            x="total_Q_kJ", y="P_peak_MPa", z="P0_MPa",
            out_html=html_path,
            title="P_peak_MPa vs total_Q_kJ vs P0_MPa",
        )
        print(f"[OK] 3D plot (default): {html_path}")


if __name__ == "__main__":
    main()

    # python plot_outputs.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\CEA\ML_Code\CEA_Output\output.csv" --out-dir "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\CEA\ML_Code\CEA_Output\plots" --plot3d --x hoc_kj_g --y P0_MPa --z P_peak_MPa --title "P_peak vs P₀ vs HOC"
    # python plot_outputs.py "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\CEA\ML_Code\CEA_Output\output.csv" --out-dir "C:\Users\joemi\OneDrive\Combustion_Lab\Projects\CEA\ML_Code\CEA_Output\plots" --plot3d --x total_Q_kJ --y P0_MPa --z P_peak_MPa --title "P_peak vs P₀ vs Total Q"
