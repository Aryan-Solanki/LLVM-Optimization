"""Matplotlib charts for the experiment results.

Each chart shows ONE measure per optimization level (no dual axes).  The O0
baseline is drawn in neutral grey, the optimized levels in one accent colour,
and every bar carries its exact value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")                      # headless: works without a display
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import font_manager

# Tokens from docs/design-system.md (Geist): mute baseline, blue accent, ink text.
BASELINE_COLOR = "#8f8f8f"
ACCENT_COLOR = "#0070f3"
INK = "#171717"
MUTED = "#8f8f8f"
GRID = "#f2f2f2"
HAIRLINE = "#ebebeb"


def _font_family() -> list:
    """Geist if installed, then the substitutes DESIGN.md names, then Matplotlib's default."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in ("Geist", "Inter", "Arial") if f in installed] + ["DejaVu Sans"]

# (column, y-axis label, title, filename, note)
CHARTS = [
    ("ir_instructions", "LLVM IR instructions", "Optimization level vs LLVM IR instruction count",
     "ir_instructions.png", "Counted from .ll files (target-independent IR)"),
    ("asm_instructions", "Assembly instructions", "Optimization level vs assembly instruction count",
     "asm_instructions.png", "Counted from .s files (machine instructions)"),
    ("exe_size_bytes", "Executable size (bytes)", "Optimization level vs executable size",
     "exe_size.png", "Includes startup/runtime code added by the linker"),
    ("compile_time_s", "Compilation time (s)", "Optimization level vs compilation time",
     "compile_time.png", "Wall-clock source -> assembly, median of repeats; noisy for tiny programs"),
]


def _fmt(value: float, column: str) -> str:
    if pd.isna(value):
        return "n/a"
    if column == "compile_time_s":
        return f"{value:.3f}"
    return f"{int(round(value)):,}"


def plot_metric(table: pd.DataFrame, column: str, ylabel: str, title: str, note: str = ""):
    """Return a Matplotlib figure with one bar per optimization level."""
    levels = list(table["opt_level"])
    values = [float(v) if not pd.isna(v) else 0.0 for v in table[column]]
    colors = [BASELINE_COLOR if lv == "O0" else ACCENT_COLOR for lv in levels]

    plt.rcParams["font.family"] = _font_family()
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    bars = ax.bar(levels, values, color=colors, width=0.5)
    top = max(values) if values and max(values) > 0 else 1.0
    for bar, raw in zip(bars, table[column]):
        ax.annotate(
            _fmt(raw, column),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="medium", color=INK,
        )
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xlabel("Optimization level (O0 = baseline, grey)", color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=INK, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(colors=MUTED, length=0)
    ax.tick_params(axis="x", colors=INK)
    if note:
        fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def generate_all(table: pd.DataFrame, results_dir: Path) -> Dict[str, Path]:
    """Save the four required charts as PNGs; return ``{column: path}``."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for column, ylabel, title, filename, note in CHARTS:
        fig = plot_metric(table, column, ylabel, title, note)
        path = results_dir / filename
        fig.savefig(path)
        plt.close(fig)
        out[column] = path
    return out
