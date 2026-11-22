"""Interactive Altair charts for the dashboard.

Rules (from docs/design-system.md + the dataviz method):
* one measure per chart, one y-axis, zero baseline;
* O0 is the reference and is drawn in neutral grey; optimized levels carry the
  series hue (IR = blue, assembly = violet, other measures = ink);
* thin bars with 4px rounded data-ends, recessive grid, values labelled
  directly, hover tooltip on every bar;
* text is always ink/mute, never the series color.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import altair as alt
import pandas as pd

from src.compiler import ALL_LEVELS

from . import theme

LEVEL_ORDER = list(ALL_LEVELS)


def _config(chart: alt.TopLevelMixin, height: int) -> alt.TopLevelMixin:
    return (
        chart.properties(height=height, width="container", background="transparent", padding={"left": 4, "right": 8, "top": 8, "bottom": 4})
        .configure(font="Geist")
        .configure_view(stroke=None)
        .configure_axis(
            labelFont="Geist", titleFont="Geist", labelFontSize=12, titleFontSize=12, titleFontWeight=400,
            labelColor=theme.MUTE, titleColor=theme.MUTE, gridColor=theme.HAIRLINE_SOFT, gridWidth=1,
            domainColor=theme.HAIRLINE, tickColor=theme.HAIRLINE, tickSize=0, labelPadding=8, titlePadding=12,
        )
        .configure_axisX(labelFont="Geist Mono", labelColor=theme.BODY, labelAngle=0, grid=False, domainColor=theme.HAIRLINE)
        .configure_axisY(domain=False, tickCount=4, labelFont="Geist Mono", labelFontSize=11)
        .configure_legend(
            labelFont="Geist", labelFontSize=12, labelColor=theme.BODY, titleFontSize=0, symbolType="square",
            symbolSize=90, orient="top", direction="horizontal", offset=4, padding=0, columnPadding=16, labelLimit=200,
        )
    )


def _fmt_value(column: str) -> Callable[[float], str]:
    if column == "compile_time_s":
        return lambda v: f"{v:.1f} ms"
    if column.endswith("_bytes"):
        return lambda v: f"{v:,.0f} B"
    return lambda v: f"{v:,.0f}"


def metric_chart(table: pd.DataFrame, column: str, color: str, y_title: str, height: int = 240) -> alt.TopLevelMixin:
    """Bar per optimization level for one measure."""
    scale = 1000.0 if column == "compile_time_s" else 1.0           # seconds -> ms for display
    fmt = _fmt_value(column)
    rows: List[Dict] = []
    base_val = None
    for lv, raw in zip(table["opt_level"], table[column]):
        if lv == "O0" and not pd.isna(raw):
            base_val = float(raw) * scale
    for lv, raw in zip(table["opt_level"], table[column]):
        missing = pd.isna(raw)
        v = 0.0 if missing else float(raw) * scale
        if lv == "O0":
            delta = "baseline"
        elif missing or not base_val:
            delta = "n/a"
        else:
            d = (v - base_val) / base_val * 100
            delta = f"{d:+.1f}%".replace("-", "−")
        label = "n/a" if missing else (f"{v:.1f}" if column == "compile_time_s" else f"{v:,.0f}")
        rows.append({"level": lv, "value": v, "label": label, "display": "n/a" if missing else fmt(v),
                     "delta": delta, "role": "Baseline (O0)" if lv == "O0" else "Optimized"})
    df = pd.DataFrame(rows)
    order = [lv for lv in LEVEL_ORDER if lv in set(df["level"])]
    hover = alt.selection_point(on="pointerover", fields=["level"], empty=False, clear="pointerout")

    base = alt.Chart(df).encode(
        x=alt.X("level:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.5, paddingOuter=0.25)),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format="~s"), scale=alt.Scale(zero=True, nice=True)),
        color=alt.condition(alt.datum.level == "O0", alt.value(theme.BASELINE), alt.value(color)),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.88)),
        tooltip=[
            alt.Tooltip("level:N", title="Level"),
            alt.Tooltip("display:N", title=y_title),
            alt.Tooltip("delta:N", title="vs O0"),
        ],
    ).add_params(hover)
    labels = base.mark_text(dy=-8, baseline="bottom", font="Geist", fontSize=12, fontWeight=500, color=theme.INK).encode(
        y=alt.Y("value:Q"), text="label:N",
    )
    # headroom so labels never clip
    top = float(df["value"].max() or 1.0)
    headroom = alt.Chart(pd.DataFrame({"y": [top * 1.14]})).mark_point(opacity=0).encode(y="y:Q")
    return _config(alt.layer(bars, labels, headroom), height)


def reduction_chart(table: pd.DataFrame, height: int = 260, show_legend: bool = False) -> alt.TopLevelMixin:
    """IR vs assembly reduction relative to O0, side by side per level."""
    opt = table[table["opt_level"] != "O0"]
    rows = []
    for _, r in opt.iterrows():
        for series, col in (("LLVM IR", "ir_reduction_pct"), ("Assembly", "asm_reduction_pct")):
            v = r[col]
            if pd.isna(v):
                continue
            v = float(v)
            rows.append({
                "level": r["opt_level"], "series": series, "value": v,
                "label": f"{v:.1f}%",
                "display": (f"−{v:.1f}%" if v > 0 else f"+{-v:.1f}%") + " instructions",
                "count": int(r["ir_instructions"] if series == "LLVM IR" else r["asm_instructions"]),
            })
    df = pd.DataFrame(rows, columns=["level", "series", "value", "label", "display", "count"])
    order = [lv for lv in LEVEL_ORDER if lv in set(df["level"])]
    color = alt.Color(
        "series:N", scale=alt.Scale(domain=["LLVM IR", "Assembly"], range=[theme.SERIES_IR, theme.SERIES_ASM]),
        legend=alt.Legend(title=None) if show_legend else None,
    )
    hover = alt.selection_point(on="pointerover", fields=["level", "series"], empty=False, clear="pointerout")
    base = alt.Chart(df).encode(
        x=alt.X("level:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.5, paddingOuter=0.25)),
        xOffset=alt.XOffset("series:N", sort=["LLVM IR", "Assembly"], scale=alt.Scale(paddingInner=0.08)),
    )
    lo = min(0.0, float(df["value"].min()) if len(df) else 0.0)
    hi = max(1.0, float(df["value"].max()) if len(df) else 1.0)
    span = hi - lo
    y = alt.Y("value:Q", title="Reduction vs O0 (%)", scale=alt.Scale(domain=[lo - (span * 0.12 if lo < 0 else 0), hi + span * 0.14], nice=False))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, cornerRadiusBottomLeft=0, cornerRadiusBottomRight=0).encode(
        y=y, color=color,
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.88)),
        tooltip=[
            alt.Tooltip("level:N", title="Level"), alt.Tooltip("series:N", title="Measure"),
            alt.Tooltip("display:N", title="Change"), alt.Tooltip("count:Q", title="Instructions", format=","),
        ],
    ).add_params(hover)
    pos = base.transform_filter(alt.datum.value >= 0).mark_text(dy=-6, baseline="bottom", fontSize=11, fontWeight=500, color=theme.INK, font="Geist").encode(y=alt.Y("value:Q"), text="label:N")
    neg = base.transform_filter(alt.datum.value < 0).mark_text(dy=6, baseline="top", fontSize=11, fontWeight=500, color=theme.INK, font="Geist").encode(y=alt.Y("value:Q"), text="label:N")
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=theme.HAIRLINE, strokeWidth=1).encode(y="z:Q")
    return _config(alt.layer(zero, bars, pos, neg), height)


# column, title, y-axis title, series color, note
METRIC_CHARTS = [
    ("ir_instructions", "LLVM IR instructions", "IR instructions", theme.SERIES_IR,
     "Counted from the .ll files — target-independent IR."),
    ("asm_instructions", "Assembly instructions", "Asm instructions", theme.SERIES_ASM,
     "Counted from the .s files — machine instructions."),
    ("exe_size_bytes", "Executable size", "Bytes", theme.SERIES_NEUTRAL,
     "Includes startup and runtime code added by the linker."),
    ("compile_time_s", "Compilation time", "Milliseconds", theme.SERIES_NEUTRAL,
     "Wall-clock source → assembly, median of repeats; noisy for tiny programs."),
]
