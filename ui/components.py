"""HTML building blocks rendered with ``st.html``.

Every dynamic value passes through :func:`esc`.  Components only *display*
values already computed by ``src.analyzer``; they never recompute results.
"""

from __future__ import annotations

import html as _html
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from src.compiler import LLM_LEVEL, OPT_LEVELS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_relative(path) -> str:
    """Show paths inside the project relative to it (no machine-specific folders in the UI)."""
    try:
        return Path(str(path)).resolve().relative_to(_PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return str(path)

from . import theme

BASELINE_LEVEL = "O0"


def esc(value) -> str:
    return _html.escape("" if value is None else str(value), quote=True)


def render(markup: str) -> None:
    st.html(markup)


def _isnan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v)
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Icons (16px line icons, stroke = currentColor)
# --------------------------------------------------------------------------
def icon(name: str) -> str:
    """Icon as a CSS-masked element (st.html strips inline SVG; see theme.ICON_PATHS)."""
    return f'<i class="lx-i lx-i-{name}" aria-hidden="true"></i>'


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def level_badge(level: str) -> str:
    cls = ("lx-level is-base" if level == BASELINE_LEVEL else
           "lx-level is-model" if level == LLM_LEVEL else "lx-level")
    return f'<span class="{cls}">{esc(level)}</span>'


# correctness status -> (css tone, icon, short label)
_STATUS = {
    "PASS": ("ok", "check", "Pass"),
    "FAIL": ("fail", "x", "Fail"),
    "ERROR": ("fail", "x", "Error"),
    "BASELINE": ("base", "circle", "Baseline"),
    "N/A": ("warn", "minus", "N/A"),
}


def pill(tone: str, ic: str, label: str, title: str = "") -> str:
    """tone: ok | fail | warn | neutral | base"""
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="lx-status {tone}"{t}>{icon(ic)}{esc(label)}</span>'


def status_badge(status: str, title: str = "") -> str:
    if str(status).startswith("SKIPPED"):
        tone, ic, label = "neutral", "minus", "Skipped"
    else:
        tone, ic, label = _STATUS.get(str(status), ("neutral", "minus", str(status)))
    return pill(tone, ic, label, title)


def callout(kind: str, title: str = "", body: str = "", compact: bool = False, body_html: bool = False) -> str:
    """kind: info | neutral | warn | error | success"""
    ic = {"info": "info", "neutral": "info", "warn": "alert", "error": "error", "success": "success"}[kind]
    parts = []
    if title:
        parts.append(f'<p class="lx-callout-title">{esc(title)}</p>')
    if body:
        parts.append(f'<p class="lx-callout-body">{body if body_html else esc(body)}</p>')
    cls = f"lx-callout {kind}" + (" compact" if compact else "")
    return f'<div class="{cls}" role="{"alert" if kind == "error" else "note"}">{icon(ic)}<div>{"".join(parts)}</div></div>'


def section(title: str, description: str = "", eyebrow: str = "", right: str = "") -> str:
    eb = f'<p class="lx-eyebrow">{esc(eyebrow)}</p>' if eyebrow else ""
    desc = f'<p class="lx-body">{esc(description)}</p>' if description else ""
    return (
        f'<div class="lx-section"><div class="lx-section-text">{eb}<h2 class="lx-h2">{esc(title)}</h2>{desc}</div>'
        f"{right}</div>"
    )


# --------------------------------------------------------------------------
# Page chrome
# --------------------------------------------------------------------------
def page_header(eyebrow: str, title: str, lead: str, chips: Sequence[Tuple[str, str]] = ()) -> str:
    chip_html = "".join(f'<span class="lx-chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in chips)
    meta = f'<div class="lx-header-meta">{chip_html}</div>' if chips else ""
    return (
        '<header class="lx-header">'
        f'<p class="lx-eyebrow">{esc(eyebrow)}</p>'
        f'<div class="lx-header-row"><h1 class="lx-h1">{esc(title)}</h1>{meta}</div>'
        f'<p class="lx-lead">{esc(lead)}</p>'
        "</header>"
    )


def sidebar_brand() -> str:
    return (
        '<div class="lx-brand"><span class="lx-mark">-O</span><div>'
        '<div class="lx-brand-name">LLVM Optimization Lab</div>'
        f'<div class="lx-brand-sub">clang 17 · {esc(OPT_LEVELS[0])} … {esc(OPT_LEVELS[-1])} · {esc(LLM_LEVEL)}</div></div></div>'
    )


def step(num: str, title: str) -> str:
    return f'<div class="lx-step"><span class="lx-step-num">{esc(num)}</span><span class="lx-step-title">{esc(title)}</span></div>'


def divider() -> str:
    return '<hr class="lx-divider">'


def short_clang_version(version_line: str) -> str:
    """'clang version 17.0.6 (https://...)' -> '17.0.6'."""
    parts = str(version_line).split()
    for i, p in enumerate(parts):
        if p == "version" and i + 1 < len(parts):
            return parts[i + 1]
    return str(version_line)


def toolchain_card(tool: Dict[str, str]) -> str:
    rows = [
        ("Target", tool.get("target_triple", "unknown")),
        ("Arch", tool.get("target_architecture", "unknown")),
        ("LLVM", f'opt {tool.get("toolchain_opt_version", "?")} · llvm-size {tool.get("toolchain_llvm_size_version", "?")}'),
        ("Config", tool.get("toolchain_config", "toolchain.json")),
        ("Host", f'{tool.get("host_os", "")} · {tool.get("host_machine", "")}'),
    ]
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
    return (
        f'<div class="lx-tool" title="{esc(tool.get("clang_version", ""))}">'
        '<div class="lx-tool-top"><span class="lx-dot"></span><span class="lx-tool-name">Clang ready</span>'
        f'<span class="lx-tool-ver">v{esc(short_clang_version(tool.get("clang_version", "")))}</span></div>'
        f"<dl>{dl}</dl></div>"
    )


def file_card(name: str, meta: str, badge: str = "C") -> str:
    return (
        f'<div class="lx-file"><span class="lx-file-icon">{esc(badge)}</span><div>'
        f'<div class="lx-file-name">{esc(name)}</div><div class="lx-file-meta">{esc(meta)}</div></div></div>'
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def fmt_int(v) -> str:
    return "—" if _isnan(v) else f"{int(round(float(v))):,}"


def fmt_bytes_short(v) -> str:
    if _isnan(v):
        return "—"
    v = float(v)
    if v >= 1024 * 1024:
        return f"{v / 1024 / 1024:.2f} MB"
    if v >= 1024:
        return f"{v / 1024:.1f} KB"
    return f"{int(v)} B"


def fmt_ms(seconds) -> str:
    return "—" if _isnan(seconds) else f"{float(seconds) * 1000:.1f}"


def fmt_pct(v, signed_reduction: bool = True) -> str:
    """A reduction of 57.5 is shown as −57.5% (the count went down)."""
    if _isnan(v):
        return "—"
    v = float(v)
    if signed_reduction:
        return f"−{v:.1f}%" if v > 0 else (f"+{-v:.1f}%" if v < 0 else "0.0%")
    return f"{v:.1f}%"


def run_bar(metadata: Dict) -> str:
    tests = metadata.get("tests") or []
    executed = metadata.get("executed_tests")
    items = [
        ("Source", metadata.get("source_file", "")),
        ("Target", (metadata.get("toolchain") or {}).get("target_triple", "")),
        ("Levels", " ".join(metadata.get("levels", []))),
        ("Tests", f"{len(tests)} executed" if executed else "execution off"),
        ("Generated", str(metadata.get("generated_utc", "")).replace("T", " ").replace("+00:00", " UTC")),
    ]
    out = []
    for k, v in items:
        out.append(f'<span class="lx-runbar-item"><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></span>')
    return f'<div class="lx-runbar">{"".join(out)}</div>'


def _best_levels(table: pd.DataFrame, column: str, mode: str) -> Tuple[List[str], Optional[float]]:
    s = pd.to_numeric(table[column], errors="coerce")
    if s.dropna().empty:
        return [], None
    target = s.max() if mode == "max" else s.min()
    return [lv for lv, v in zip(table["opt_level"], s) if not _isnan(v) and v == target], float(target)


def kpi_grid(table: pd.DataFrame) -> str:
    t = table
    base = t[t["opt_level"] == BASELINE_LEVEL].iloc[0] if (t["opt_level"] == BASELINE_LEVEL).any() else None
    opt = t[t["opt_level"] != BASELINE_LEVEL]
    tiles = []

    def badges(levels):
        return "".join(level_badge(lv) for lv in levels)

    for col, count_col, label, color in (
        ("ir_reduction_pct", "ir_instructions", "Largest IR reduction", theme.SERIES_IR),
        ("asm_reduction_pct", "asm_instructions", "Largest assembly reduction", theme.SERIES_ASM),
    ):
        levels, best = _best_levels(opt, col, "max") if len(opt) else ([], None)
        if best is None:
            value, sub = "—", "No optimized level selected"
        else:
            row = opt[opt["opt_level"] == levels[0]].iloc[0]
            value = fmt_pct(best).replace("%", "<small>%</small>")
            sub = f'{badges(levels)}<span>{fmt_int(base[count_col]) if base is not None else "—"} → {fmt_int(row[count_col])} instructions</span>'
        tiles.append((label, value, sub, color))

    levels, best = _best_levels(t, "exe_size_bytes", "min")
    if best is None:
        tiles.append(("Smallest executable", "—", "<span>No executable was linked</span>", None))
    else:
        short = fmt_bytes_short(best)
        num, _, unit = short.partition(" ")
        delta = ""
        if base is not None and not _isnan(base["exe_size_bytes"]) and float(base["exe_size_bytes"]):
            d = (float(base["exe_size_bytes"]) - best) / float(base["exe_size_bytes"]) * 100
            delta = f"<span>{fmt_pct(d)} vs O0</span>" if levels != [BASELINE_LEVEL] else "<span>baseline is smallest</span>"
        tiles.append(("Smallest executable", f"{num}<small>{esc(unit)}</small>",
                      f"{badges(levels)}<span>{best:,.0f} B</span>{delta}", None))

    statuses = list(opt["correctness"]) if len(opt) else []
    n_pass = sum(s == "PASS" for s in statuses)
    n_fail = sum(s in ("FAIL", "ERROR") for s in statuses)
    if statuses and all(str(s).startswith("SKIPPED") for s in t["correctness"]):
        value, sub = "Skipped", status_badge("SKIPPED") + "<span>Execution disabled</span>"
    elif not statuses:
        value, sub = "—", "<span>Only the baseline was compiled</span>"
    else:
        value = f"{n_pass}<small>/ {len(statuses)}</small>"
        if n_fail:
            sub = status_badge("FAIL") + f"<span>{n_fail} level(s) differ from O0</span>"
        elif n_pass == len(statuses):
            sub = status_badge("PASS") + "<span>All outputs match O0</span>"
        else:
            sub = status_badge("N/A") + "<span>Some levels could not be compared</span>"
    tiles.append(("Correctness vs O0", value, sub, None))

    cells = []
    for label, value, sub, color in tiles:
        sw = f'<span class="lx-kpi-swatch" style="background:{color}"></span>' if color else ""
        cells.append(
            f'<div class="lx-kpi"><div class="lx-kpi-label">{sw}{esc(label)}</div>'
            f'<div class="lx-kpi-value">{value}</div><div class="lx-kpi-sub">{sub}</div></div>'
        )
    return f'<div class="lx-kpis">{"".join(cells)}</div>'


def _meter(pct, color: str, is_base: bool) -> str:
    if is_base:
        return '<span class="lx-dash">baseline</span>'
    if _isnan(pct):
        return '<span class="lx-dash">—</span>'
    p = float(pct)
    width = max(0.0, min(100.0, p))
    cls = "pos" if p > 0 else ("neg" if p < 0 else "")
    return (
        f'<span class="lx-meter"><span class="lx-meter-val {cls}">{fmt_pct(p)}</span>'
        f'<span class="lx-meter-track"><span class="lx-meter-fill" style="width:{width:.1f}%;background:{color}"></span></span></span>'
    )


def results_table(table: pd.DataFrame) -> str:
    """Comparison table.  Bold ink marks the best value in each column."""
    best = {
        "ir_instructions": set(_best_levels(table, "ir_instructions", "min")[0]),
        "asm_instructions": set(_best_levels(table, "asm_instructions", "min")[0]),
        "ir_size_bytes": set(_best_levels(table, "ir_size_bytes", "min")[0]),
        "asm_size_bytes": set(_best_levels(table, "asm_size_bytes", "min")[0]),
        "exe_size_bytes": set(_best_levels(table, "exe_size_bytes", "min")[0]),
        "compile_time_s": set(_best_levels(table, "compile_time_s", "min")[0]),
    }
    sw_ir = f'<span class="sw" style="background:{theme.SERIES_IR}"></span>'
    sw_asm = f'<span class="sw" style="background:{theme.SERIES_ASM}"></span>'
    head = (
        "<thead><tr>"
        "<th>Level</th>"
        f"<th>{sw_ir}IR instr.</th><th>IR vs O0</th>"
        f"<th>{sw_asm}Asm instr.</th><th>Asm vs O0</th>"
        "<th>IR size</th><th>Asm size</th><th>Executable</th>"
        "<th>Compile</th><th class='l'>Correctness</th>"
        "</tr></thead>"
    )
    rows = []
    for _, r in table.iterrows():
        lv = r["opt_level"]
        is_base = lv == BASELINE_LEVEL

        def cell(col, text, unit=""):
            cls = "best" if lv in best.get(col, ()) and len(table) > 1 else ""
            u = f'<span class="unit">{unit}</span>' if unit and text != "—" else ""
            return f'<td class="{cls}"><span class="num">{text}</span>{u}</td>'

        rows.append(
            "<tr>"
            f"<td>{level_badge(lv)}</td>"
            + cell("ir_instructions", fmt_int(r["ir_instructions"]))
            + f"<td>{_meter(r['ir_reduction_pct'], theme.SERIES_IR, is_base)}</td>"
            + cell("asm_instructions", fmt_int(r["asm_instructions"]))
            + f"<td>{_meter(r['asm_reduction_pct'], theme.SERIES_ASM, is_base)}</td>"
            + cell("ir_size_bytes", fmt_int(r["ir_size_bytes"]), "B")
            + cell("asm_size_bytes", fmt_int(r["asm_size_bytes"]), "B")
            + cell("exe_size_bytes", fmt_int(r["exe_size_bytes"]), "B")
            + cell("compile_time_s", fmt_ms(r["compile_time_s"]), "ms")
            + f"<td class='l'>{status_badge(r['correctness'], r.get('correctness_detail', ''))}</td>"
            "</tr>"
        )
    return f'<div class="lx-table-wrap"><table class="lx-table">{head}<tbody>{"".join(rows)}</tbody></table></div>'


def legend(items: Iterable[Tuple[str, str]]) -> str:
    return "".join(f'<span class="lx-legend-item"><i style="background:{c}"></i>{esc(t)}</span>' for t, c in items)


# --------------------------------------------------------------------------
# Details
# --------------------------------------------------------------------------
def kv(rows: Sequence[Tuple[str, str, bool]]) -> str:
    """rows: (label, value, monospace)"""
    body = "".join(
        f'<dt>{esc(k)}</dt><dd class="{"mono" if mono else ""}">{esc(v)}</dd>' for k, v, mono in rows
    )
    return f'<dl class="lx-kv">{body}</dl>'


def card(title: str, inner_html: str, eyebrow: str = "", right: str = "") -> str:
    eb = f'<p class="lx-eyebrow">{esc(eyebrow)}</p>' if eyebrow else ""
    return (
        f'<section class="lx-card"><div class="lx-card-head"><div>{eb}<h3 class="lx-h3">{esc(title)}</h3></div>{right}</div>'
        f"{inner_html}</section>"
    )


def tools_list(tool: Dict) -> str:
    items = []
    for key, val in tool.items():
        if not key.startswith("tool_"):
            continue
        missing = val in ("not found", "unknown", "")
        items.append(
            f'<div class="lx-toolitem{" missing" if missing else ""}"><span>{esc(key[5:])}</span><span>{esc(val)}</span></div>'
        )
    return f'<div class="lx-tools">{"".join(items)}</div>' if items else '<p class="lx-small">No companion tools checked.</p>'


def llm_card(llm: Dict) -> str:
    """Run-details card for the LLM level: the model's pipeline (verbatim) and where it came from."""
    pipe = llm.get("pipeline") or "(no passes: the model says the code cannot be reduced further)"
    when = (f"saved prediction from {llm.get('predicted_at', '')} reused (same file, greedy decoding)"
            if llm.get("reused") else f"asked in this run ({llm.get('predicted_at', '')})")
    before, after = llm.get("predicted_object_size_before"), llm.get("predicted_object_size_after")
    claim = f"{before:,} → {after:,} B object size" if isinstance(before, int) and isinstance(after, int) else "—"
    rows = kv([
        ("Prediction", when, False),
        ("Model", str(llm.get("model", "")), True),
        ("Prompt", f"{llm.get('prompt_tokens', '—')} tokens · greedy · {llm.get('inference_seconds', '—')} s", True),
        ("Pipeline type", {"preset": "preset only", "mixed": "preset + passes", "custom": "custom passes"}
         .get(str(llm.get("kind")), str(llm.get("kind") or "none")), False),
        ("Model's own claim", claim + " (its prediction, in the paper's object-size measure)", False),
        ("Prediction file", _project_relative(llm.get("result_json", "")), True),
    ])
    return (f'<pre class="lx-pipeline">{esc(pipe).replace(",", ",<wbr>")}</pre>' + rows
            + '<p class="lx-small">The LLM level is -Oz with the IR pass pipeline replaced by this one: Listing 4 IR '
              "(clang -Oz -Xclang -disable-llvm-passes) → opt -p &lt;pipeline&gt; → code generation at -Oz. Compile "
              "times exclude model inference.</p>")


def tests_table(tests: List[Dict]) -> str:
    if not tests:
        return '<p class="lx-small">Execution was disabled, so no test cases were run.</p>'
    rows = "".join(
        f"<tr><td>{esc(t.get('name', ''))}</td>"
        f"<td>{esc(' '.join(t.get('args', [])) or '—')}</td>"
        f"<td class='m'>{esc(repr(t.get('stdin', '')) if t.get('stdin') else '—')}</td></tr>"
        for t in tests
    )
    return f'<table class="lx-tests"><thead><tr><th>Name</th><th>Args</th><th>Stdin</th></tr></thead><tbody>{rows}</tbody></table>'


def empty_state(source_label: str, n_levels: int, executing: bool, ready: bool, llm: bool = False) -> str:
    steps = [
        ("01", "Source file", source_label, ready),
        ("02", "Optimization levels", f"{n_levels} selected · O0 always included" + (" · + LLM" if llm else ""), True),
        ("03", "Correctness testing", "execute binaries vs O0" if executing else "execution disabled", True),
    ]
    step_html = "".join(
        f'<div class="lx-empty-step"><span class="n">{n}</span><div><div class="t">{esc(t)}</div>'
        f'<div class="d">{esc(d)}</div></div>'
        f'{pill("ok", "check", "Ready") if ok else pill("warn", "minus", "Needed")}</div>'
        for n, t, d, ok in steps
    )
    return (
        '<div class="lx-empty"><div style="display:flex;flex-direction:column;gap:12px;justify-content:center">'
        '<p class="lx-eyebrow">No experiment yet</p>'
        '<h2 class="lx-h2">Compile once, compare every level.</h2>'
        '<p class="lx-body">The experiment compiles your C file with the same Clang at each selected '
        "<code>-O</code> level, counts LLVM IR and assembly instructions, measures sizes and compile time, "
        "and optionally checks every binary against the O0 output.</p>"
        '<p class="lx-small">Configure the run in the sidebar (tap the menu icon on small screens), then start the experiment.</p>'
        f'</div><div class="lx-empty-steps">{step_html}</div></div>'
    )
