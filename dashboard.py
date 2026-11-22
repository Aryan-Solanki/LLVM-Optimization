"""Streamlit dashboard:  streamlit run dashboard.py"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analyzer import discover_tests, load_tests, run_experiment
from src.runlog import RunLog
from src.compiler import LLM_LEVEL, OPT_LEVELS, CompilationError, LLVMNotFoundError, find_clang, get_tool_info
from ui import charts, components as ui, llm_page, theme
from ui.code_viewer import render_viewer

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "input" / "example.c"
WORKSPACE = ROOT / "workspace"          # uploads and their outputs live here

SOURCE_SAMPLE = "Bundled sample (example.c)"
SOURCE_UPLOAD = "Upload a C file"

# "LLM" is a level like O0…Oz: -Oz with the IR pass pipeline predicted by LLM Compiler 7B (src/compiler.py,
# compile_llm_level). It is compiled, counted, timed and tested exactly like the other levels.
LLM_OPTION = LLM_LEVEL
LEVEL_OPTIONS = [*OPT_LEVELS, LLM_OPTION]

st.set_page_config(page_title="LLVM Optimization Lab", page_icon="⚙", layout="wide", initial_sidebar_state="auto")
theme.inject_css()

LEAD = (
    "Compile one C file at several -O levels with Clang and compare LLVM IR and assembly instruction "
    "counts, sizes and compile time. IR and assembly instructions are different units, and fewer "
    "instructions does not necessarily mean faster execution."
)


def _request_start() -> None:
    st.session_state["start_requested"] = True


# ---- Toolchain: the project's single LLVM 17.0.6 toolchain (toolchain.json), version-checked ------------------
@st.cache_data(ttl=60, show_spinner=False)
def _toolchain() -> tuple[str, dict]:
    clang = find_clang()                  # refuses anything but the configured clang 17.0.6; never uses PATH
    return clang, get_tool_info(clang)


try:
    CLANG, tool = _toolchain()
except LLVMNotFoundError as exc:
    ui.render(ui.page_header("LLVM · Clang", "Optimization level experiment", LEAD))
    ui.render(ui.callout("error", "The project's LLVM 17 toolchain is not usable",
                         "The dashboard only runs with the toolchain in toolchain.json (clang, opt and llvm-size, all "
                         "LLVM 17.0.6). It never falls back to another compiler. Details and install steps below; "
                         "reload after fixing."))
    st.code(str(exc), language="text")
    st.stop()

# ---- Sidebar: configuration ----------------------------------------------
with st.sidebar:
    ui.render(ui.sidebar_brand())
    ui.render(ui.toolchain_card(tool))

    # 1. Source file
    ui.render(ui.step("01", "Source file"))
    choice = st.segmented_control(
        "Source", [SOURCE_SAMPLE, SOURCE_UPLOAD], default=SOURCE_SAMPLE, required=True,
        format_func=lambda o: "Sample program" if o == SOURCE_SAMPLE else "Upload .c",
        key="source_choice", label_visibility="collapsed", width="stretch",
    ) or SOURCE_SAMPLE
    source_path: Path | None = None
    tests_path: Path | None = None

    if choice.startswith("Bundled"):
        source_path, tests_path = SAMPLE, discover_tests(SAMPLE)
        n_tests = len(load_tests(tests_path)) if tests_path else 1
        lines = SAMPLE.read_text(encoding="utf-8", errors="replace").count("\n")
        ui.render(ui.file_card(SAMPLE.name, f"{lines} lines · {n_tests} test case{'s' if n_tests != 1 else ''}"))
    else:
        up = st.file_uploader("C source", type=["c"], help="A single self-contained .c file with a main().")
        if up is not None:
            data = up.getvalue()
            folder = WORKSPACE / f"{Path(up.name).stem}_{hashlib.sha256(data).hexdigest()[:8]}"
            folder.mkdir(parents=True, exist_ok=True)
            source_path = folder / Path(up.name).name
            source_path.write_bytes(data)
            ui.render(ui.file_card(source_path.name, f"{len(data):,} bytes · default test (no args, empty stdin)"))

    ui.render(ui.divider())

    # 2. Optimization levels (+ LLM Compiler)
    ui.render(ui.step("02", "Optimization levels"))
    levels = st.pills(
        "Levels", LEVEL_OPTIONS, selection_mode="multi", default=OPT_LEVELS, key="levels",
        label_visibility="collapsed", width="stretch",
        help="O0…Oz: clang -O levels. LLM: -Oz with the IR pass pipeline predicted by Meta's LLM Compiler 7B.",
    ) or []
    use_llm = LLM_OPTION in levels
    opt_levels = [lv for lv in levels if lv != LLM_OPTION]
    ui.render(ui.callout("neutral", body="O0 is always included: it is the baseline for reductions and correctness.", compact=True))

    llm_settings = None
    if use_llm:
        ui.render(ui.divider())
        llm_settings = llm_page.sidebar("03", source_path, SAMPLE, ROOT)

    ui.render(ui.divider())

    # 3/4. Correctness testing
    ui.render(ui.step("04" if use_llm else "03", "Correctness testing"))
    run_tests = st.toggle(
        "Execute compiled programs",
        value=choice.startswith("Bundled"),
        key=f"run_{choice}",
        help="Runs one executable per optimization level on your machine, without a sandbox.",
    )
    if run_tests and not choice.startswith("Bundled"):
        ui.render(ui.callout("warn", "Untrusted code runs with your permissions",
                             "Executables built from an uploaded file will run with your permissions. Only continue if you trust this code.",
                             compact=True))
    if not run_tests:
        ui.render(ui.callout("info", body="Execution disabled: correctness will be SKIPPED.", compact=True))
    elif choice.startswith("Bundled"):
        ui.render(ui.callout("success", body="Each level's stdout and exit code are compared with O0.", compact=True))

    st.space("small")
    clicked = st.button("Start experiment", type="primary", disabled=source_path is None, width="stretch",
                        on_click=_request_start)
    if source_path is None:
        ui.render('<p class="lx-small" style="text-align:center">Upload a .c file to enable the run.</p>')

go = bool(st.session_state.pop("start_requested", False)) or clicked

# ---- Header -----------------------------------------------------------------
ui.render(ui.page_header(
    f"LLVM · Clang {ui.short_clang_version(tool['clang_version'])} · {tool['target_architecture']}",
    "Optimization level experiment",
    LEAD,
))

# ---- Run ------------------------------------------------------------------
if go and source_path is not None:
    base = source_path.parent if source_path != SAMPLE else ROOT
    out_root = base / "output"
    res_dir = base / "results"
    log = st.empty()
    messages: list[str] = []
    runlog = RunLog("dashboard")                       # logs/<time>_dashboard.log
    runlog.write(f"source {source_path.name} · levels {' '.join(levels) or 'O0'} · execute {run_tests}")

    def progress(msg: str) -> None:
        messages.append(msg)
        runlog.write(msg)
        log.code("\n".join(messages[-80:]), language="text")

    st.session_state.pop("llm_error", None)
    try:
        llm_info = None
        if use_llm:                                        # the model's pipeline is needed before the LLM level
            spin = ("Using the saved LLM Compiler prediction..." if llm_settings.cached else
                    "Asking LLM Compiler for a pass pipeline (a few minutes)...")
            with st.spinner(spin):
                llm_info, llm_err = llm_page.get_prediction(
                    source_path, llm_page.out_dir_for(source_path, SAMPLE, ROOT), llm_settings, progress)
            if llm_err:
                st.session_state["llm_error"] = llm_err
        run_levels = (opt_levels or ["O0"]) + ([LLM_LEVEL] if llm_info else [])
        with st.spinner("Compiling and measuring..."):
            res = run_experiment(
                source_path, levels=run_levels, output_root=out_root, results_dir=res_dir,
                clang=CLANG, run_tests=run_tests, tests=load_tests(tests_path), progress=progress, llm=llm_info,
            )
        if (res.metadata.get("llm") or {}).get("error"):
            st.session_state["llm_error"] = res.metadata["llm"]["error"]
        st.session_state["result"] = res
        st.session_state["source_text"] = source_path.read_text(encoding="utf-8", errors="replace")
        log.empty()
    except CompilationError as exc:
        log.empty()
        runlog.write(f"ERROR {exc}\n{' '.join(exc.command or [])}\n{exc.stderr or ''}")
        ui.render(ui.callout("error", "Compilation failed", str(exc)))
        st.code(" ".join(exc.command or []), language="text", wrap_lines=True)
        st.code(exc.stderr or "(no stderr)", language="text")
    except Exception as exc:                               # noqa: BLE001 - show anything to the user
        runlog.write(f"ERROR {type(exc).__name__}: {exc}")
        ui.render(ui.callout("error", type(exc).__name__, str(exc)))
    finally:
        runlog.write("done")
        runlog.close()

res = st.session_state.get("result")
if res is None:
    src_label = source_path.name if source_path is not None else "no file uploaded yet"
    ui.render(ui.empty_state(src_label, len(set(opt_levels) | {"O0"}), run_tests, source_path is not None, use_llm))
    st.space("small")
    st.button("Start experiment", type="primary", disabled=source_path is None, key="start_main", on_click=_request_start)
    st.stop()

table: pd.DataFrame = res.table
levels_run = list(res.artifacts)

if st.session_state.get("llm_error"):
    ui.render(ui.callout("error", "The LLM level was not run",
                         str(st.session_state["llm_error"])[:600] + " The other levels are shown."))
    st.space("small")
ui.render(ui.run_bar(res.metadata))
ui.render(ui.kpi_grid(table))
st.space("small")

tabs = st.tabs(["Results", "Graphs", "LLVM IR", "Assembly", "Source", "Run details"])

# ---- Results ----------------------------------------------------------------
with tabs[0]:
    bad = table[table["correctness"] == "FAIL"]
    if len(bad):
        ui.render(ui.callout("error", "Correctness mismatch",
                             "Some levels produced different output from O0: " + ", ".join(bad["opt_level"])))
        st.space("small")

    head_l, head_r = st.columns([3, 1], vertical_alignment="bottom")
    with head_l:
        ui.render(ui.section("Results comparison", "Every value is measured from real Clang output. Bold marks the best value in each column."))
    with head_r:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.download_button("Download results.csv", table.to_csv(index=False), "results.csv", "text/csv",
                               icon=":material/download:")
    ui.render(ui.results_table(table))
    ui.render(
        '<div class="lx-foot">'
        + ui.legend([("LLVM IR", theme.SERIES_IR), ("Assembly", theme.SERIES_ASM)])
        + '<span class="lx-small">Reduction (%) = (O0 count − optimized count) / O0 count × 100, computed separately for IR and assembly.</span>'
        "</div>"
    )
    st.space("small")
    with st.expander("All columns · raw data"):
        st.dataframe(
            table, width="stretch", hide_index=True,
            column_config={
                "opt_level": st.column_config.TextColumn("Level"),
                "ir_reduction_pct": st.column_config.NumberColumn("ir_reduction_pct", format="%.2f"),
                "asm_reduction_pct": st.column_config.NumberColumn("asm_reduction_pct", format="%.2f"),
                "compile_time_s": st.column_config.NumberColumn("compile_time_s", format="%.4f"),
                "ir_time_s": st.column_config.NumberColumn("ir_time_s", format="%.4f"),
                "asm_time_s": st.column_config.NumberColumn("asm_time_s", format="%.4f"),
                "exe_time_s": st.column_config.NumberColumn("exe_time_s", format="%.4f"),
                "correctness_detail": st.column_config.TextColumn("correctness_detail", width="large"),
            },
        )

# ---- Graphs -----------------------------------------------------------------
with tabs[1]:
    ui.render(ui.section("Graphs", "One measure per chart. O0 is the grey baseline; hover a bar for exact values."))
    with st.container(border=True, key="lxcard_reduction"):
        ui.render(
            '<div class="lx-chart-head"><div class="row"><h3 class="lx-h3">Instruction reduction vs O0</h3>'
            f'<div class="lx-chart-legend">{ui.legend([("LLVM IR", theme.SERIES_IR), ("Assembly", theme.SERIES_ASM)])}</div></div>'
            '<p class="lx-small">Share of O0 instructions removed at each level. IR and assembly are different units, so compare the trend, not the bars across colors.</p></div>'
        )
        if len(table) > 1:
            st.altair_chart(charts.reduction_chart(table), width="stretch", theme=None)
        else:
            ui.render(ui.callout("neutral", body="Select at least one optimized level to compare against O0."))
    st.space("small")
    cols = st.columns(2, gap="medium")
    for i, (column, title, ytitle, color, note) in enumerate(charts.METRIC_CHARTS):
        with cols[i % 2]:
            with st.container(border=True, key=f"lxcard_{column}"):
                ui.render(
                    f'<div class="lx-chart-head"><div class="row"><h3 class="lx-h3">{ui.esc(title)}</h3>'
                    f'<div class="lx-chart-legend">{ui.legend([("O0 baseline", theme.BASELINE), ("Optimized", color)])}</div></div>'
                    f'<p class="lx-small">{ui.esc(note)}</p></div>'
                )
                st.altair_chart(charts.metric_chart(table, column, color, ytitle), width="stretch", theme=None)


# ---- Code viewers -------------------------------------------------------------
def code_tab(lang: str, count_col: str, key: str) -> None:
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        lv = st.segmented_control("Level", levels_run, default=levels_run[0], required=True, key=key)
        others = [x for x in levels_run if x != lv]
        cmp = st.selectbox("Compare with", ["None", *others], key=f"{key}_cmp", width=150)
        query = st.text_input("Search", key=f"{key}_q", placeholder="Find in file…", width=220)
        with st.container(width="content", gap=None):
            only = st.toggle("Matches only", key=f"{key}_only", disabled=not query.strip())
            wrap = st.toggle("Wrap lines", key=f"{key}_wrap")
    lv = lv or levels_run[0]
    shown = [lv] + ([cmp] if cmp and cmp != "None" else [])
    cols = st.columns(len(shown), gap="small") if len(shown) > 1 else [st.container()]
    for col, level in zip(cols, shown):
        art = res.artifacts[level]
        path = art.ir_path if lang == "llvm" else art.asm_path
        text = path.read_text(encoding="utf-8", errors="replace")
        row = table[table["opt_level"] == level]
        count = int(row[count_col].iloc[0]) if len(row) and not pd.isna(row[count_col].iloc[0]) else None
        meta = [("instructions", f"{count:,}")] if count is not None else []
        with col:
            ui.render(render_viewer(text, lang, path.name, meta, query, only, wrap, ui.level_badge(level)))
            st.download_button(f"Download {path.name}", text, path.name, "text/plain", key=f"dl_{key}_{level}",
                               icon=":material/download:", type="tertiary")


with tabs[2]:
    ui.render(ui.section("LLVM IR", "Target-independent intermediate representation emitted by clang -S -emit-llvm."))
    code_tab("llvm", "ir_instructions", "ir_level")

with tabs[3]:
    ui.render(ui.section("Assembly", f"Machine code for {tool['target_triple']} emitted by clang -S."))
    code_tab("asm", "asm_instructions", "asm_level")

with tabs[4]:
    src_text = st.session_state.get("source_text", "")
    src_name = res.metadata.get("source_file", "source.c")
    ui.render(ui.section("Source", "The exact C file that was compiled at every level."))
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        q = st.text_input("Search", key="src_q", placeholder="Find in file…", width=220)
        with st.container(width="content", gap=None):
            only = st.toggle("Matches only", key="src_only", disabled=not q.strip())
            wrap = st.toggle("Wrap lines", key="src_wrap")
    ui.render(render_viewer(src_text, "c", src_name, [("", f"sha256 {res.metadata.get('source_sha256', '')[:12]}")], q, only, wrap))
    st.download_button(f"Download {src_name}", src_text, src_name, "text/plain", key="dl_src",
                       icon=":material/download:", type="tertiary")

# ---- Run details --------------------------------------------------------------
with tabs[5]:
    md = res.metadata
    tc = md.get("toolchain", {})
    ui.render(ui.section("Run details", "Everything needed to reproduce this experiment."))
    if tc.get("warnings"):
        ui.render(ui.callout("warn", "Toolchain version mismatch", tc["warnings"]))
        st.space("small")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.render(ui.card("Experiment", ui.kv([
            ("Source file", md.get("source_file", ""), True),
            ("SHA-256", md.get("source_sha256", ""), True),
            ("Generated", str(md.get("generated_utc", "")).replace("T", " ").replace("+00:00", " UTC"), True),
            ("Levels", " ".join(md.get("levels", [])), True),
            ("Extra flags", " ".join(md.get("extra_flags", [])) or "none", True),
            ("Timing", f"median of {md.get('timing_repeats_median_of', '?')} runs", False),
            ("Execution", f"enabled · {len(md.get('tests', []))} test(s)" if md.get("executed_tests") else "disabled", False),
        ]), eyebrow="Metadata"))
    with c2:
        ui.render(ui.card("Toolchain", ui.kv([
            ("Clang", tc.get("clang_version", ""), True),
            ("Target triple", tc.get("target_triple", ""), True),
            ("Architecture", tc.get("target_architecture", ""), True),
            ("Host", f"{tc.get('host_os', '')} · {tc.get('host_machine', '')}", True),
            ("Python", tc.get("python_version", ""), True),
            ("Clang path", tc.get("clang_path", ""), True),
            ("Toolchain", f"{tc.get('toolchain_config', 'n/a')} · LLVM {tc.get('llvm_version_required', 'n/a')}", True),
            ("opt", f"{tc.get('toolchain_opt_version', 'n/a')} · {tc.get('toolchain_opt_path', '')}", True),
            ("llvm-size", f"{tc.get('toolchain_llvm_size_version', 'n/a')} · {tc.get('toolchain_llvm_size_path', '')}", True),
        ]), eyebrow="Compiler"))
    st.space("small")

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        ui.render(ui.card("Companion LLVM tools", ui.tools_list(tc), eyebrow="Versions"))
    with c4:
        ui.render(ui.card("Test cases", ui.tests_table(md.get("tests", [])), eyebrow="Correctness inputs"))
    st.space("small")

    if md.get("llm") and not md["llm"].get("error"):
        ui.render(ui.card("LLM level: predicted pass pipeline", ui.llm_card(md["llm"]), eyebrow="LLM Compiler 7B"))
        st.space("small")

    with st.container(border=True, key="lxcard_commands"):
        ui.render('<div class="lx-chart-head"><p class="lx-eyebrow">Reproduce</p><h3 class="lx-h3">Compiler commands</h3></div>')
        cmd_lv = st.segmented_control("Level", levels_run, default=levels_run[0], required=True, key="cmd_level",
                                      label_visibility="collapsed")
        for step_name, cmd in md.get("commands", {}).get(cmd_lv or levels_run[0], {}).items():
            label = {"frontend": "C → Listing 4 IR (front end, no IR passes)",
                     "ir": "LLVM IR (opt -p <model's pipeline>)" if cmd_lv == LLM_LEVEL else "LLVM IR",
                     "asm": "Assembly", "exe": "Executable"}.get(step_name, step_name)
            ui.render(f'<p class="lx-small" style="margin:8px 0 4px">{ui.esc(label)}</p>')
            st.code(" ".join(f'"{c}"' if " " in c else c for c in cmd), language="text", wrap_lines=True)

    if res.test_outputs:
        st.space("small")
        ui.render(ui.section("Program output", "Captured stdout of each level, per test case."))
        tests_meta = md.get("tests", [])
        for lv, runs in res.test_outputs.items():
            with st.expander(f"{lv} · program output ({len(runs)} test{'s' if len(runs) != 1 else ''})"):
                for i, run in enumerate(runs):
                    name = tests_meta[i]["name"] if i < len(tests_meta) else f"test{i}"
                    ui.render(
                        f'<div class="lx-out-head"><span class="name">{ui.esc(name)}</span>'
                        f'<span class="meta">exit {ui.esc(run.get("returncode"))} · {ui.esc(run.get("status"))}</span></div>'
                    )
                    st.code(run["stdout"].decode("utf-8", errors="replace") or "(no output)", language="text")

    st.space("small")
    ui.render(ui.callout("neutral", "Notes", "<br>".join(ui.esc(n) for n in md.get("notes", [])), body_html=True))
    st.space("small")
    with st.expander("Raw metadata.json"):
        st.json(res.metadata)
