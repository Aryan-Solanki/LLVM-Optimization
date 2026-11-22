"""LLM Compiler in the dashboard: the sidebar step and getting the model's prediction.

"LLM" is one of the levels next to O0…Oz. Before the levels are compiled, the dashboard needs the pass pipeline
LLM Compiler predicts for the source. It reuses a saved prediction for an unchanged file (greedy decoding gives the
same answer) or runs ``src.llm_runner.run_c_workflow`` -> ``llm/get_passes.py``, the same command ``run_all.py``
uses, so the paper's method (prompt, decoding, LLVM 17 validation) is unchanged. The experiment then builds the
"LLM" level like any other level (``src.compiler.compile_llm_level``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import streamlit as st

from src import llm_runner as runner
from src.runlog import new_log_path
from src.toolchain import ToolchainError, verify

from . import components as ui


@dataclass
class Settings:
    server_url: str = runner.DEFAULT_SERVER_URL
    model: Path = runner.DEFAULT_MODEL
    srv: dict = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    cached: bool = False


@st.cache_data(ttl=5, show_spinner=False)
def _server_status(url: str) -> dict:
    return runner.server_status(url)


@st.cache_data(ttl=60, show_spinner=False)
def _toolchain_problem() -> Optional[str]:
    """None if clang, opt and llvm-size are the project's LLVM 17.0.6, else the error text."""
    try:
        verify()
        return None
    except ToolchainError as exc:
        return str(exc)


def out_dir_for(source_path: Path, sample: Path, root: Path) -> Path:
    """Same place run_all.py writes: <results>/llm/<stem>/ (uploads keep their own workspace folder)."""
    base = root if source_path == sample else source_path.parent
    return base / "results" / "llm" / source_path.stem


# ---------------------------------------------------------------- sidebar
def sidebar(step_num: str, source_path: Optional[Path], sample: Path, root: Path) -> Settings:
    """Status only, nothing to configure: Listing 4 IR, default server (used if running, else started and stopped)."""
    s = Settings(server_url=os.environ.get("LLM_SERVER_URL", runner.DEFAULT_SERVER_URL).rstrip("/"),
                 model=Path(os.environ.get("LLM_MODEL_PATH", str(runner.DEFAULT_MODEL))))
    ui.render(ui.step(step_num, "LLM Compiler"))
    tc_problem = _toolchain_problem()
    if source_path is not None:
        s.cached = runner.cached_result(source_path, out_dir_for(source_path, sample, root)) is not None
    if s.cached and not tc_problem:
        ui.render(ui.callout("success", body="A saved prediction for this file is reused: the model gives the same "
                                             "answer for an unchanged file, so it does not run again.", compact=True))
        return s
    s.srv = _server_status(s.server_url)
    s.problems = runner.llm_problems(s.server_url, s.model, s.srv)
    if tc_problem:
        s.problems.insert(0, "LLVM 17 toolchain not usable: " + tc_problem)
    if s.problems:
        for p in s.problems:
            ui.render(ui.callout("error", body=p, compact=True))
    elif s.srv.get("reachable"):
        ui.render(ui.callout("success", body="The LLM Compiler server is running and will be used. Asking the model "
                                             "takes a few minutes.", compact=True))
    else:
        ui.render(ui.callout("neutral", body="The model server starts for the run and stops afterwards. Asking the "
                                             "model takes a few minutes; keep this tab open.", compact=True))
    return s


# ---------------------------------------------------------------- prediction
def get_prediction(source_path: Path, out_dir: Path, s: Settings,
                   log: Callable[[str], None]) -> Tuple[Optional[dict], Optional[str]]:
    """(llm info for run_experiment, None) or (None, error). Reuses a saved prediction for an unchanged file."""
    cached = runner.cached_result(source_path, out_dir)
    if cached is not None:
        log(f"[LLM] reusing the saved prediction in {out_dir / 'result.json'}")
        info, err = runner.prediction_for_experiment(cached, out_dir / "result.json")
        if info:
            info["reused"] = True
        return info, err
    if s.problems:
        return None, "the LLM step cannot run: " + "; ".join(s.problems)

    def line(msg: str) -> None:
        for ln in str(msg).rstrip("\n").splitlines() or [""]:
            log(f"[LLM] {ln}")

    try:
        r = runner.run_c_workflow(source_path, out_dir, server_url=s.server_url, model=s.model, frontend="listing4",
                                  check=False, log_path=new_log_path("llama-server"), echo=line, on_line=line)
    except (RuntimeError, OSError, ValueError) as exc:
        return None, str(exc)
    finally:
        _server_status.clear()
    if not r["result_written"]:
        return None, f"llm/get_passes.py exited with code {r['returncode']} and wrote no result (see the log)"
    result = runner.load_result(out_dir)
    if result is None:
        return None, "result.json could not be read"
    info, err = runner.prediction_for_experiment(result, out_dir / "result.json")
    if info:
        info["reused"] = False
    return info, err
