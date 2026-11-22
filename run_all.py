#!/usr/bin/env python3
"""One command that runs the whole project with its single LLVM 17.0.6 toolchain.

    python run_all.py                    # sample program: compiler experiment + LLM Compiler workflow
    python run_all.py prog.c             # your own C file
    python run_all.py --check            # only check the toolchain, model and server; run nothing
    python run_all.py --no-llm           # compiler experiment only (same as main.py)
    python run_all.py --no-experiment    # LLM workflow only
    python run_all.py --dashboard        # open the Streamlit dashboard at the end

Steps
  1. Preflight: the project toolchain from toolchain.json (clang, opt, llvm-size; each must report LLVM 17.0.6 and
     clang must target x86_64-w64-windows-gnu, otherwise nothing runs), llama-server, the model file, the server.
  2. LLM workflow: connects to a running llama-server serving LLM Compiler, or starts one from llm/ and stops
     it again when done (only the process this script started; it never kills any other llama-server).
     Then runs llm/get_passes.py --evaluate on the C file: the model predicts a pass pipeline.
     Artifacts: results/llm/<stem>/ (result.json with the prediction and the paper's object-size measurements).
  3. Compiler experiment: runs main.py for O0 O1 O2 O3 Os Oz and, if step 2 produced a usable pipeline, the
     LLM level (-Oz with the model's IR pass pipeline), measured exactly like the others
     (results/results.csv, metadata.json, charts, output/<level>/).
  4. (optional) dashboard: the same levels, with LLM selectable next to O0…Oz.

Both workflows use the same toolchain: the experiment compiles with its clang; the LLM workflow compiles the C file
to unoptimized IR with the same clang (the paper's Listing 4 form by default), applies the predicted pipeline with
its opt and measures object size with its clang and llvm-size. No tool is ever taken from PATH.

results/results.csv holds every level including LLM (IR/assembly instructions, sizes, compile time, correctness).
results/llm/<stem>/result.json additionally keeps the paper's own measurement of the prediction (object size,
text+data) and the prompt and reply.

Configuration (command-line flag, else environment variable, else default)
  --server-url   LLM_SERVER_URL   http://127.0.0.1:8080
  --model        LLM_MODEL_PATH   llm/models/llm-compiler-7b-ftd.Q4_K_M.gguf
  Tools          toolchain.json   (no flag or environment variable on purpose: one toolchain for everything)
The LLM part is Windows-only as shipped (prebuilt llama.cpp CUDA binaries).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.toolchain import TOOLS, ToolchainError, inspect, report, verify  # noqa: E402  (the one toolchain)
from src.runlog import new_log_path, tee_output  # noqa: E402  (every run writes logs/<time>_run_all.log)
from src.llm_runner import (  # noqa: E402  (shared with the dashboard: server lifecycle, get_passes command)
    DEFAULT_MODEL, LLM, SERVER_EXE, get_passes_command, llm_problems, load_result, run_streaming,
    server_status, start_server, stop_server, wait_ready,
)

DEFAULT_SOURCE = ROOT / "input" / "example.c"
ARTIFACTS: dict = {}

OK, FAILED, SKIPPED = "ok", "FAILED", "skipped"

EXEC_WARNING = """\
WARNING: this COMPILES AND RUNS the C program you gave me (once per optimization level, and once per LLM
variant for the differential test). The programs run with your user permissions and are NOT sandboxed.
Only run source code you wrote or trust.  Use --no-run to compile and measure without executing anything.
"""


# ---------------------------------------------------------------- helpers
def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


# ---------------------------------------------------------------- preflight
def preflight(a) -> dict:
    """Print the toolchain and LLM readiness; return {'toolchain', 'llm_problems': [...], 'server': {...}}."""
    banner("1. Preflight")
    try:
        tc = inspect()
        print(report(tc))
        print("  (taken only from toolchain.json; nothing is looked up on PATH)")
    except ToolchainError as exc:                  # e.g. toolchain.json missing or malformed
        tc = None
        print(f"Toolchain: NOT USABLE\n{exc}")
    srv = server_status(a.server_url)
    problems = llm_problems(a.server_url, a.model, srv, no_start_server=a.no_start_server)
    if srv["reachable"]:
        print(f"LLM server   : reachable at {a.server_url}, model {Path(srv['model_path']).name}, context {srv['n_ctx']}")
    else:
        print(f"LLM server   : not running at {a.server_url}")
        if not problems:
            print(f"               will start one from {SERVER_EXE.relative_to(ROOT)} with {Path(a.model).name}")
    print("LLM step     : " + ("ready" if not problems else "NOT READY"))
    for p in problems:
        print(f"  - {p}")
    return {"toolchain": tc, "llm_problems": problems, "server": srv}


# ---------------------------------------------------------------- steps
def decide_execution(a, source: Path) -> bool:
    if a.no_run:
        return False
    if source.resolve() == DEFAULT_SOURCE.resolve() or a.yes:
        return True
    print(EXEC_WARNING)
    if not sys.stdin.isatty():
        print("Non-interactive session and no --yes given: execution disabled.\n")
        return False
    return input("Run the compiled programs? [y/N] ").strip().lower() in ("y", "yes")


def step_experiment(a, source: Path, may_run: bool, llm_result: Path | None = None) -> int:
    banner("3. Compiler experiment (main.py)" + (": O0 … Oz and LLM" if llm_result else ""))
    cmd = [sys.executable, str(ROOT / "main.py"), str(source),
           "--yes" if may_run else "--no-run",
           "--output-dir", str(a.output_dir), "--results-dir", str(a.results_dir), "--no-log"]
    if llm_result is not None:
        cmd += ["--llm-result", str(llm_result)]
    return run_streaming(cmd, ROOT)[0]            # echoed live and copied into this run's log


def step_llm(a, source: Path, may_run: bool, status: dict) -> str:
    banner("2. LLM Compiler workflow (llm/get_passes.py)")
    started: subprocess.Popen | None = None
    log_path = new_log_path("llama-server")
    try:
        if status["server"]["reachable"]:
            print(f"connecting to the running server at {a.server_url} (it will not be stopped)")
        else:
            started = start_server(Path(a.model), a.server_url, a.gpu_layers, a.context, log_path)
            print(f"waiting for the model to load (up to {a.server_timeout} s) ...", flush=True)
            wait_ready(started, a.server_url, log_path, a.server_timeout)
            print("server ready")

        frontend = a.llm_frontend
        print(f"frontend: {frontend} (C -> unoptimized IR with the project clang; 'listing4' = the paper's Listing 4 form)")

        def run_get_passes(module: Path, out: Path, driver: Path | None, has_main: bool, c_frontend: str | None):
            if not may_run:
                print("differential test (--check) skipped: execution is disabled")
            elif driver is None and not has_main:
                print("differential test (--check) skipped: no main() and no driver")
            cmd = get_passes_command(module, out, a.server_url, c_frontend,
                                     check=may_run and has_main, driver=driver if may_run else None)
            rc, text = run_streaming(cmd, LLM)
            summarize_llm(out)
            if rc == 0:
                ARTIFACTS["llm"] = (out, module.name)      # where the results that count were written
            return rc, text

        results_llm = Path(a.results_dir) / "llm"
        if a.llm_input:                                  # a ready-made module (LLVM 17 must be able to read it)
            module = Path(a.llm_input).resolve()
            mtext = module.read_text(encoding="utf-8", errors="replace")
            driver = Path(a.llm_driver).resolve() if a.llm_driver else None
            rc, _ = run_get_passes(module, results_llm / module.stem, driver,
                                   re.search(r"^define[^\n]*@main\(", mtext, re.M) is not None, None)
            return OK if rc == 0 else FAILED

        has_main = re.search(r"\bmain\s*\(", source.read_text(encoding="utf-8", errors="replace")) is not None
        rc, _ = run_get_passes(source, results_llm / source.stem, None, has_main, frontend)
        return OK if rc == 0 else FAILED
    except (RuntimeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return FAILED
    finally:
        if started is not None and not a.keep_server:
            stop_server(started)
        elif started is not None:
            print(f"leaving the server running (PID {started.pid}); stop it with: Stop-Process -Id {started.pid}")


def summarize_llm(out: Path) -> None:
    r = load_result(out)
    if r is None:
        return
    s, p = r.get("summary") or {}, r.get("prediction") or {}
    print("\n--- LLM result (object size = text+data via llvm-size; NOT comparable to the experiment's instruction counts) ---")
    print(f"pipeline : {p.get('pipeline') or p.get('outcome')}   ({p.get('kind')})")
    if s:
        print(f"measured : model {s.get('model_object_size')} B | unoptimized {s.get('unoptimized_object_size')} B | "
              f"-Oz {s.get('Oz_object_size')} B | vs unoptimized {s.get('improvement_over_unoptimized_pct')}% | "
              f"vs -Oz {s.get('improvement_over_Oz_pct')}%")
    print(f"artifacts: {out}")


# ---------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the experiment and the LLM workflow with one command.")
    ap.add_argument("source", nargs="?", default=str(DEFAULT_SOURCE), help="C file (default: input/example.c)")
    ap.add_argument("--check", action="store_true", help="preflight only: report what is installed, run nothing")
    ap.add_argument("--no-experiment", action="store_true", help="skip the compiler experiment")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM workflow")
    ap.add_argument("--no-run", action="store_true", help="never execute compiled programs (no correctness checks)")
    ap.add_argument("--yes", "-y", action="store_true", help="allow executing a non-bundled source without asking")
    ap.add_argument("--dashboard", action="store_true", help="open the Streamlit dashboard when the steps are done")
    ap.add_argument("--llm-input", default=None, metavar="MODULE.ll",
                    help="give the LLM step a ready-made LLVM IR module (LLVM 17 must be able to read it) instead of compiling the C file")
    ap.add_argument("--llm-driver", default=None, metavar="DRIVER.c",
                    help="with --llm-input: C driver with main() linked against each variant for the differential test")
    ap.add_argument("--server-url", default=os.environ.get("LLM_SERVER_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL_PATH", str(DEFAULT_MODEL)), help="GGUF file used when a server must be started")
    ap.add_argument("--no-start-server", action="store_true", help="only connect to a running server; never start one")
    ap.add_argument("--keep-server", action="store_true", help="leave a server this script started running afterwards")
    ap.add_argument("--gpu-layers", type=int, default=20, help="llama-server -ngl (default 20, tuned for 6 GB VRAM)")
    ap.add_argument("--context", type=int, default=16384, help="llama-server -c (default 16384 = the model's trained context)")
    ap.add_argument("--server-timeout", type=int, default=240, help="seconds to wait for the model to load")
    ap.add_argument("--llm-frontend", choices=["listing4", "O0"], default="listing4",
                    help="how the C file becomes unoptimized IR for the LLM (default listing4: the paper's Listing 4 "
                         "form, clang -Oz -Xclang -disable-llvm-passes; O0: clang -O0 -Xclang -disable-O0-optnone)")
    ap.add_argument("--output-dir", default=str(ROOT / "output"), help="experiment per-level files")
    ap.add_argument("--results-dir", default=str(ROOT / "results"), help="experiment results; LLM artifacts go to <dir>/llm/")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):          # never crash on a character the console cannot show
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    a = parse_args(argv)
    if a.check:
        return _main(a)
    with tee_output("run_all") as log:
        print(f"log: {log}")
        rc = _main(a)
        print(f"\nlog: {log}")
        return rc


def _main(a) -> int:
    source = Path(a.source).resolve()
    if not source.is_file():
        print(f"error: source file not found: {source}", file=sys.stderr)
        return 2
    if source.suffix != ".c":
        print("error: the source must be a .c file", file=sys.stderr)
        return 2

    pre = preflight(a)
    if a.check:
        ok = pre["toolchain"] is not None and not any(st.problem for st in pre["toolchain"].status.values())
        print("\n--check: nothing was run.")
        return 0 if (ok and not pre["llm_problems"]) else 1

    todo_exp = not a.no_experiment
    todo_llm = not a.no_llm
    try:                                   # experiment needs clang; the LLM workflow needs clang, opt and llvm-size
        verify(need=TOOLS if todo_llm else ("clang",))
    except ToolchainError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 3
    if todo_llm and pre["llm_problems"]:
        print("\nerror: the LLM step cannot run (see above). Fix that, or pass --no-llm.", file=sys.stderr)
        return 3

    may_run = decide_execution(a, source) if (todo_exp or todo_llm) else False
    results: dict[str, str] = {}
    if todo_llm:                           # first: the model's pipeline is an input of the LLM level
        results["llm"] = step_llm(a, source, may_run, pre)
    else:
        results["llm"] = SKIPPED
    llm_result = None
    if results["llm"] == OK and not a.llm_input:     # a prediction for this C file (not for a separate .ll module)
        llm_result = Path(a.results_dir) / "llm" / source.stem / "result.json"
    if todo_exp:
        results["experiment"] = OK if step_experiment(a, source, may_run, llm_result) == 0 else FAILED
    else:
        results["experiment"] = SKIPPED

    banner("Summary")
    for k, v in results.items():
        print(f"  {k:<11}: {v}")
    if todo_exp:
        print(f"  experiment files : {Path(a.results_dir)}  and  {Path(a.output_dir)}")
    if "llm" in ARTIFACTS:
        print(f"  LLM artifacts    : {ARTIFACTS['llm'][0]}   (module: {ARTIFACTS['llm'][1]})")
    print("  logs             : logs/  (this run, and llama-server output if this run started a server)")
    if llm_result is not None and todo_exp:
        print("  results.csv includes the LLM level next to O0 ... Oz.")

    if a.dashboard:
        banner("4. Dashboard")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard.py")], cwd=str(ROOT))
    return 0 if FAILED not in results.values() else 1


if __name__ == "__main__":
    sys.exit(main())
