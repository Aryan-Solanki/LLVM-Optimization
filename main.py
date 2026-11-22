#!/usr/bin/env python3
"""Command-line entry point.

    python main.py                          # run on input/example.c
    python main.py path/to/prog.c           # your own file
    python main.py prog.c --no-run          # never execute the compiled programs
    python main.py prog.c --levels O0 O2 Oz
    python main.py prog.c --llm-result results/llm/prog/result.json   # adds the LLM level (model's pipeline)

The compiler is always the project's LLVM 17.0.6 clang from toolchain.json (verified before every run);
no clang from PATH or an environment variable is ever used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.analyzer import discover_tests, format_table, load_tests, run_experiment
from src.compiler import LLM_LEVEL, OPT_LEVELS, CompilationError, LLVMNotFoundError, find_clang
from src.llm_runner import load_result, prediction_for_experiment
from src.runlog import tee_output
from src.toolchain import inspect, report
from src.visualizer import generate_all

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "input" / "example.c"

EXEC_WARNING = """\
WARNING: correctness testing COMPILES AND RUNS the C program you gave me, once
per optimization level.  The programs run with your user permissions and are NOT
sandboxed.  Only run source code you wrote or trust.  Use --no-run to skip this.
"""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare LLVM/Clang optimization levels on a C file.")
    p.add_argument("source", nargs="?", default=str(DEFAULT_SOURCE), help="C source file (default: input/example.c)")
    p.add_argument("--levels", nargs="+", default=OPT_LEVELS, choices=OPT_LEVELS, metavar="LEVEL",
                   help=f"optimization levels (default: all of {OPT_LEVELS}); O0 is always added as the baseline")
    p.add_argument("--no-run", action="store_true", help="do NOT execute the compiled programs (skips correctness testing)")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation before executing a non-bundled source file")
    p.add_argument("--tests", help="JSON file with test cases (default: <source>.tests.json if present)")
    p.add_argument("--flag", action="append", default=[], dest="flags", metavar="FLAG",
                   help="extra compiler flag applied to every level, e.g. --flag=-march=native (repeatable)")
    p.add_argument("--repeats", type=int, default=3, help="compile-time measurements per step; the median is used (default 3)")
    p.add_argument("--timeout", type=int, default=10, help="seconds allowed per test run (default 10)")
    p.add_argument("--output-dir", default=str(ROOT / "output"), help="where per-level files are written")
    p.add_argument("--results-dir", default=str(ROOT / "results"), help="where CSV, metadata and charts are written")
    p.add_argument("--no-plots", action="store_true", help="skip chart generation")
    p.add_argument("--no-log", action="store_true", help="do not write logs/<time>_main.log (run_all.py logs instead)")
    p.add_argument("--llm-result", metavar="RESULT.json",
                   help="add the LLM level: -Oz with the pass pipeline LLM Compiler predicted for this source "
                        "(result.json written by llm/get_passes.py; run_all.py passes it automatically)")
    return p.parse_args(argv)


def confirm_execution(args, source: Path) -> bool:
    """Decide whether executables may run.  Default is cautious for foreign files."""
    if args.no_run:
        return False
    if source.resolve() == DEFAULT_SOURCE.resolve() or args.yes:
        return True
    print(EXEC_WARNING)
    if not sys.stdin.isatty():
        print("Non-interactive session and no --yes given: execution disabled (compile-only).\n")
        return False
    answer = input("Run the compiled programs? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.no_log:
        return _main(args)
    with tee_output("main") as log:
        rc = _main(args)
        print(f"Log      : {log}")
        return rc


def _main(args) -> int:
    source = Path(args.source)
    if not source.is_file():
        print(f"error: source file not found: {source}", file=sys.stderr)
        return 2

    try:                                   # fail fast, before asking any questions
        clang = find_clang()               # the project toolchain (toolchain.json), version-checked
    except LLVMNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(report(inspect()))

    run_tests = confirm_execution(args, source)
    tests_path = Path(args.tests) if args.tests else discover_tests(source)
    tests = load_tests(tests_path)
    if run_tests:
        print(f"Test inputs: {len(tests)} case(s) from {tests_path or 'built-in default (no args, empty stdin)'}")
    else:
        print("Execution disabled: correctness will be reported as SKIPPED.")

    levels, llm = list(args.levels), None
    if args.llm_result:
        r = load_result(Path(args.llm_result).parent)
        if r is None:
            print(f"error: cannot read {args.llm_result}", file=sys.stderr)
            return 2
        if Path(r.get("input", "")).resolve() != source.resolve():
            print(f"error: {args.llm_result} is a prediction for {r.get('input')}, not for {source}", file=sys.stderr)
            return 2
        llm, why = prediction_for_experiment(r, Path(args.llm_result))
        if llm is None:
            print(f"warning: LLM level skipped: {why}", file=sys.stderr)
        else:
            levels.append(LLM_LEVEL)
            print(f"LLM level: pipeline {llm['pipeline'] or '(none)'} from {args.llm_result}")

    try:
        result = run_experiment(
            source,
            levels=levels,
            llm=llm,
            output_root=Path(args.output_dir),
            results_dir=Path(args.results_dir),
            clang=clang,
            extra_flags=args.flags,
            run_tests=run_tests,
            tests=tests,
            repeats=args.repeats,
            run_timeout=args.timeout,
            progress=lambda m: print(m, flush=True),
        )
    except LLVMNotFoundError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 3
    except CompilationError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        if exc.command:
            print("command: " + " ".join(exc.command), file=sys.stderr)
        print(exc.stderr, file=sys.stderr)
        return 4

    charts = {} if args.no_plots else generate_all(result.table, Path(args.results_dir))

    tc = result.metadata["toolchain"]
    print("\n=== Environment ===")
    print(f"Compiler : {tc['clang_version']}")
    print(f"Target   : {tc['target_triple']}  (architecture: {tc['target_architecture']})")
    print(f"Host     : {tc['host_os']} / {tc['host_machine']}")
    print(f"Source   : {result.metadata['source_file']}  sha256={result.metadata['source_sha256'][:16]}...")
    print("\n=== Results ===")
    print(format_table(result.table))
    print(
        "\nNotes: 'IR instr' counts LLVM IR instructions; 'Asm instr' counts machine instructions."
        "\n       They are different units. Fewer instructions does not necessarily mean faster execution."
    )
    if (result.metadata.get("llm") or {}).get("error"):
        print(f"\nwarning: the LLM level was left out: {result.metadata['llm']['error']}", file=sys.stderr)
    print(f"\nCSV      : {result.csv_path}")
    print(f"Metadata : {result.metadata_path}")
    for p in charts.values():
        print(f"Chart    : {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
