"""Run the whole experiment: compile, count, measure, test, and tabulate.

Everything in the resulting table is *measured* from real Clang output; nothing
is estimated or hard-coded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .compiler import (
    ALL_LEVELS,
    LLM_LEVEL,
    OPT_LEVELS,
    CompilationError,
    LevelArtifacts,
    check_tools,
    compile_level,
    compile_llm_level,
    find_clang,
)
from .instruction_counter import count_asm_file, count_ir_file

BASELINE = "O0"

# Correctness status values
PASS = "PASS"
FAIL = "FAIL"
BASELINE_STATUS = "BASELINE"
SKIPPED = "SKIPPED (execution disabled)"
ERROR = "ERROR"
NA = "N/A"

CSV_COLUMNS = [
    "opt_level",
    "ir_instructions",
    "asm_instructions",
    "ir_reduction_pct",
    "asm_reduction_pct",
    "ir_size_bytes",
    "asm_size_bytes",
    "exe_size_bytes",
    "compile_time_s",
    "ir_time_s",
    "asm_time_s",
    "exe_time_s",
    "correctness",
    "correctness_detail",
]


# --------------------------------------------------------------------------
# Test cases / execution
# --------------------------------------------------------------------------
@dataclass
class TestCase:
    """One run of an executable: command-line arguments plus stdin text."""

    name: str = "default"
    args: List[str] = field(default_factory=list)
    stdin: str = ""


DEFAULT_TESTS = [TestCase()]


def load_tests(path: Optional[Path]) -> List[TestCase]:
    """Load test cases from a JSON list of ``{"name", "args", "stdin"}`` objects."""
    if path is None:
        return list(DEFAULT_TESTS)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list of test cases")
    tests = []
    for i, item in enumerate(data):
        tests.append(
            TestCase(
                name=str(item.get("name", f"test{i}")),
                args=[str(a) for a in item.get("args", [])],
                stdin=str(item.get("stdin", "")),
            )
        )
    return tests


def discover_tests(source: Path) -> Optional[Path]:
    """``foo.c`` -> ``foo.tests.json`` next to it, if it exists."""
    candidate = Path(source).with_suffix(".tests.json")
    return candidate if candidate.is_file() else None


def run_executable(exe: Path, test: TestCase, timeout: int = 10, max_output: int = 1_000_000) -> Dict:
    """Run ``exe`` on one test case.  NOTE: this is *not* sandboxed."""
    with tempfile.TemporaryDirectory(prefix="llvmexp_run_") as scratch:   # private cwd
        try:
            proc = subprocess.run(
                [str(exe), *test.args],
                input=test.stdin.encode("utf-8"),
                capture_output=True,
                cwd=scratch,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "returncode": None, "stdout": b"", "stderr": b""}
        except OSError as exc:
            return {"status": f"cannot start: {exc}", "returncode": None, "stdout": b"", "stderr": b""}
    return {
        "status": "ok",
        "returncode": proc.returncode,
        "stdout": proc.stdout[:max_output],
        "stderr": proc.stderr[:max_output],
    }


def compare_runs(baseline_runs: List[Dict], runs: List[Dict], tests: List[TestCase]) -> Tuple[str, str]:
    """Compare stdout and exit code against the O0 baseline for every test."""
    mismatches = []
    for test, base, cur in zip(tests, baseline_runs, runs):
        if cur["status"] != "ok":
            mismatches.append(f"{test.name}: {cur['status']}")
        elif cur["returncode"] != base["returncode"]:
            mismatches.append(f"{test.name}: exit code {cur['returncode']} != {base['returncode']}")
        elif cur["stdout"] != base["stdout"]:
            mismatches.append(f"{test.name}: stdout differs")
    if mismatches:
        return FAIL, "; ".join(mismatches)
    return PASS, f"{len(tests)} test(s) identical to {BASELINE} (stdout + exit code)"


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    table: pd.DataFrame
    metadata: Dict
    csv_path: Path
    metadata_path: Path
    artifacts: Dict[str, LevelArtifacts]
    test_outputs: Dict[str, List[Dict]]


def reduction_pct(baseline: float, value: float) -> float:
    """Reduction (%) = (baseline - value) / baseline * 100; NaN if baseline is 0."""
    if baseline is None or baseline == 0 or pd.isna(baseline) or pd.isna(value):
        return float("nan")
    return (baseline - value) / baseline * 100.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _size(path: Path) -> float:
    return float(Path(path).stat().st_size) if Path(path).is_file() else float("nan")


def run_experiment(
    source: Path,
    levels: Sequence[str] = tuple(OPT_LEVELS),
    output_root: Path = Path("output"),
    results_dir: Path = Path("results"),
    clang: Optional[str] = None,
    extra_flags: Sequence[str] = (),
    run_tests: bool = True,
    tests: Optional[List[TestCase]] = None,
    repeats: int = 3,
    run_timeout: int = 10,
    progress: Optional[Callable[[str], None]] = None,
    llm: Optional[Dict] = None,
) -> ExperimentResult:
    """Compile ``source`` at every level and build the results table.

    ``O0`` is always included because it is the baseline for the reduction
    percentages and the correctness comparison.

    The level ``"LLM"`` is measured exactly like the others; it needs ``llm``, the LLM Compiler prediction for this
    source (``{"pipeline": str | None, ...}``, see ``src.llm_runner.prediction_for_experiment``). If the LLM level
    cannot be built, the other levels are still reported and the reason is in ``metadata["llm"]["error"]``.
    """
    log = progress or (lambda msg: None)
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    wanted = [lv for lv in ALL_LEVELS if lv in set(levels) or lv == BASELINE]
    unknown = set(levels) - set(ALL_LEVELS)
    if unknown:
        raise ValueError(f"Unknown optimization level(s): {sorted(unknown)}; choose from {ALL_LEVELS}")
    if LLM_LEVEL in wanted and llm is None:
        raise ValueError("the LLM level needs the LLM Compiler prediction (llm=...)")
    llm_meta = dict(llm) if llm else None

    clang_path = find_clang(clang)                  # raises LLVMNotFoundError with help text
    info = check_tools(clang_path)
    log(f"Using {info['clang_version']}  (target {info['target_triple']}; project toolchain {info.get('toolchain_config', '?')}, "
        f"LLVM {info.get('llvm_version_required', '?')})")
    if info.get("warnings"):
        log(f"WARNING: {info['warnings']}")

    output_root, results_dir = Path(output_root), Path(results_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    tests = tests or list(DEFAULT_TESTS)

    rows: List[Dict] = []
    artifacts: Dict[str, LevelArtifacts] = {}
    test_outputs: Dict[str, List[Dict]] = {}

    for level in list(wanted):
        log(f"[{level}] compiling ...")
        if level == LLM_LEVEL:
            try:
                from .toolchain import verify              # opt comes from the same toolchain.json, version-checked
                opt_path = str(verify(need=("clang", "opt")).opt)
                art = compile_llm_level(clang_path, opt_path, source, llm.get("pipeline"), output_root, extra_flags,
                                        repeats=repeats)
            except (CompilationError, RuntimeError) as exc:
                detail = getattr(exc, "stderr", "") or ""
                llm_meta["error"] = (str(exc) + "\n" + detail).strip()[:2000]
                log(f"[{level}] ERROR: {exc}; the LLM level is left out of this run")
                wanted.remove(level)
                continue
        else:
            art = compile_level(clang_path, source, level, output_root, extra_flags, repeats=repeats)
        artifacts[level] = art

        ir = count_ir_file(art.ir_path)
        asm = count_asm_file(art.asm_path)
        row: Dict = {
            "opt_level": level,
            "ir_instructions": ir.total,
            "asm_instructions": asm.total,
            "ir_size_bytes": _size(art.ir_path),
            "asm_size_bytes": _size(art.asm_path),
            "exe_size_bytes": _size(art.exe_path) if art.exe_built else float("nan"),
            # Headline compile time = source -> assembly (front end + optimizer +
            # code generator, no linker).  The others are kept for reference.
            "compile_time_s": art.asm_time_s,
            "ir_time_s": art.ir_time_s,
            "asm_time_s": art.asm_time_s,
            "exe_time_s": art.exe_time_s,
            "correctness": NA,
            "correctness_detail": "",
        }
        if not art.exe_built:
            row["correctness_detail"] = f"executable not built: {art.exe_error[:300]}"
            log(f"[{level}] WARNING: executable was not built ({art.exe_error.splitlines()[-1] if art.exe_error else 'unknown error'})")
        rows.append(row)

    # ---- correctness ----------------------------------------------------
    by_level = {r["opt_level"]: r for r in rows}
    if not run_tests:
        for r in rows:
            r["correctness"] = SKIPPED
    else:
        log("Running executables on the test inputs (not sandboxed) ...")
        for level in wanted:
            art = artifacts[level]
            if not art.exe_built:
                by_level[level]["correctness"] = ERROR
                continue
            test_outputs[level] = [run_executable(art.exe_path, t, run_timeout) for t in tests]

        base_runs = test_outputs.get(BASELINE)
        base_ok = base_runs is not None and all(r["status"] == "ok" for r in base_runs)
        for level in wanted:
            r = by_level[level]
            if r["correctness"] == ERROR:
                continue
            if not base_ok:
                r["correctness"] = NA
                r["correctness_detail"] = "O0 baseline could not be executed, nothing to compare to"
            elif level == BASELINE:
                r["correctness"] = BASELINE_STATUS
                r["correctness_detail"] = f"reference output ({len(tests)} test(s))"
            else:
                r["correctness"], r["correctness_detail"] = compare_runs(base_runs, test_outputs[level], tests)

    table = pd.DataFrame(rows)
    base = by_level[BASELINE]
    table["ir_reduction_pct"] = [reduction_pct(base["ir_instructions"], v) for v in table["ir_instructions"]]
    table["asm_reduction_pct"] = [reduction_pct(base["asm_instructions"], v) for v in table["asm_instructions"]]
    table = table[CSV_COLUMNS]

    csv_path = results_dir / "results.csv"
    table.to_csv(csv_path, index=False, float_format="%.6g")

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": source.name,
        "source_sha256": _sha256(source),
        "levels": wanted,
        "extra_flags": list(extra_flags),
        "timing_repeats_median_of": repeats,
        "executed_tests": run_tests,
        "tests": [t.__dict__ for t in tests] if run_tests else [],
        "toolchain": {k: v for k, v in info.items() if k != "clang_version_full"},
        "clang_version_full": info["clang_version_full"],
        "commands": {lv: artifacts[lv].commands for lv in wanted},
        "llm": llm_meta,
        "notes": [
            "IR instruction counts and assembly instruction counts are different units and are not comparable.",
            "Fewer instructions does not necessarily mean faster execution.",
            "compile_time_s is wall-clock source->assembly (median of repeats) and is noisy for tiny programs.",
            "Toolchain: the project-local LLVM toolchain from toolchain.json; clang, opt and llvm-size versions are "
            "recorded under 'toolchain'. Results from another compiler version are not comparable.",
        ] + ([
            "LLM level: the C file is compiled to the paper's Listing 4 IR (clang -Oz -Xclang -disable-llvm-passes), "
            "the pass pipeline predicted by LLM Compiler 7B is applied with LLVM 17 opt, and clang generates code at "
            "-Oz without re-running IR passes. It is -Oz with the IR pipeline replaced by the model's. Its compile "
            "times cover front end + opt + code generation; model inference time is not included.",
        ] if llm_meta else []),
    }
    metadata_path = results_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return ExperimentResult(table, metadata, csv_path, metadata_path, artifacts, test_outputs)


def format_table(table: pd.DataFrame) -> str:
    """Human-readable results table (used by the CLI and the README)."""
    t = table.copy()
    t = t.rename(
        columns={
            "opt_level": "Level",
            "ir_instructions": "IR instr",
            "asm_instructions": "Asm instr",
            "ir_reduction_pct": "IR red.%",
            "asm_reduction_pct": "Asm red.%",
            "ir_size_bytes": "IR bytes",
            "asm_size_bytes": "Asm bytes",
            "exe_size_bytes": "Exe bytes",
            "compile_time_s": "Compile s",
            "correctness": "Correct",
        }
    )
    cols = ["Level", "IR instr", "Asm instr", "IR red.%", "Asm red.%", "IR bytes", "Asm bytes", "Exe bytes", "Compile s", "Correct"]
    t = t[cols]
    for c in ("IR red.%", "Asm red.%"):
        t[c] = t[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}")
    for c in ("IR bytes", "Asm bytes", "Exe bytes"):
        t[c] = t[c].map(lambda v: "n/a" if pd.isna(v) else f"{int(v)}")
    t["Compile s"] = t["Compile s"].map(lambda v: f"{v:.3f}")
    return t.to_string(index=False)
