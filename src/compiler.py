"""Compile a C file at several optimization levels with the project's LLVM 17.0.6 toolchain.

For every level this module produces three artefacts, all from the *same*
source file and the *same* Clang binary:

    output/<LEVEL>/output_<LEVEL>.ll    LLVM IR   (clang -S -emit-llvm)
    output/<LEVEL>/output_<LEVEL>.s     assembly  (clang -S)
    output/<LEVEL>/output_<LEVEL>[.exe] native executable

The clang used is the one in ``toolchain.json`` (see ``src/toolchain.py``), verified to be exactly the required
version and target before anything is compiled. Clang is never taken from PATH or an environment variable.

Clang is always run with the source file's directory as its working directory
and the bare file name as argument, so ``source_filename`` inside the IR does
not depend on where the project lives (better reproducibility).
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .toolchain import INSTALL_HELP, ToolchainError, inspect, verify

# Each level is passed verbatim to clang as -<level>.  Os and Oz are distinct
# clang flags (Os: optimize for size; Oz: size even more aggressively).
OPT_LEVELS: List[str] = ["O0", "O1", "O2", "O3", "Os", "Oz"]
EXE_SUFFIX = ".exe" if os.name == "nt" else ""


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
# Kept under its old name so existing callers keep working: raised when the project toolchain is missing,
# has the wrong version or the wrong target.
LLVMNotFoundError = ToolchainError


class CompilationError(RuntimeError):
    """Raised when Clang rejects the source (or an essential step fails)."""

    def __init__(self, message: str, command: Optional[Sequence[str]] = None, stderr: str = ""):
        super().__init__(message)
        self.command = list(command) if command else None
        self.stderr = stderr


# --------------------------------------------------------------------------
# The project's clang
# --------------------------------------------------------------------------
def find_clang(explicit: Optional[str] = None) -> str:
    """Return the verified clang of the project toolchain (``toolchain.json``).

    ``explicit`` replaces the configured clang path (used by the test suite with a test double); it must pass the
    same version and target check. Raises :class:`LLVMNotFoundError` (= ToolchainError) with install help.
    """
    return str(verify(need=("clang",), clang_override=explicit).clang)


# --------------------------------------------------------------------------
# Tool / environment information
# --------------------------------------------------------------------------
def _run_text(cmd: Sequence[str], timeout: int = 30) -> str:
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "").strip()


def get_tool_info(clang: str) -> Dict[str, str]:
    """Compiler and LLVM tool versions, target triple and host details, for metadata and the dashboard."""
    version_out = _run_text([clang, "--version"])
    first = version_out.splitlines()[0] if version_out else "unknown"
    target = ""
    for line in version_out.splitlines():
        if line.lower().startswith("target:"):
            target = line.split(":", 1)[1].strip()
    if not target:
        target = _run_text([clang, "-dumpmachine"]) or "unknown"

    tc = inspect(clang_override=clang)
    info: Dict[str, str] = {
        "clang_path": str(clang),
        "clang_version": first,
        "clang_version_full": version_out,
        "target_triple": target,
        "target_architecture": target.split("-")[0] if target else "unknown",
        "host_os": f"{platform.system()} {platform.release()}",
        "host_machine": platform.machine(),
        "python_version": sys.version.split()[0],
    }
    info.update(tc.info())
    # tool_* keys: the LLVM tools of the same toolchain (shown in the dashboard's "Run details").
    for name, st in tc.status.items():
        info[f"tool_{name.replace('_', '-')}"] = st.version or ("not found" if st.problem else "unknown")
    return info


def check_tools(clang: str) -> Dict[str, str]:
    """Tool info for a clang that already passed :func:`find_clang`; refuses a mixed toolchain."""
    info = get_tool_info(clang)
    verify(need=("clang",), clang_override=clang)     # raises if any present tool has another version
    info["warnings"] = ""
    return info


# --------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------
@dataclass
class LevelArtifacts:
    """Files and timings produced for one optimization level."""

    level: str
    directory: Path
    ir_path: Path
    asm_path: Path
    exe_path: Path
    commands: Dict[str, List[str]] = field(default_factory=dict)
    ir_time_s: float = float("nan")
    asm_time_s: float = float("nan")
    exe_time_s: float = float("nan")
    exe_built: bool = False
    exe_error: str = ""


def _timed_run(cmd: Sequence[str], cwd: Path, repeats: int, timeout: int):
    """Run ``cmd`` ``repeats`` times; return (median wall seconds, last CompletedProcess)."""
    times: List[float] = []
    proc = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        proc = subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        times.append(time.perf_counter() - start)
        if proc.returncode != 0:
            break
    return statistics.median(times), proc


def compile_level(
    clang: str,
    source: Path,
    level: str,
    output_root: Path,
    extra_flags: Sequence[str] = (),
    link_flags: Optional[Sequence[str]] = None,
    repeats: int = 3,
    timeout: int = 300,
) -> LevelArtifacts:
    """Compile ``source`` at ``-<level>`` into IR, assembly and an executable.

    Raises :class:`CompilationError` if IR or assembly generation fails (that
    means the source is not valid C).  A failing *link* step is not fatal: it is
    recorded in ``exe_error`` so IR/assembly results are still usable.
    """
    if level not in OPT_LEVELS:
        raise ValueError(f"Unknown optimization level {level!r}; expected one of {OPT_LEVELS}")

    source = Path(source).resolve()
    out_dir = (Path(output_root) / level).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    art = LevelArtifacts(
        level=level,
        directory=out_dir,
        ir_path=out_dir / f"output_{level}.ll",
        asm_path=out_dir / f"output_{level}.s",
        exe_path=out_dir / f"output_{level}{EXE_SUFFIX}",
    )
    for stale in (art.ir_path, art.asm_path, art.exe_path):   # never mix runs
        if stale.exists():
            stale.unlink()

    opt = f"-{level}"
    src_arg = source.name                   # run in the source dir -> stable IR header
    cwd = source.parent
    extra = list(extra_flags)
    if link_flags is None:
        link_flags = [] if os.name == "nt" else ["-lm"]

    art.commands["ir"] = [clang, opt, "-S", "-emit-llvm", *extra, src_arg, "-o", str(art.ir_path)]
    art.commands["asm"] = [clang, opt, "-S", *extra, src_arg, "-o", str(art.asm_path)]
    art.commands["exe"] = [clang, opt, *extra, src_arg, "-o", str(art.exe_path), *link_flags]

    art.ir_time_s, proc = _timed_run(art.commands["ir"], cwd, repeats, timeout)
    if proc.returncode != 0:
        raise CompilationError(f"clang failed generating LLVM IR at {opt}", art.commands["ir"], proc.stderr)

    art.asm_time_s, proc = _timed_run(art.commands["asm"], cwd, repeats, timeout)
    if proc.returncode != 0:
        raise CompilationError(f"clang failed generating assembly at {opt}", art.commands["asm"], proc.stderr)

    art.exe_time_s, proc = _timed_run(art.commands["exe"], cwd, repeats, timeout)
    if proc.returncode == 0 and art.exe_path.exists():
        art.exe_built = True
    else:
        art.exe_error = (proc.stderr or "linking failed").strip()
    return art


# --------------------------------------------------------------------------
# The "LLM" level: -Oz with the IR pass pipeline predicted by LLM Compiler
# --------------------------------------------------------------------------
LLM_LEVEL = "LLM"
ALL_LEVELS: List[str] = [*OPT_LEVELS, LLM_LEVEL]
# The paper's Listing 4 input form (same flags as llm/get_passes.py FRONTEND_FLAGS["listing4"]): clang at -Oz with
# the IR optimization passes disabled. Used for the front end and, with -x ir, for code generation, so the back
# end runs at -Oz exactly as for the "Oz" level; only the IR pass pipeline differs (the model's instead of -Oz's).
LLM_CLANG_FLAGS = ["-Oz", "-Xclang", "-disable-llvm-passes"]


def compile_llm_level(
    clang: str,
    opt: str,
    source: Path,
    pipeline: Optional[str],
    output_root: Path,
    extra_flags: Sequence[str] = (),
    link_flags: Optional[Sequence[str]] = None,
    repeats: int = 3,
    timeout: int = 300,
) -> LevelArtifacts:
    """Build the "LLM" level: C -> Listing 4 IR -> ``opt -p <pipeline>`` -> assembly and executable.

    ``pipeline`` is the model's prediction exactly as written (``None`` = the model said no passes). Times are
    measured like the other levels and cover front end + opt (+ code generation); model inference is not included.
    """
    source = Path(source).resolve()
    out_dir = (Path(output_root) / LLM_LEVEL).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art = LevelArtifacts(
        level=LLM_LEVEL,
        directory=out_dir,
        ir_path=out_dir / f"output_{LLM_LEVEL}.ll",
        asm_path=out_dir / f"output_{LLM_LEVEL}.s",
        exe_path=out_dir / f"output_{LLM_LEVEL}{EXE_SUFFIX}",
    )
    input_ll = out_dir / f"input_{LLM_LEVEL}.ll"
    for stale in (input_ll, art.ir_path, art.asm_path, art.exe_path):
        if stale.exists():
            stale.unlink()

    cwd = source.parent
    extra = list(extra_flags)
    if link_flags is None:
        link_flags = [] if os.name == "nt" else ["-lm"]

    art.commands["frontend"] = [clang, *LLM_CLANG_FLAGS, "-S", "-emit-llvm", *extra, source.name, "-o", str(input_ll)]
    # opt runs inside out_dir with bare file names, so the IR's ModuleID is "input_LLM.ll", not a machine path
    art.commands["ir"] = [opt, input_ll.name, "-S", "-o", art.ir_path.name, *(["-p", pipeline] if pipeline else [])]
    art.commands["asm"] = [clang, *LLM_CLANG_FLAGS, "-S", "-x", "ir", str(art.ir_path), "-o", str(art.asm_path)]
    art.commands["exe"] = [clang, *LLM_CLANG_FLAGS, "-x", "ir", str(art.ir_path), "-o", str(art.exe_path), *link_flags]

    fe_time, proc = _timed_run(art.commands["frontend"], cwd, repeats, timeout)
    if proc.returncode != 0:
        raise CompilationError("clang failed generating the Listing 4 IR for the LLM level", art.commands["frontend"], proc.stderr)
    opt_time, proc = _timed_run(art.commands["ir"], out_dir, repeats, timeout)
    if proc.returncode != 0:
        raise CompilationError("opt failed applying the LLM pipeline", art.commands["ir"], proc.stderr)
    art.ir_time_s = fe_time + opt_time

    asm_time, proc = _timed_run(art.commands["asm"], cwd, repeats, timeout)
    if proc.returncode != 0:
        raise CompilationError("clang failed generating assembly for the LLM level", art.commands["asm"], proc.stderr)
    art.asm_time_s = art.ir_time_s + asm_time

    exe_time, proc = _timed_run(art.commands["exe"], cwd, repeats, timeout)
    art.exe_time_s = art.ir_time_s + exe_time
    if proc.returncode == 0 and art.exe_path.exists():
        art.exe_built = True
    else:
        art.exe_error = (proc.stderr or "linking failed").strip()
    return art
