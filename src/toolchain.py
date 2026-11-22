"""The project's single LLVM toolchain definition (``toolchain.json`` in the project root).

Every active workflow gets its tools from here:

* the compiler experiment (``src/compiler.py``, used by ``main.py``, ``dashboard.py`` and ``run_all.py``);
* the LLM Compiler workflow (``llm/get_passes.py``, alone or through ``run_all.py``).

Rules
  * Tool paths come only from ``toolchain.json`` (relative to the project root). Nothing is looked up on PATH, in
    environment variables or in system install folders, so a globally installed clang is never used by accident.
  * Each tool is run with ``--version`` and must report exactly the required LLVM version; clang must also target
    the configured triple. Anything else raises :class:`ToolchainError`, and nothing is compiled. There is no
    fallback to another compiler.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "toolchain.json"
TOOLS = ("clang", "opt", "llvm_size")
_VERSION_RE = re.compile(r"(?:clang|LLVM) version (\d+\.\d+\.\d+)")
_TARGET_RE = re.compile(r"^Target:\s*(\S+)", re.M)

INSTALL_HELP = """\
The project uses one project-local LLVM 17.0.6 toolchain (toolchain.json):
  clang + llvm-size : toolchains/llvm-mingw-17.0.6-ucrt/   (llvm-mingw 20231128 = LLVM 17.0.6, UCRT, x86_64)
  opt               : llm/tools/llvm17/bin/opt.exe        (LLVM 17.0.6)
To (re)install clang + llvm-size: download
  https://github.com/mstorsjo/llvm-mingw/releases/download/20231128/llvm-mingw-20231128-ucrt-x86_64.zip
and extract its folder 'llvm-mingw-20231128-ucrt-x86_64' as toolchains/llvm-mingw-17.0.6-ucrt/.
Do not put it on PATH. No other clang is used as a fallback. Details: README.md, section 2."""


class ToolchainError(RuntimeError):
    """The configured toolchain is missing, the wrong version, or targets the wrong platform."""


@dataclass(frozen=True)
class ToolStatus:
    name: str
    path: Path
    version: Optional[str]
    version_line: str
    problem: Optional[str] = None


@dataclass(frozen=True)
class Toolchain:
    llvm_version: str
    target_triple: str
    config_path: Path
    status: Dict[str, ToolStatus]

    @property
    def clang(self) -> Path:
        return self.status["clang"].path

    @property
    def opt(self) -> Path:
        return self.status["opt"].path

    @property
    def llvm_size(self) -> Path:
        return self.status["llvm_size"].path

    def info(self) -> Dict[str, str]:
        """Flat description for metadata files and the dashboard."""
        out = {"toolchain_config": _display(self.config_path), "llvm_version_required": self.llvm_version,
               "target_triple_required": self.target_triple}
        for name, st in self.status.items():          # prefixed: must not collide with compiler.get_tool_info keys
            out[f"toolchain_{name}_path"] = str(st.path)
            out[f"toolchain_{name}_version"] = st.version or "unknown"
        return out


def _display(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_config(path: Path = CONFIG_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ToolchainError(f"toolchain configuration missing: {path}\n\n{INSTALL_HELP}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ToolchainError(f"{path} is not valid JSON: {exc}") from None
    missing = [k for k in ("llvm_version", "target_triple", *TOOLS) if not cfg.get(k)]
    if missing:
        raise ToolchainError(f"{path} lacks: {', '.join(missing)}")
    return cfg


def resolve(value: str, base: Path = ROOT) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _version_output(path: Path) -> str:
    try:
        r = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<could not run: {exc}>"
    return (r.stdout or "") + (r.stderr or "")


def probe(name: str, path: Path, required: str, target: Optional[str] = None) -> ToolStatus:
    """Run ``path --version`` and check it against the requirement."""
    path = Path(path)
    if not path.is_file():
        return ToolStatus(name, path, None, "", f"missing: {path}")
    out = _version_output(path)
    m = _VERSION_RE.search(out)
    version = m.group(1) if m else None
    line = next((l.strip() for l in out.splitlines() if "version" in l.lower()), out.strip()[:120])
    problem = None
    if version is None:
        problem = f"could not read a version from `{path.name} --version` ({line!r})"
    elif version != required:
        problem = f"version {version}, but the project requires {required}"
    elif target is not None:
        t = _TARGET_RE.search(out)
        found = t.group(1) if t else None
        if found != target:
            problem = f"targets {found}, but the project requires {target}"
    return ToolStatus(name, path, version, line, problem)


def inspect(clang_override: Optional[str] = None, config_path: Path = CONFIG_PATH) -> Toolchain:
    """Probe every configured tool. Never raises for tool problems (see ``status[...].problem``)."""
    config_path = Path(config_path)
    cfg = load_config(config_path)
    base = config_path.resolve().parent
    paths = {t: resolve(cfg[t], base) for t in TOOLS}
    if clang_override:
        paths["clang"] = Path(clang_override)
    status = {t: probe(t, paths[t], cfg["llvm_version"], cfg["target_triple"] if t == "clang" else None)
              for t in TOOLS}
    return Toolchain(cfg["llvm_version"], cfg["target_triple"], config_path, status)


def verify(need: Iterable[str] = TOOLS, clang_override: Optional[str] = None,
           config_path: Path = CONFIG_PATH) -> Toolchain:
    """Return the toolchain, or raise ToolchainError.

    Every tool in ``need`` must be present with the required version. A tool outside ``need`` may be absent, but if it
    is present with a different version the run is refused as well: a mixed toolchain is never used.
    """
    tc = inspect(clang_override, config_path)
    need = set(need)
    problems = [f"  - {n}: {st.problem}" for n, st in tc.status.items()
                if st.problem and (n in need or st.version is not None)]
    if problems:
        raise ToolchainError(
            f"The project's LLVM toolchain is not usable ({_display(tc.config_path)} requires LLVM {tc.llvm_version}, "
            f"target {tc.target_triple}):\n" + "\n".join(problems) + "\n\n" + INSTALL_HELP)
    return tc


def report(tc: Toolchain) -> str:
    """Human-readable table of the toolchain (used by run_all.py and main.py)."""
    rows = [f"Toolchain ({_display(tc.config_path)}): LLVM {tc.llvm_version}, target {tc.target_triple}"]
    for name, st in tc.status.items():
        state = "OK" if st.problem is None else f"PROBLEM: {st.problem}"
        rows.append(f"  {name:<10} {st.version or '-':<8} {_display(st.path)}  [{state}]")
    return "\n".join(rows)
