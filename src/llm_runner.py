"""Shared runner for the LLM Compiler workflow, used by ``run_all.py`` and by the dashboard.

This module only *orchestrates* ``llm/get_passes.py`` (the paper's method: prompt, greedy decoding, LLVM 17
validation, object-size measurement). It never changes that method: it starts the script as a subprocess with
the same arguments ``run_all.py`` always used, and reads the ``result.json`` the script writes.

Server lifecycle: if no llama-server is reachable at the URL, one is started from ``llm/llama.cpp`` with the
llama.cpp settings for LLM Compiler (``-c 16384 --temp 0`` with the Meta chat template) and stopped again afterwards. Only the process started
here is ever stopped; a server that was already running is left alone.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
LLM = ROOT / "llm"
SERVER_EXE = LLM / "llama.cpp" / "llama-server.exe"
TEMPLATE = LLM / "templates" / "llm-compiler.jinja"
GET_PASSES = LLM / "get_passes.py"
DEFAULT_MODEL = LLM / "models" / "llm-compiler-7b-ftd.Q4_K_M.gguf"
ALIAS = "llm-compiler-7b-ftd"

DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
DEFAULT_GPU_LAYERS = 20           # tuned for 6 GB VRAM
DEFAULT_CONTEXT = 16384           # the model's trained context
DEFAULT_SERVER_TIMEOUT = 240      # seconds to wait for the model to load

FRONTENDS = ("listing4", "O0")    # see get_passes.py --frontend

Echo = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- server
def http_json(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def server_status(url: str) -> dict:
    """{'reachable': bool, 'model_path', 'is_llm_compiler', 'n_ctx'}; never raises."""
    try:
        props = http_json(url.rstrip("/") + "/props", timeout=3)
    except (OSError, ValueError):
        return {"reachable": False}
    path = str(props.get("model_path") or "")
    return {"reachable": True, "model_path": path, "is_llm_compiler": "llm-compiler" in path.lower(),
            "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx")}


def is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in ("127.0.0.1", "localhost", "::1")


def llm_problems(server_url: str, model, srv: dict, no_start_server: bool = False) -> list[str]:
    """Why the LLM step cannot run (empty list = ready)."""
    problems = []
    if srv.get("reachable"):
        if not srv.get("is_llm_compiler"):
            problems.append(f"the server at {server_url} is not serving LLM Compiler (model: {srv.get('model_path')}); "
                            "it is never stopped by this project. Stop it yourself or use another server URL.")
    elif no_start_server:
        problems.append(f"no server at {server_url} and starting one is disabled")
    elif not is_local(server_url):
        problems.append(f"no server at {server_url}, and a non-local URL is never started by this project")
    else:
        for p, what in ((SERVER_EXE, "llama-server.exe"), (TEMPLATE, "chat template")):
            if not p.is_file():
                problems.append(f"{what} missing: {p}")
        if not Path(model).is_file():
            problems.append(f"model file missing: {model}  (set --model or LLM_MODEL_PATH)")
    if not GET_PASSES.is_file():
        problems.append(f"get_passes.py missing: {GET_PASSES}")
    return problems


def start_server(model: Path, url: str, gpu_layers: int, context: int, log_path: Path,
                 echo: Echo = _print) -> subprocess.Popen:
    """Start llama-server in the background (LLM Compiler settings: trained context 16384, temperature 0, Meta [INST] template)."""
    port = urlparse(url).port or 8080
    cmd = [str(SERVER_EXE), "-m", str(model), "--alias", ALIAS, "-ngl", str(gpu_layers), "-c", str(context),
           "--host", "127.0.0.1", "--port", str(port), "--jinja", "--chat-template-file", str(TEMPLATE), "--temp", "0"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "wb")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    echo(f"starting: {' '.join(cmd)}\nserver log: {log_path}")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)


def wait_ready(proc: subprocess.Popen, url: str, log_path: Path, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:]
            raise RuntimeError(f"llama-server exited with code {proc.returncode} while starting:\n  " + "\n  ".join(tail))
        try:
            if http_json(url.rstrip("/") + "/health", timeout=3).get("status") == "ok":
                return
        except (OSError, ValueError):
            pass                                   # still loading (503) or not listening yet
        time.sleep(2)
    raise RuntimeError(f"llama-server did not become ready within {timeout} s (see {log_path})")


def stop_server(proc: subprocess.Popen, echo: Echo = _print) -> None:
    """Stop ONLY the given process (one this project started)."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)
    echo(f"stopped the llama-server this run started (PID {proc.pid})")


# ---------------------------------------------------------------- get_passes.py
def has_main_c(source: Path) -> bool:
    return re.search(r"\bmain\s*\(", source.read_text(encoding="utf-8", errors="replace")) is not None


def get_passes_command(module: Path, out: Path, server_url: str, frontend: Optional[str] = None,
                       check: bool = False, driver: Optional[Path] = None) -> list:
    """The exact command run_all.py has always used (``-I``: isolated mode, ``-B``: no .pyc files)."""
    cmd = [sys.executable, "-B", "-I", str(GET_PASSES), str(module), "--evaluate",
           "--server", server_url, "--out", str(out)]
    if frontend:
        cmd += ["--frontend", frontend]
    if driver is not None:
        cmd += ["--check", str(driver)]
    elif check:
        cmd.append("--check")
    return cmd


def run_streaming(cmd: list, cwd: Path, on_line: Optional[Callable[[str], None]] = None) -> tuple[int, str]:
    """Run a command, passing each output line to ``on_line`` (default: echo to stdout) and returning all of it."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    lines: list[str] = []
    try:
        for line in proc.stdout:
            if on_line is None:
                print(line, end="", flush=True)
            else:
                on_line(line)
            lines.append(line)
        proc.wait()
    except BaseException:
        proc.kill()
        raise
    return proc.returncode, "".join(lines)


def load_result(out: Path) -> Optional[dict]:
    """``<out>/result.json`` as written by get_passes.py, or None."""
    f = Path(out) / "result.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cached_result(source: Path, out: Path) -> Optional[dict]:
    """A saved Listing 4 result for exactly this source that is newer than the source file, else None.

    Decoding is greedy (temperature 0), so asking again for an unchanged file gives the same answer; reusing the
    saved prediction saves the minutes of model loading and inference."""
    r = load_result(out)
    if r is None or (r.get("frontend") or {}).get("mode") != "listing4":
        return None
    try:
        if Path(r.get("input", "")).resolve() != Path(source).resolve():
            return None
        if (Path(out) / "result.json").stat().st_mtime <= Path(source).stat().st_mtime:
            return None
    except OSError:
        return None
    return r


def prediction_for_experiment(result: dict, result_file: Path) -> tuple[Optional[dict], Optional[str]]:
    """(llm info for ``run_experiment(llm=...)``, None) or (None, why the model's answer cannot be used)."""
    pred = result.get("prediction") or {}
    val = result.get("validation") or {}
    outcome = pred.get("outcome")
    if outcome == "unparsed":
        return None, "the model's reply contains no `opt -p` pipeline"
    if outcome == "pipeline" and not val.get("valid"):
        return None, "LLVM 17 opt rejects the predicted pipeline: " + str(val.get("error", ""))[:400]
    if outcome not in ("pipeline", "no-change"):
        return None, f"unexpected prediction outcome {outcome!r}"
    srv = result.get("server") or {}
    try:
        generated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(Path(result_file).stat().st_mtime))
    except OSError:
        generated = ""
    return {
        "pipeline": pred.get("pipeline") if outcome == "pipeline" else None,
        "outcome": outcome,
        "kind": pred.get("kind"),
        "model": Path(str(srv.get("model_path", ""))).name,
        "llama_cpp_build": srv.get("build"),
        "prompt_tokens": result.get("prompt_tokens"),
        "inference_seconds": result.get("inference_seconds"),
        "finish_reason": result.get("finish_reason"),
        "predicted_object_size_before": pred.get("predicted_size_before"),
        "predicted_object_size_after": pred.get("predicted_size_after"),
        "measured_object_size_after": (result.get("summary") or {}).get("model_object_size"),
        "frontend": (result.get("frontend") or {}).get("mode"),
        "result_json": str(result_file),
        "predicted_at": generated,
    }, None


def run_c_workflow(source: Path, out: Path, *, server_url: str = DEFAULT_SERVER_URL, model: Path = DEFAULT_MODEL,
                   frontend: str = "listing4", check: bool = True, gpu_layers: int = DEFAULT_GPU_LAYERS,
                   context: int = DEFAULT_CONTEXT, server_timeout: int = DEFAULT_SERVER_TIMEOUT,
                   log_path: Optional[Path] = None, echo: Echo = _print,
                   on_line: Optional[Callable[[str], None]] = None) -> dict:
    """C file -> get_passes.py --evaluate [--check], starting and stopping a server if none is running.

    Returns {'returncode', 'output', 'started_server', 'result_written'}. ``result_written`` is True only if
    result.json was (re)written by this run; a failed run never leaves an older result looking current.
    """
    out = Path(out)
    log_path = Path(log_path) if log_path else out.parent / "server.log"
    if frontend not in FRONTENDS:
        raise ValueError(f"frontend must be one of {FRONTENDS}")
    srv = server_status(server_url)
    problems = llm_problems(server_url, model, srv)
    if problems:
        raise RuntimeError("the LLM step cannot run:\n  - " + "\n  - ".join(problems))
    result_file = out / "result.json"
    before = result_file.stat().st_mtime if result_file.is_file() else None
    started: Optional[subprocess.Popen] = None
    try:
        if srv["reachable"]:
            echo(f"connecting to the running server at {server_url} (it will not be stopped)")
        else:
            started = start_server(Path(model), server_url, gpu_layers, context, log_path, echo)
            echo(f"waiting for the model to load (up to {server_timeout} s) ...")
            wait_ready(started, server_url, log_path, server_timeout)
            echo("server ready")
        do_check = check and has_main_c(source)
        if check and not do_check:
            echo("differential test (--check) skipped: no main()")
        elif not check:
            echo("differential test (--check) skipped: execution is disabled")
        cmd = get_passes_command(source, out, server_url, frontend, do_check)
        rc, text = run_streaming(cmd, LLM, on_line)
    finally:
        if started is not None:
            stop_server(started, echo)
    after = result_file.stat().st_mtime if result_file.is_file() else None
    return {"returncode": rc, "output": text, "started_server": started is not None,
            "result_written": after is not None and after != before}
