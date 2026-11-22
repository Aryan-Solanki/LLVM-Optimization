"""Run logs: every run writes a timestamped log file to ``logs/``.

    logs/20261122-171500_run_all.log        python run_all.py   (everything it prints, incl. main.py and get_passes.py)
    logs/20261122-171500_main.log           python main.py      (when run on its own)
    logs/20261122-171500_dashboard.log      one dashboard run   (the progress log shown while it runs)
    logs/20261122-171500_llama-server.log   llama-server output, when a run had to start the server
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"


def new_log_path(kind: str, stamp: Optional[str] = None) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{stamp or datetime.now().strftime('%Y%m%d-%H%M%S')}_{kind}.log"


class _Tee:
    def __init__(self, stream: TextIO, log: TextIO):
        self._stream, self._log = stream, log

    def write(self, data: str) -> int:
        self._log.write(data)
        self._log.flush()
        return self._stream.write(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):                   # isatty, encoding, reconfigure, ...
        return getattr(self._stream, name)


class tee_output:
    """``with tee_output("run_all") as path:`` copies everything printed to stdout/stderr into a new log file."""

    def __init__(self, kind: str):
        self.path = new_log_path(kind)

    def __enter__(self) -> Path:
        self._file = open(self.path, "w", encoding="utf-8", errors="replace")
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(self._out, self._file), _Tee(self._err, self._file)
        return self.path

    def __exit__(self, *exc) -> None:
        sys.stdout, sys.stderr = self._out, self._err
        self._file.close()


class RunLog:
    """A log file that is written line by line (the dashboard's progress messages)."""

    def __init__(self, kind: str):
        self.path = new_log_path(kind)
        self._file = open(self.path, "w", encoding="utf-8", errors="replace")

    def write(self, line: str) -> None:
        self._file.write(f"{datetime.now().strftime('%H:%M:%S')}  {line.rstrip()}\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
