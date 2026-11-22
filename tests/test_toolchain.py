"""Tests for the central toolchain definition (src/toolchain.py, toolchain.json).

Most tests use small fake tools that only answer --version, so they need no LLVM. The last class checks the real,
installed project toolchain and is skipped if it is not installed.

Run:  python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.toolchain import CONFIG_PATH, ROOT, ToolchainError, inspect, load_config, resolve, verify  # noqa: E402


def fake_tool(folder: Path, name: str, version_text: str) -> Path:
    """A command that prints `version_text` for --version (a .cmd on Windows, a shell script elsewhere)."""
    script = folder / f"{name}_ver.txt"
    script.write_text(version_text, encoding="utf-8")
    if os.name == "nt":
        tool = folder / f"{name}.cmd"
        tool.write_text(f'@echo off\r\ntype "{script}"\r\n')
    else:
        tool = folder / name
        tool.write_text(f'#!/bin/sh\ncat "{script}"\n')
        tool.chmod(0o755)
    return tool


CLANG17 = "clang version 17.0.6 (test)\nTarget: x86_64-w64-windows-gnu\nThread model: posix\n"
LLVM17 = "LLVM (http://llvm.org/):\n  LLVM version 17.0.6\n"


class FakeToolchainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def config(self, clang=CLANG17, opt=LLVM17, size=LLVM17, **override) -> Path:
        cfg = {"llvm_version": "17.0.6", "target_triple": "x86_64-w64-windows-gnu",
               "clang": str(fake_tool(self.dir, "clang", clang)),
               "opt": str(fake_tool(self.dir, "opt", opt)),
               "llvm_size": str(fake_tool(self.dir, "llvm-size", size))}
        cfg.update(override)
        path = self.dir / "toolchain.json"
        path.write_text(json.dumps(cfg))
        return path

    def test_consistent_toolchain_is_accepted(self):
        tc = verify(config_path=self.config())
        self.assertEqual({n: s.version for n, s in tc.status.items()},
                         {"clang": "17.0.6", "opt": "17.0.6", "llvm_size": "17.0.6"})

    def test_newer_clang_is_refused(self):
        cfg = self.config(clang="clang version 23.1.4 (x)\nTarget: x86_64-w64-windows-gnu\n")
        with self.assertRaises(ToolchainError) as cm:
            verify(config_path=cfg)
        self.assertIn("clang: version 23.1.4, but the project requires 17.0.6", str(cm.exception))

    def test_mixed_toolchain_refused_even_for_a_tool_the_step_does_not_need(self):
        # the experiment needs only clang, but a present opt of another version is still refused
        cfg = self.config(opt="LLVM (http://llvm.org/):\n  LLVM version 23.1.6\n")
        with self.assertRaises(ToolchainError) as cm:
            verify(need=("clang",), config_path=cfg)
        self.assertIn("opt: version 23.1.6", str(cm.exception))

    def test_absent_tool_only_matters_when_needed(self):
        cfg = self.config(opt=str(self.dir / "missing" / "opt.exe"))
        verify(need=("clang",), config_path=cfg)                  # experiment: fine without opt
        with self.assertRaises(ToolchainError):
            verify(need=("clang", "opt", "llvm_size"), config_path=cfg)   # LLM workflow: refused

    def test_wrong_target_is_refused(self):
        cfg = self.config(clang="clang version 17.0.6 (x)\nTarget: x86_64-pc-windows-msvc\n")
        with self.assertRaisesRegex(ToolchainError, "targets x86_64-pc-windows-msvc"):
            verify(config_path=cfg)

    def test_unreadable_version_is_refused(self):
        with self.assertRaisesRegex(ToolchainError, "could not read a version"):
            verify(config_path=self.config(clang="something else\n"))

    def test_incomplete_or_missing_config(self):
        bad = self.dir / "bad.json"
        bad.write_text(json.dumps({"llvm_version": "17.0.6"}))
        with self.assertRaisesRegex(ToolchainError, "lacks"):
            load_config(bad)
        with self.assertRaisesRegex(ToolchainError, "configuration missing"):
            load_config(self.dir / "nope.json")

    def test_relative_paths_resolve_against_the_config_folder(self):
        self.config()
        cfg = json.loads((self.dir / "toolchain.json").read_text())
        cfg["clang"] = Path(cfg["clang"]).name                    # relative
        path = self.dir / "toolchain.json"
        path.write_text(json.dumps(cfg))
        self.assertEqual(inspect(config_path=path).clang, self.dir / Path(cfg["clang"]).name)


@unittest.skipUnless(CONFIG_PATH.is_file() and resolve(load_config()["clang"]).is_file(),
                     "project toolchain not installed")
class ProjectToolchainTests(unittest.TestCase):
    def test_project_toolchain_is_llvm_17_0_6(self):
        tc = verify()
        self.assertEqual(tc.llvm_version, "17.0.6")
        for name, st in tc.status.items():
            self.assertEqual(st.version, "17.0.6", name)
            self.assertIsNone(st.problem, name)
        # project-local: every tool lives inside the project folder
        for p in (tc.clang, tc.opt, tc.llvm_size):
            self.assertTrue(Path(p).resolve().is_relative_to(ROOT), p)


if __name__ == "__main__":
    unittest.main()
