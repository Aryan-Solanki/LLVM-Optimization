"""Plumbing tests for analyzer/compiler/visualizer using tests/fake_clang.py.

These verify file layout, CSV columns, the reduction formula, correctness
comparison logic and chart generation.  They do NOT test real Clang behaviour;
run `python main.py` (the project's LLVM 17 toolchain) for that.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.analyzer import (  # noqa: E402
    CSV_COLUMNS, FAIL, PASS, SKIPPED, TestCase, compare_runs, reduction_pct, run_executable, run_experiment,
)
from src.compiler import CompilationError, LLVMNotFoundError, find_clang  # noqa: E402
from src.visualizer import generate_all  # noqa: E402

HERE = Path(__file__).resolve().parent


def make_fake_clang(tmp: Path) -> str:
    if os.name == "nt":
        wrapper = tmp / "fakeclang.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{HERE / "fake_clang.py"}" %*\r\n')
    else:
        wrapper = tmp / "fakeclang"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE / "fake_clang.py"}" "$@"\n')
        wrapper.chmod(0o755)
    return str(wrapper)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.clang = make_fake_clang(self.tmp)
        self.src = self.tmp / "prog.c"
        self.src.write_text("int main(void){return 0;}\n")

    def tearDown(self):
        self._tmp.cleanup()

    def run_it(self, **kw):
        return run_experiment(
            self.src, output_root=self.tmp / "output", results_dir=self.tmp / "results",
            clang=self.clang, run_tests=False, repeats=1, **kw,
        )

    def test_files_columns_and_reductions(self):
        res = self.run_it()
        for lv in ("O0", "O1", "O2", "O3", "Os", "Oz"):
            self.assertTrue((self.tmp / "output" / lv / f"output_{lv}.ll").is_file())
            self.assertTrue((self.tmp / "output" / lv / f"output_{lv}.s").is_file())
        self.assertEqual(list(res.table.columns), CSV_COLUMNS)
        t = res.table.set_index("opt_level")
        self.assertEqual(t.loc["O0", "ir_instructions"], 12)
        self.assertEqual(t.loc["O2", "ir_instructions"], 6)
        self.assertAlmostEqual(t.loc["O2", "ir_reduction_pct"], 50.0)
        self.assertAlmostEqual(t.loc["O0", "ir_reduction_pct"], 0.0)
        self.assertEqual(list(t.index), ["O0", "O1", "O2", "O3", "Os", "Oz"])
        # Os is compiled with its own flag, not as an alias of Oz
        self.assertEqual(t.loc["Os", "ir_instructions"], 7)
        self.assertEqual(t.loc["Oz", "ir_instructions"], 5)
        self.assertIn("-Os", res.metadata["commands"]["Os"]["ir"])
        self.assertIn("-Os", res.metadata["commands"]["Os"]["asm"])
        self.assertIn("-Os", res.metadata["commands"]["Os"]["exe"])
        self.assertTrue((res.table["correctness"] == SKIPPED).all())
        self.assertTrue((self.tmp / "results" / "results.csv").is_file())
        self.assertTrue((self.tmp / "results" / "metadata.json").is_file())

    def test_baseline_always_included(self):
        res = self.run_it(levels=["O2"])
        self.assertEqual(list(res.table["opt_level"]), ["O0", "O2"])
        res = self.run_it(levels=["Os"])
        self.assertEqual(list(res.table["opt_level"]), ["O0", "Os"])

    def test_charts_written(self):
        res = self.run_it()
        paths = generate_all(res.table, self.tmp / "results")
        self.assertEqual(len(paths), 4)
        for p in paths.values():
            self.assertGreater(p.stat().st_size, 1000)

    def test_compile_error_reported(self):
        bad = self.tmp / "bad.txt"          # fake clang rejects non-.c input
        bad.write_text("x")
        with self.assertRaises(CompilationError):
            run_experiment(bad, output_root=self.tmp / "o", results_dir=self.tmp / "r",
                           clang=self.clang, run_tests=False, repeats=1)

    def test_wrong_clang_version_is_refused(self):
        """A clang that is not the required 17.0.6 must never be used (e.g. a global clang 23)."""
        from unittest import mock
        with mock.patch.dict(os.environ, {"FAKE_CLANG_VERSION": "23.1.4"}):
            with self.assertRaises(LLVMNotFoundError) as cm:
                self.run_it()
        self.assertIn("version 23.1.4, but the project requires 17.0.6", str(cm.exception))
        self.assertFalse((self.tmp / "results" / "results.csv").exists())   # nothing was produced

    def test_wrong_target_is_refused(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"FAKE_CLANG_TARGET": "x86_64-pc-windows-msvc"}):
            with self.assertRaises(LLVMNotFoundError) as cm:
                find_clang(self.clang)
        self.assertIn("targets x86_64-pc-windows-msvc", str(cm.exception))

    def test_missing_clang_has_no_fallback(self):
        """A missing clang is an error even if another clang is on PATH or in $CLANG: no silent fallback."""
        from unittest import mock
        with mock.patch.dict(os.environ, {"PATH": str(self.tmp) + os.pathsep + os.environ.get("PATH", ""),
                                          "CLANG": self.clang}):
            with self.assertRaises(LLVMNotFoundError) as cm:
                find_clang(str(self.tmp / "does-not-exist" / "clang.exe"))
        self.assertIn("missing", str(cm.exception))
        self.assertIn("llvm-mingw-20231128-ucrt-x86_64.zip", str(cm.exception))   # install help is included

    def test_metadata_records_toolchain_versions(self):
        res = self.run_it()
        tc = res.metadata["toolchain"]
        self.assertIn("clang version 17.0.6", tc["clang_version"])
        for key in ("toolchain_config", "llvm_version_required", "toolchain_opt_version", "toolchain_llvm_size_version"):
            self.assertIn(key, tc)


class MathAndComparison(unittest.TestCase):
    def test_reduction_formula(self):
        self.assertAlmostEqual(reduction_pct(200, 150), 25.0)
        self.assertAlmostEqual(reduction_pct(200, 250), -25.0)
        self.assertTrue(pd.isna(reduction_pct(0, 5)))

    def test_compare_runs(self):
        tests = [TestCase("a"), TestCase("b")]
        ok = {"status": "ok", "returncode": 0, "stdout": b"x", "stderr": b""}
        self.assertEqual(compare_runs([ok, ok], [ok, ok], tests)[0], PASS)
        diff = dict(ok, stdout=b"y")
        status, detail = compare_runs([ok, ok], [ok, diff], tests)
        self.assertEqual(status, FAIL)
        self.assertIn("b: stdout differs", detail)
        code = dict(ok, returncode=1)
        self.assertEqual(compare_runs([ok, ok], [code, ok], tests)[0], FAIL)
        self.assertEqual(compare_runs([ok, ok], [dict(ok, status="timeout"), ok], tests)[0], FAIL)

    def test_run_executable_passes_args_and_stdin(self):
        r = run_executable(Path(sys.executable), TestCase("t", ["-c", "import sys;print(sys.stdin.read().upper())"], "hi"))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["stdout"].strip(), b"HI")

    def test_run_executable_timeout(self):
        r = run_executable(Path(sys.executable), TestCase("t", ["-c", "import time;time.sleep(5)"]), timeout=1)
        self.assertEqual(r["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
