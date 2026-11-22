"""LLM Compiler as an experiment level: the shared runner and the "LLM" level (offline, no model, no server)."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import llm_runner as runner  # noqa: E402
from src.analyzer import PASS, run_experiment  # noqa: E402
from src.compiler import ALL_LEVELS, LLM_LEVEL  # noqa: E402
from src.toolchain import ToolchainError, verify  # noqa: E402

SAMPLE_RESULT = {
    "input": "", "prompt_tokens": 8088, "inference_seconds": 151.5, "finish_reason": "stop",
    "frontend": {"mode": "listing4"},
    "server": {"model_path": "m/llm-compiler-7b-ftd.Q4_K_M.gguf", "build": "b1"},
    "prediction": {"pipeline": "module(default<O1>),module(iroutliner),module(default<Oz>)",
                   "predicted_size_before": 1390, "predicted_size_after": 879, "outcome": "pipeline", "kind": "mixed"},
    "validation": {"valid": True, "error": ""},
    "summary": {"model_object_size": 878},
}


def _have_toolchain() -> bool:
    try:
        verify(need=("clang", "opt"))
        return True
    except ToolchainError:
        return False


class RunnerTests(unittest.TestCase):
    def test_get_passes_command_is_the_run_all_command(self):
        """Exactly what run_all.py ran before the runner was shared (the method must not change)."""
        cmd = runner.get_passes_command(Path("p.c"), Path("out"), "http://127.0.0.1:8080", "listing4", check=True)
        self.assertEqual(cmd, [sys.executable, "-B", "-I", str(runner.GET_PASSES), "p.c", "--evaluate",
                               "--server", "http://127.0.0.1:8080", "--out", "out", "--frontend", "listing4", "--check"])
        with_driver = runner.get_passes_command(Path("m.ll"), Path("o"), "u", None, check=True, driver=Path("d.c"))
        self.assertEqual(with_driver[-2:], ["--check", "d.c"])
        self.assertNotIn("--check", runner.get_passes_command(Path("p.c"), Path("o"), "u", "O0", check=False))

    def test_problems(self):
        self.assertTrue(any("not serving LLM Compiler" in p for p in runner.llm_problems(
            "http://127.0.0.1:8080", runner.DEFAULT_MODEL, {"reachable": True, "is_llm_compiler": False, "model_path": "qwen"})))
        self.assertTrue(any("non-local" in p for p in runner.llm_problems(
            "http://example.org:8080", runner.DEFAULT_MODEL, {"reachable": False})))
        self.assertTrue(any("model file missing" in p for p in runner.llm_problems(
            "http://127.0.0.1:8080", ROOT / "nope.gguf", {"reachable": False})))

    def test_unreachable_server_never_raises(self):
        self.assertEqual(runner.server_status("http://127.0.0.1:9"), {"reachable": False})

    def test_bad_frontend_is_refused(self):
        with self.assertRaises(ValueError):
            runner.run_c_workflow(ROOT / "input" / "example.c", ROOT / "workspace" / "_never", frontend="O2")


class PredictionTests(unittest.TestCase):
    def test_usable_prediction(self):
        info, err = runner.prediction_for_experiment(SAMPLE_RESULT, Path("result.json"))
        self.assertIsNone(err)
        self.assertEqual(info["pipeline"], SAMPLE_RESULT["prediction"]["pipeline"])
        self.assertEqual((info["predicted_object_size_before"], info["predicted_object_size_after"]), (1390, 879))

    def test_unusable_predictions(self):
        r = copy.deepcopy(SAMPLE_RESULT)
        r["prediction"]["outcome"] = "unparsed"
        self.assertIsNone(runner.prediction_for_experiment(r, Path("x"))[0])
        r = copy.deepcopy(SAMPLE_RESULT)
        r["validation"] = {"valid": False, "error": "unknown pass"}
        self.assertIn("rejects", runner.prediction_for_experiment(r, Path("x"))[1])

    def test_no_change_means_no_passes(self):
        r = copy.deepcopy(SAMPLE_RESULT)
        r["prediction"].update(outcome="no-change", pipeline=None)
        info, err = runner.prediction_for_experiment(r, Path("x"))
        self.assertIsNone(err)
        self.assertIsNone(info["pipeline"])

    def test_cached_result_only_for_unchanged_source(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            src = tmp / "p.c"
            src.write_text("int main(void){return 0;}\n")
            out = tmp / "llm" / "p"
            out.mkdir(parents=True)
            time.sleep(0.05)
            r = dict(SAMPLE_RESULT, input=str(src))
            (out / "result.json").write_text(json.dumps(r))
            self.assertIsNotNone(runner.cached_result(src, out))
            time.sleep(0.05)
            src.write_text("int main(void){return 1;}\n")         # edited after the prediction
            self.assertIsNone(runner.cached_result(src, out))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(_have_toolchain(), "project LLVM 17 toolchain not installed")
class LLMLevelTests(unittest.TestCase):
    """The LLM level with the real project toolchain (clang + opt 17.0.6); the pipeline is given, no model."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_llm_is_a_level(self):
        self.assertEqual(ALL_LEVELS[-1], LLM_LEVEL)

    def test_llm_level_is_measured_like_the_others(self):
        res = run_experiment(ROOT / "input" / "example.c", levels=["O0", "Oz", LLM_LEVEL],
                             output_root=self.tmp / "o", results_dir=self.tmp / "r", repeats=1,
                             llm={"pipeline": SAMPLE_RESULT["prediction"]["pipeline"]})
        t = res.table.set_index("opt_level")
        self.assertEqual(list(t.index), ["O0", "Oz", LLM_LEVEL])
        self.assertGreater(t.loc[LLM_LEVEL, "ir_instructions"], 0)
        self.assertGreater(t.loc[LLM_LEVEL, "asm_instructions"], 0)
        self.assertEqual(t.loc[LLM_LEVEL, "correctness"], PASS)
        self.assertIn("-p", res.metadata["commands"][LLM_LEVEL]["ir"])
        self.assertIn("results.csv", res.csv_path.name)

    def test_default_Oz_pipeline_matches_the_Oz_level(self):
        """Sanity check of the construction: with default<Oz> as 'prediction', LLM must equal clang -Oz."""
        res = run_experiment(ROOT / "input" / "example.c", levels=["O0", "Oz", LLM_LEVEL], run_tests=False,
                             output_root=self.tmp / "o", results_dir=self.tmp / "r", repeats=1,
                             llm={"pipeline": "default<Oz>"})
        t = res.table.set_index("opt_level")
        self.assertEqual(t.loc[LLM_LEVEL, "ir_instructions"], t.loc["Oz", "ir_instructions"])
        self.assertEqual(t.loc[LLM_LEVEL, "asm_instructions"], t.loc["Oz", "asm_instructions"])

    def test_invalid_pipeline_leaves_the_other_levels(self):
        res = run_experiment(ROOT / "input" / "example.c", levels=["O0", LLM_LEVEL], run_tests=False,
                             output_root=self.tmp / "o", results_dir=self.tmp / "r", repeats=1,
                             llm={"pipeline": "no-such-pass"})
        self.assertEqual(list(res.table["opt_level"]), ["O0"])
        self.assertIn("opt failed", res.metadata["llm"]["error"])

    def test_llm_level_needs_a_prediction(self):
        with self.assertRaises(ValueError):
            run_experiment(ROOT / "input" / "example.c", levels=["O0", LLM_LEVEL],
                           output_root=self.tmp / "o", results_dir=self.tmp / "r")


if __name__ == "__main__":
    unittest.main()
