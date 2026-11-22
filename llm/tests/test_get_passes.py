"""Tests for the 2024 (arXiv:2407.02524) workflow in get_passes.py.

Offline: no model needed. A local stub server stands in for llama-server, records the exact request and returns a
fixed reply; LLVM 17 opt, clang and llvm-size are real (the project toolchain, ../toolchain.json). Run from llm/:
    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import get_passes as gp  # noqa: E402

INPUTS = HERE / "inputs"
LISTING4 = INPUTS / "listing4_add_two.ll"           # the paper's Listing 4 module, verbatim
MULTI = INPUTS / "multi_function.O0.ll"             # clang -O0 output: 3 functions, globals, attributes, metadata
DRIVER = INPUTS / "multi_function_driver.c"

# The paper's Listing 4 prompt, written out literally (not built with get_passes)
LISTING4_PROMPT = (
    "[INST] Tell me how to optimize this LLVM-IR for object file size:\n\n"
    "<code>; ModuleID = '<stdin>'\n"
    'source_filename = "-"\n'
    'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"\n'
    'target triple = "x86_64-unknown-linux-gnu"\n'
    "\n"
    "; Function Attrs: minsize nounwind optsize uwtable\n"
    "define dso_local i32 @add_two(i32 noundef %0, i32 noundef %1) #0 {\n"
    "  %3 = alloca i32, align 4\n"
    "  %4 = alloca i32, align 4\n"
    "  store i32 %0, ptr %3, align 4, !tbaa !5\n"
    "  store i32 %1, ptr %4, align 4, !tbaa !5\n"
    "  %5 = load i32, ptr %3, align 4, !tbaa !5\n"
    "  %6 = load i32, ptr %4, align 4, !tbaa !5\n"
    "  %7 = add nsw i32 %5, %6\n"
    "  ret i32 %7\n"
    "}\n"
    "\n"
    'attributes #0 = { minsize nounwind optsize uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" '
    '"stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,'
    '+x87" "tune-cpu"="generic" }\n'
    "\n"
    "!llvm.module.flags = !{!0, !1, !2, !3}\n"
    "!llvm.ident = !{!4}\n"
    "\n"
    '!0 = !{i32 1, !"wchar_size", i32 4}\n'
    '!1 = !{i32 8, !"PIC Level", i32 2}\n'
    '!2 = !{i32 7, !"PIE Level", i32 2}\n'
    '!3 = !{i32 7, !"uwtable", i32 2}\n'
    '!4 = !{!"clang version 17.0.6 (git@github.com:fairinternal/CodeGen.git b05db9bbf7a92019267416c1bb9996fe6134e3f1)"}\n'
    "!5 = !{!6, !6, i64 0}\n"
    '!6 = !{!"int", !7, i64 0}\n'
    '!7 = !{!"omnipotent char", !8, i64 0}\n'
    '!8 = !{!"Simple C/C++ TBAA"}</code> [/INST]')


# ------------------------------------------------------------------ stub llama-server
class Stub:
    def __init__(self, reply, n_ctx=8192):
        self.reply, self.n_ctx, self.requests = reply, n_ctx, []
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/props":
                    self._send({"model_path": "models/llm-compiler-7b-ftd.Q4_K_M.gguf", "build_info": "stub",
                                "total_slots": 1, "default_generation_settings": {"n_ctx": stub.n_ctx}})
                else:
                    self._send({"data": [{"meta": {"n_ctx_train": 16384}}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append((self.path, body))
                if self.path == "/tokenize":
                    self._send({"tokens": list(range(len(body["content"]) // 4))})
                else:
                    self._send({"choices": [{"text": stub.reply, "finish_reason": "stop"}], "usage": {}})

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def completions(self):
        return [b for p, b in self.requests if p == "/v1/completions"]

    def close(self):
        self.httpd.shutdown()


def run_cli(args, reply, n_ctx=8192):
    """Run get_passes.main against the stub; record every external command get_passes executes."""
    stub, calls = Stub(reply, n_ctx), []
    real_sh, real_check = gp.sh, gp.check_no_grammar_server

    def recording_sh(cmd, timeout=600):
        calls.append([str(c) for c in cmd])
        return real_sh(cmd, timeout)
    gp.sh, gp.check_no_grammar_server = recording_sh, lambda: None
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = gp.main([*args, "--server", stub.url])
    finally:
        gp.sh, gp.check_no_grammar_server = real_sh, real_check
        stub.close()
    return code, out.getvalue(), err.getvalue(), stub, calls


def reply_for(pipeline, before=65, after=53):
    return (f"The code has object file size of {before} which can be reduced to {after} by running:\n\n"
            f"`opt -p '{pipeline}'`\n\n")


# ------------------------------------------------------------------ tests
class PromptTests(unittest.TestCase):
    def test_listing4_prompt_is_exactly_the_papers(self):
        self.assertEqual(gp.build_prompt(LISTING4.read_text()), LISTING4_PROMPT)

    def test_whole_module_reaches_prompt_unchanged(self):
        raw = MULTI.read_bytes().decode()
        text, crlf = gp.read_module(MULTI)
        self.assertEqual(text, raw.replace("\r\n", "\n"))  # only line endings decoded (clang on Windows writes CRLF)
        prompt = gp.build_prompt(text)
        self.assertIn(text.rstrip("\n"), prompt)  # the complete module, every character
        self.assertNotIn("\r", prompt)
        for must in ("; ModuleID = 'multi_function.c'", 'source_filename = "multi_function.c"', "target datalayout",
                     'target triple = "x86_64-w64-windows-gnu"', "@call_count = dso_local global",
                     "@weights = internal constant", "define dso_local i32 @weighted_sum",
                     "define internal i32 @clamp", "define dso_local i32 @max_of", "attributes #0 = { noinline",
                     "; Function Attrs:", "!llvm.module.flags", "!llvm.ident", "!llvm.loop !7", "    "):
            self.assertIn(must, prompt, must)

    def test_no_2023_preprocessing_left(self):
        for name in ("normalize_ir", "extract_function", "EXTRACT17", "_strip_comment", "_DROP_LINE"):
            self.assertFalse(hasattr(gp, name), name)
        self.assertNotIn("instruction count", gp.INSTRUCTION)


class ParseTests(unittest.TestCase):
    def test_preset(self):
        p = gp.parse_reply(reply_for("module(default<Oz>)"))
        self.assertEqual((p["outcome"], p["pipeline"], p["predicted_size_before"], p["predicted_size_after"]),
                         ("pipeline", "module(default<Oz>)", 65, 53))
        self.assertEqual(gp.classify(p["pipeline"]), "preset")
        self.assertEqual(gp.classify("default<O1>"), "preset")

    def test_custom_and_mixed_verbatim(self):
        custom = "module(constmerge),function(instcombine,simplifycfg),cgscc(inline),function(loop(licm))"
        self.assertEqual(gp.parse_reply(reply_for(custom))["pipeline"], custom)
        self.assertEqual(gp.classify(custom), "custom")
        mixed = "module(default<Oz>),module(iroutliner)"  # the paper's Listing 1 pipeline
        self.assertEqual(gp.parse_reply(reply_for(mixed))["pipeline"], mixed)
        self.assertEqual(gp.classify(mixed), "mixed")

    def test_other_spellings_and_whitespace_kept(self):
        self.assertEqual(gp.parse_reply('opt -passes="function(dce)"')["pipeline"], "function(dce)")
        self.assertEqual(gp.parse_reply("`opt -p function(dce)`")["pipeline"], "function(dce)")
        self.assertEqual(gp.parse_reply("opt -p ' function(dce)'")["pipeline"], " function(dce)")  # not stripped

    def test_no_change_and_unparsed(self):
        p = gp.parse_reply("The code has object file size of 53 which cannot be reduced further")
        self.assertEqual((p["outcome"], p["pipeline"], p["predicted_size_before"]), ("no-change", None, 53))
        self.assertEqual(gp.parse_reply("something else")["outcome"], "unparsed")


class LLVMTests(unittest.TestCase):
    def setUp(self):
        self.tools = gp.Tools()
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_object_size_reproduces_paper_listing4(self):
        none = gp.run_variant(self.tools, "none", None, LISTING4, self.tmp, "x86_64-unknown-linux-gnu")
        oz = gp.run_variant(self.tools, "Oz", "module(default<Oz>)", LISTING4, self.tmp, "x86_64-unknown-linux-gnu")
        self.assertEqual((none["object_size"], oz["object_size"]), (65, 53))   # Listing 4: "65 ... reduced to 53"
        self.assertEqual((none["section_text"], oz["section_text"]), (17, 5))  # .text alone (supplementary)
        self.assertIn("--target=x86_64-unknown-linux-gnu", none["compile_command"])

    def test_exact_pipeline_reaches_opt(self):
        pipe = "function(sroa,instcombine)"
        v = gp.run_variant(self.tools, "model", pipe, LISTING4, self.tmp, "x86_64-unknown-linux-gnu")
        self.assertTrue(v["opt_ok"], v.get("opt_error"))
        self.assertEqual(v["opt_argv"][-2:], ["-p", pipe])
        self.assertEqual(Path(v["opt_argv"][0]).resolve(), gp.OPT17.resolve())

    def test_invalid_pipeline_is_reported(self):
        val = gp.validate(self.tools, "function(no-such-pass)", LISTING4)
        self.assertFalse(val["valid"])
        self.assertIn("no-such-pass", val["error"])

    def test_tools_are_the_project_llvm17_toolchain(self):
        """clang, opt and llvm-size all come from ../toolchain.json and are 17.0.6; nothing from PATH."""
        from src.toolchain import verify
        tc = verify()
        self.assertEqual((Path(self.tools.clang), Path(self.tools.opt), Path(self.tools.size)),
                         (tc.clang, tc.opt, tc.llvm_size))
        v = self.tools.versions()
        self.assertEqual(v["opt"], "17.0.6")
        self.assertIn("clang version 17.0.6", v["clang"])
        self.assertEqual(v["llvm_size_version"], "17.0.6")

    def test_clang_option_removed(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            gp.main([str(LISTING4), "--clang", "clang"])


class EndToEndStubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_evaluate_multi_function_module(self):
        pipe = "module(default<Oz>),function(instcombine)"
        reply = reply_for(pipe, 400, 120)
        code, out, err, stub, calls = run_cli(
            [str(MULTI), "--evaluate", "--check", str(DRIVER), "--out", str(self.tmp)], reply)
        self.assertEqual(code, 0, out + err)
        # 1-3: exactly one completion request, whose prompt is the whole, unchanged module
        (body,) = stub.completions()
        self.assertEqual(body["prompt"], gp.build_prompt(MULTI.read_bytes().decode().replace("\r\n", "\n")))
        self.assertEqual(body["temperature"], 0.0)
        # 4: raw reply preserved
        self.assertEqual((self.tmp / "reply.txt").read_bytes().decode(), reply)
        self.assertEqual((self.tmp / "prompt.txt").read_bytes().decode(), body["prompt"])
        # no llvm-extract, no 2023 step anywhere
        self.assertFalse([c for c in calls if "llvm-extract" in Path(c[0]).name], calls)
        res = json.loads((self.tmp / "result.json").read_text())
        variants = {v["variant"]: v for v in res["variants"]}
        self.assertEqual(list(variants), ["none", "O1", "O2", "O3", "Os", "Oz", "model"])
        # 7: the exact predicted pipeline reached LLVM 17
        self.assertEqual(variants["model"]["opt_argv"][-2:], ["-p", pipe])
        self.assertEqual(res["prediction"]["kind"], "mixed")
        # 10: identical conditions: same opt invocation except the -p argument and output file
        base = variants["none"]["opt_argv"]
        for name, v in variants.items():
            self.assertEqual(v["opt_argv"][:4], base[:4], name)          # opt, module, -S, -o
            self.assertEqual(v["opt_argv"][5:], [] if name == "none" else ["-p", v["pipeline"]], name)
            self.assertEqual(v["compile_command"], variants["none"]["compile_command"].replace("none.", name + "."))
            self.assertTrue(v["object_ok"], name)
            self.assertEqual(v["object_size"], v["size_text"] + v["size_data"])
            self.assertTrue(v["correctness"] in ("reference", "passed (same exit code and output)"), v["correctness"])
        self.assertEqual(res["summary"]["Oz_object_size"], variants["Oz"]["object_size"])
        self.assertEqual(res["summary"]["predicted_size_after"], 120)

    def test_invalid_prediction_fails_without_substitution(self):
        code, out, err, stub, calls = run_cli([str(LISTING4), "--evaluate", "--out", str(self.tmp)],
                                              reply_for("function(no-such-pass)"))
        self.assertEqual(code, 1)
        self.assertIn("LLVM 17 rejects the predicted pipeline", out)
        res = json.loads((self.tmp / "result.json").read_text())
        self.assertNotIn("model", [v["variant"] for v in res["variants"]])  # nothing applied in its place

    def test_module_too_large_is_rejected_before_inference(self):
        code, out, err, stub, calls = run_cli([str(MULTI), "--out", str(self.tmp)], reply_for("default<Oz>"),
                                              n_ctx=1000)
        self.assertEqual(code, 2)
        self.assertIn("does not fit the model's context", err)
        self.assertEqual(stub.completions(), [])

    def test_no_change_answer(self):
        code, out, err, stub, calls = run_cli(
            [str(LISTING4), "--apply", "--out", str(self.tmp)],
            "The code has object file size of 65 which cannot be reduced further")
        self.assertEqual(code, 0, out + err)
        res = json.loads((self.tmp / "result.json").read_text())
        self.assertEqual(res["prediction"]["outcome"], "no-change")
        self.assertEqual(res["summary"]["model_object_size"], 65)

    def test_function_option_removed(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            gp.main([str(MULTI), "--function", "max_of"])


if __name__ == "__main__":
    unittest.main()
