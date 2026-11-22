"""LLM Compiler 7B FTD: predict an LLVM 17 optimization pipeline that reduces the OBJECT FILE SIZE of a whole LLVM
module, then optionally apply it with LLVM 17 and measure the real object size.

Follows the flag-tuning workflow of Cummins et al., "Meta Large Language Model Compiler: Foundation Models of
Compiler Optimization" (arXiv:2407.02524, 2024): Sec. 3.1, Fig. 3 (inference path), Listings 1, 4, 5 and 6.

  python get_passes.py prog.ll                              # predict a pipeline for the whole module
  python get_passes.py prog.ll --apply                      # + apply it with LLVM 17 opt, measure object size
  python get_passes.py prog.ll --evaluate                   # + compare with no optimization and default<O1..Oz>
  python get_passes.py prog.ll --evaluate --check drv.c     # + differential test: drv.c linked with every variant
  python get_passes.py prog.c [--frontend listing4|O0] [--target TRIPLE]

Needs llama-server serving the LLM Compiler GGUF without a grammar (run_all.py and the dashboard start one), and
the project's LLVM 17.0.6 toolchain from ..\\toolchain.json (clang, opt and llvm-size, all verified to be 17.0.6
before anything runs; see ..\\src\\toolchain.py). No tool is taken from PATH.
Every run writes its artifacts to <input-stem>.llmc\\ next to the input (or --out DIR).

Workflow                                                                                    Paper
 1 Input: one whole LLVM module (translation unit), sent exactly as given: module ID,      Sec. 3.1, Listing 4
   source filename, datalayout, triple, globals, every function, attributes, attribute
   groups, metadata, comments. No function extraction, no normalization, no rewriting.
   (Windows CRLF line endings are read as LF, the paper's Linux line ending; see read_module.)
   A .c file is first compiled by clang to unoptimized IR (FRONTEND_FLAGS).                 Sec. 3.1 "as emitted by
                                                                                            the clang frontend"
 2 Prompt (build_prompt): "[INST] Tell me how to optimize this LLVM-IR for object file      Listing 4
   size:\\n\\n<code>{module}</code> [/INST]". The only edit to the module text: trailing
   line breaks are removed, so </code> follows the last line as in Listing 4.
 3 Context: the prompt plus room for the reply must fit the server's context window.       Sec. 5.1 (modules over
   Otherwise the run stops with an error. Nothing is truncated or split.                    15k tokens were split
                                                                                            with llvm-extract; not
                                                                                            done here)
 4 Decoding: greedy (temperature 0). Generation stops after the pass list.                 Fig. 3 (2): "we generate
                                                                                            only the pass list"
 5 Parsing: "...size of N which can be reduced to M by running: `opt -p '<pipeline>'`", or  Listings 4 and 5
   "...size of N which cannot be reduced further". The pipeline string is kept verbatim.
 6 Validation: LLVM 17 opt parses the pipeline (-print-pipeline-passes). Invalid -> error. Sec. 2.2/3.1: opt 17.0.6,
   Nothing is substituted, translated or rewritten.                                         new pass manager
 7 Application, always to the ORIGINAL module: opt in.ll -S -o out.ll -p '<pipeline>'.     Listing 1
   IR written by the model is never used.
 8 Object size = text + data as reported by `size` (Berkeley format) for the object that   Sec. 2.2, Listings 1, 6
   `clang out.ll -c` (no -O flag) produces for the module's own target triple. Checked:
   the Listing 4 module measures 65 bytes, and 53 after default<Oz>, as in the paper.
 9 Baselines (--evaluate): no optimization and default<O1|O2|O3|Os|Oz>, same commands.     Sec. 5.1 (-Oz baseline)
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))       # the project root: the one toolchain definition is src/toolchain.py
from src.toolchain import ToolchainError, load_config, resolve as _resolve_tool, verify as _verify_toolchain  # noqa: E402

OPT17 = _resolve_tool(load_config()["opt"])  # LLVM 17.0.6 opt, as configured in ../toolchain.json
LLVM17 = OPT17.parent

# ================================================================ single source of truth: prompt and decoding
INSTRUCTION = "Tell me how to optimize this LLVM-IR for object file size:"
GENERATION = {"temperature": 0.0, "max_tokens": 200,
              # the reply continues with "This will produce code:\n\n<code>..." (the model's own IR): stop there
              "stop": ["This will produce code", "<code>", "</s>", "[INST]"]}


def build_prompt(module_text):
    """The exact prompt sent to the model (paper Listing 4). llama-server prepends the BOS token itself."""
    return f"[INST] {INSTRUCTION}\n\n<code>{module_text.rstrip(chr(13) + chr(10))}</code> [/INST]"


# ================================================================ unoptimized IR for a .c input
FRONTEND_FLAGS = {
    # Listing 4's input has `minsize optsize` function attributes and TBAA metadata, yet no optimization applied:
    # that is clang's output at -Oz with the IR optimization passes disabled.
    "listing4": ["-Oz", "-Xclang", "-disable-llvm-passes"],
    # Plain unoptimized IR (adds `noinline` to every function). -disable-O0-optnone: otherwise every function is
    # `optnone` and opt leaves it untouched.
    "O0": ["-O0", "-Xclang", "-disable-O0-optnone"],
}

# ================================================================ evaluation
PRESETS = ["O1", "O2", "O3", "Os", "Oz"]
PIPELINE_RE = re.compile(r"""opt\s+(?:-p|--?passes)(?:\s+|=)(?:'([^']*)'|"([^"]*)"|([^\s`'"]+))""")
REDUCE_RE = re.compile(r"object file size of (\d+) which can be reduced to (\d+)")
NOCHANGE_RE = re.compile(r"object file size of (\d+) which cannot be reduced further")
PRESET_RE = re.compile(r"(?:module\()?default<(O[0-3sz])>\)?")


class Failure(Exception):
    """A step of the workflow could not be done. The message says which and why; nothing is substituted."""


def sh(cmd, timeout=600):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)


def show(cmd):
    """Printable command line (single quotes around arguments with spaces or shell characters)."""
    return " ".join(f"'{c}'" if re.search(r"""[\s<>()|;&'"]""", c) else c for c in map(str, cmd))


def write_exact(path, text):
    """Write text byte-for-byte (no newline translation)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def first_lines(text, n=3):
    return " | ".join(text.strip().splitlines()[:n])


class Tools:
    """The project's LLVM toolchain (../toolchain.json): clang, opt and llvm-size, each verified to report the
    required version (17.0.6, the version LLM Compiler targets) before anything runs. Nothing comes from PATH, and
    there is no fallback to another compiler: a missing or different tool stops the run."""

    def __init__(self):
        try:
            tc = _verify_toolchain()
        except ToolchainError as e:
            raise Failure(str(e)) from None
        self.opt = tc.opt
        self.clang = str(tc.clang)
        self.size = str(tc.llvm_size)
        self.llvm_version = tc.llvm_version

    def need(self, what):
        if what == "clang" and not self.clang:
            raise Failure("clang missing from the project toolchain (../toolchain.json)")
        if what == "size" and not self.size:
            raise Failure("llvm-size missing from the project toolchain (../toolchain.json)")

    def versions(self):
        v = {"opt": re.search(r"LLVM version (\S+)", sh([self.opt, "--version"]).stdout).group(1)}
        v["clang"] = sh([self.clang, "--version"]).stdout.splitlines()[0]
        v["clang_path"] = self.clang
        v["host_triple"] = sh([self.clang, "-print-target-triple"]).stdout.strip()
        v["llvm_size"] = self.size
        v["llvm_size_version"] = (re.search(r"LLVM version (\S+)", sh([self.size, "--version"]).stdout) or [None, None])[1]
        v["toolchain"] = f"../toolchain.json (LLVM {self.llvm_version})"
        return v


# ================================================================ module
def read_module(path):
    """The module text as the model sees it. The file's bytes are decoded as UTF-8; Windows CRLF line endings
    (clang on Windows writes .ll files with CRLF) are decoded to LF, the line ending of the paper's Linux IR.
    Nothing else is changed. Returns (text, crlf_was_decoded)."""
    try:
        raw = Path(path).read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise Failure(f"{path} is not UTF-8 text: {e}") from None
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def defined_functions(ir):
    return [n.strip('"') for n in re.findall(r'^define\b[^\n]*?@("[^"]+"|[-\w.$]+)\s*\(', ir, re.M)]


def module_info(ir):
    get = lambda pat: (re.search(pat, ir, re.M) or [None, None])[1]  # noqa: E731
    return {"target_triple": get(r'^target triple = "([^"]*)"'), "datalayout": get(r'^target datalayout = "([^"]*)"'),
            "defined_functions": defined_functions(ir),
            "declared_functions": len(re.findall(r"^declare\b", ir, re.M)),
            "globals": len(re.findall(r"^@[^\s=]+\s*=", ir, re.M)),
            "attribute_groups": len(re.findall(r"^attributes #\d+", ir, re.M)),
            "metadata_lines": len(re.findall(r"^!", ir, re.M)),
            "has_main": "main" in defined_functions(ir),
            "optnone": bool(re.search(r"\boptnone\b", ir)), "bytes": len(ir.encode()),
            "sha256": hashlib.sha256(ir.encode()).hexdigest()}


def count_instr(ir):
    """Supplementary diagnostic only (never decides anything): non-label lines inside function bodies."""
    n, inside = 0, False
    for line in ir.splitlines():
        s = line.strip()
        if s.startswith("define"):
            inside = True
        elif inside:
            if s == "}":
                inside = False
            elif s and not s.startswith(";") and not re.match(r'^[\w.$"-]+:', s):
                n += 1
    return n


def compile_c(tools, src, out, frontend, target):
    tools.need("clang")
    cmd = [tools.clang, *FRONTEND_FLAGS[frontend], *([f"--target={target}"] if target else []),
           "-S", "-emit-llvm", src, "-o", out]
    r = sh(cmd)
    if r.returncode:
        raise Failure(f"clang could not compile {src}:\n{r.stderr.strip()}")
    return cmd


def llvm17_parses(tools, module):
    r = sh([tools.opt, "-disable-output", module])
    return r.returncode == 0, r.stderr.strip()


# ================================================================ model
def http_json(url, body=None, timeout=900):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def check_no_grammar_server():
    """A server-wide --grammar-file would constrain decoding."""
    r = sh(["powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\").CommandLine"])
    if r.returncode == 0 and "--grammar-file" in r.stdout:
        raise Failure("The running llama-server was started with a grammar (--grammar-file), which constrains decoding."
                      "\nStop it; python run_all.py starts one without a grammar.")


def server_info(server):
    try:
        props = http_json(server + "/props", timeout=30)
        meta = (http_json(server + "/v1/models", timeout=30).get("data") or [{}])[0].get("meta") or {}
    except OSError as e:
        raise Failure(f"server not reachable at {server} ({e}). From the project root, python run_all.py starts one.")
    info = {"model_path": props.get("model_path"), "build": props.get("build_info"),
            "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
            "n_ctx_train": meta.get("n_ctx_train"), "total_slots": props.get("total_slots")}
    if "llm-compiler" not in str(info["model_path"]).lower():
        raise Failure(f"the server is not serving LLM Compiler (model: {info['model_path']}). "
                      "From the project root, python run_all.py starts the right one.")
    return info


def check_context(server, prompt, info):
    n = len(http_json(server + "/tokenize", {"content": prompt, "add_special": True}, timeout=120)["tokens"])
    need = n + GENERATION["max_tokens"]
    if not info["n_ctx"] or need > info["n_ctx"]:
        raise Failure(f"the module does not fit the model's context: prompt {n} tokens + {GENERATION['max_tokens']} "
                      f"for the reply = {need} > {info['n_ctx']} (server context; the model was trained with "
                      f"{info['n_ctx_train']}).\nNothing was sent. The module is not truncated and no function is "
                      "extracted. Options: restart the server with a larger -Context (see README), or use a smaller "
                      "translation unit.")
    return n


def ask(server, prompt):
    body = {"prompt": prompt, **GENERATION}
    t0 = time.perf_counter()
    try:
        resp = http_json(server + "/v1/completions", body)
    except OSError as e:
        raise Failure(f"request to {server} failed: {e}") from None
    return body, resp, round(time.perf_counter() - t0, 2)


def parse_reply(text):
    """The pipeline exactly as the model wrote it (no stripping, no rewriting), plus the model's size claims."""
    m, red, same = PIPELINE_RE.search(text), REDUCE_RE.search(text), NOCHANGE_RE.search(text)
    out = {"pipeline": next(g for g in m.groups() if g is not None) if m else None,
           "predicted_size_before": int((red or same).group(1)) if (red or same) else None,
           "predicted_size_after": int(red.group(2)) if red else (int(same.group(1)) if same else None)}
    out["outcome"] = "pipeline" if m else ("no-change" if same else "unparsed")
    return out


def classify(pipeline):
    """'preset' = only a built-in pipeline, 'mixed' = built-in pipeline plus other passes, 'custom' = no preset."""
    if pipeline is None:
        return None
    if PRESET_RE.fullmatch(pipeline):
        return "preset"
    return "mixed" if re.search(r"default<O[0-3sz]>", pipeline) else "custom"


# ================================================================ LLVM 17: validate, apply, measure, test
def validate(tools, pipeline, module):
    cmd = [tools.opt, "-p", pipeline, "-print-pipeline-passes", "-disable-output", module]
    r = sh(cmd)
    return {"valid": r.returncode == 0, "expanded": r.stdout.strip(), "error": r.stderr.strip(),
            "command": show(cmd)}


def object_size(tools, ll, obj, triple):
    """Paper: `clang output.bc -c`, then `size output.o`, text + data. The module's own triple is passed because
    clang otherwise compiles for the host and only warns."""
    tools.need("clang")
    tools.need("size")
    cmd = [tools.clang, *([f"--target={triple}"] if triple else []), "-c", "-x", "ir", ll, "-o", obj]
    r = sh(cmd)
    if r.returncode:
        return {"object_ok": False, "object_error": first_lines(r.stderr), "compile_command": show(cmd)}
    berkeley = sh([tools.size, obj]).stdout.strip().splitlines()
    text, data = (int(x) for x in berkeley[-1].split()[:2])
    sections = {}
    for line in sh([tools.size, "-A", obj]).stdout.splitlines():
        p = line.split()
        if len(p) == 3 and p[0].startswith(".") and p[1].isdigit():
            sections[p[0]] = sections.get(p[0], 0) + int(p[1])
    pick = lambda base: sum(v for k, v in sections.items() if k == base or k.startswith((base + ".", base + "$")))  # noqa: E731
    return {"object_ok": True, "object_size": text + data, "size_text": text, "size_data": data,
            "section_text": pick(".text"), "section_data": pick(".data"), "sections": sections,
            "compile_command": show(cmd), "size_command": show([tools.size, obj])}


def run_variant(tools, name, pipeline, module, outdir, triple):
    """opt <module> -S -o <name>.ll [-p <pipeline>], then the object size. pipeline None = no passes at all."""
    ll, obj = outdir / f"{name}.ll", outdir / f"{name}.o"
    cmd = [tools.opt, module, "-S", "-o", ll, *(["-p", pipeline] if pipeline is not None else [])]
    r = sh(cmd)
    v = {"variant": name, "pipeline": pipeline, "opt_argv": [str(c) for c in cmd], "opt_command": show(cmd),
         "opt_ok": r.returncode == 0, "ll": str(ll)}
    if r.returncode:
        v["opt_error"] = r.stderr.strip()[:3000]
        return v
    v["ir_instructions"] = count_instr(ll.read_text())  # supplementary diagnostic
    v.update(object_size(tools, ll, obj, triple))
    return v


def build_and_run(tools, ll, driver, exe, triple):
    cmd = [tools.clang, *([f"--target={triple}"] if triple else []), *([driver] if driver else []),
           "-x", "ir", ll, "-o", exe]
    r = sh(cmd)
    if r.returncode:
        return None, "build failed: " + first_lines(r.stderr, 1)
    try:
        p = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None, "timed out after 30 s"
    return (p.returncode, p.stdout), ""


def differential_test(tools, variants, driver, outdir, triple):
    """Run every variant (driver + variant, or the module's own main) and compare exit code and stdout with the
    unoptimized module. Passing means: identical on this driver's inputs, not a proof of equivalence."""
    host = tools.versions().get("host_triple")
    if triple and host and triple != host:
        for v in variants:
            v["correctness"] = f"not checked (module targets {triple}; this host runs {host})"
        return
    ref, err = build_and_run(tools, outdir / "none.ll", driver, outdir / "none.exe", triple)
    for v in variants:
        if not v.get("opt_ok"):
            v["correctness"] = "not checked (no optimized IR)"
        elif ref is None:
            v["correctness"] = f"not checked (reference: {err})"
        elif v["variant"] == "none":
            v["correctness"] = "reference"
        else:
            got, err2 = build_and_run(tools, v["ll"], driver, outdir / f"{v['variant']}.exe", triple)
            v["correctness"] = (f"not checked ({err2})" if got is None else
                                "passed (same exit code and output)" if got == ref else "MISMATCH")


# ================================================================ CLI
def pct(base, x):
    return round((base - x) / base * 100, 2) if base else None


def print_table(variants):
    print(f"\n{'variant':10} {'object size':>11} {'(text+data)':>11} {'.text':>6} {'.data':>6} {'IR instr':>8} "
          f"{'vs none':>8}  correctness")
    none = next((v.get("object_size") for v in variants if v["variant"] == "none"), None)
    for v in variants:
        if not v.get("opt_ok") or not v.get("object_ok"):
            print(f"{v['variant']:10} FAILED: {v.get('opt_error') or v.get('object_error')}"[:160])
            continue
        print(f"{v['variant']:10} {v['object_size']:>11} {str(v['size_text']) + '+' + str(v['size_data']):>11} "
              f"{v['section_text']:>6} {v['section_data']:>6} {v['ir_instructions']:>8} "
              f"{str(pct(none, v['object_size'])) + '%':>8}  {v.get('correctness', '-')}")


def run(a):
    tools = Tools()
    src = Path(a.input).resolve()
    outdir = Path(a.out).resolve() if a.out else src.with_name(src.stem + ".llmc")
    outdir.mkdir(parents=True, exist_ok=True)
    result = {"input": str(src), "outdir": str(outdir), "paper": "arXiv:2407.02524 (LLM Compiler, 2024)",
              "generation": GENERATION, "instruction": INSTRUCTION, "tools": tools.versions()}

    # 1. the module
    if src.suffix == ".c":
        module = outdir / (src.stem + ".ll")
        cmd = compile_c(tools, src, module, a.frontend, a.target)
        result["frontend"] = {"mode": a.frontend, "command": show(cmd)}
        print(f"(C compiled to unoptimized IR: {show(cmd)})", file=sys.stderr)
    elif src.suffix == ".ll":
        module = src
    else:
        raise Failure("input must be LLVM IR (.ll) or C (.c)")
    text, crlf = read_module(module)
    info = module_info(text)
    info["crlf_decoded_to_lf"] = crlf
    result.update(module=str(module), module_info=info)
    print(f"Module: {module}\n  functions defined: {', '.join(info['defined_functions']) or '(none)'}\n"
          f"  target triple: {info['target_triple']}  ({info['bytes']} bytes, sent whole and unchanged"
          + (", CRLF line endings read as LF)" if crlf else ")"))
    parses, err = llvm17_parses(tools, module)
    result["llvm17_parses_input"] = parses
    if not parses:
        msg = (f"LLVM 17 cannot read this module, so no predicted pipeline could be applied to it:\n{err}\n"
               "It was probably produced by a newer clang than the project's 17.0.6. Pass the .c file instead (it is "
               "compiled with the project toolchain), or IR produced by clang 17. The module is never edited.")
        if a.apply or a.evaluate:
            raise Failure(msg)
        print("warning: " + msg, file=sys.stderr)
    if info["optnone"]:
        print("warning: the module contains optnone functions; opt leaves them unchanged.", file=sys.stderr)

    # 2-4. prompt, context, model
    check_no_grammar_server()
    srv = server_info(a.server)
    result["server"] = srv
    prompt = build_prompt(text)
    write_exact(outdir / "prompt.txt", prompt)
    result["prompt_tokens"] = check_context(a.server, prompt, srv)
    print(f"  prompt: {result['prompt_tokens']} tokens (server context {srv['n_ctx']}, trained {srv['n_ctx_train']})")
    body, resp, secs = ask(a.server, prompt)
    reply = (resp.get("choices") or [{}])[0].get("text", "")
    (outdir / "request.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
    (outdir / "response.json").write_text(json.dumps(resp, indent=1), encoding="utf-8")
    write_exact(outdir / "reply.txt", reply)
    result["inference_seconds"] = secs
    result["finish_reason"] = (resp.get("choices") or [{}])[0].get("finish_reason")

    # 5-6. parse, classify, validate
    pred = parse_reply(reply)
    pred["kind"] = classify(pred["pipeline"])
    result["prediction"] = pred
    print("\nModel reply (raw, generation stopped after the pass list):\n  " + reply.strip().replace("\n", "\n  "))
    ok = True
    if pred["outcome"] == "pipeline":
        print(f"\nPredicted pipeline (model output, verbatim):\n{pred['pipeline']}")
        print(f"Type: {pred['kind']}.  Model's size claim: {pred['predicted_size_before']} -> "
              f"{pred['predicted_size_after']} (a prediction, not a measurement)")
        val = validate(tools, pred["pipeline"], module)
        result["validation"] = val
        if val["valid"]:
            print("LLVM 17 accepts the pipeline.")
            if a.expand:
                print("Expanded:\n" + val["expanded"])
        else:
            ok = False
            print(f"ERROR: LLVM 17 rejects the predicted pipeline (it is not modified or replaced):\n{val['error']}")
        print(f"\nTo apply it with LLVM 17 opt:\n  {show([tools.opt, module, '-S', '-o', 'out.ll', '-p', pred['pipeline']])}")
    elif pred["outcome"] == "no-change":
        print(f"\nModel: the code cannot be reduced further (size {pred['predicted_size_before']}). No pipeline.")
    else:
        ok = False
        print("\nERROR: the reply contains no `opt -p` pipeline and no 'cannot be reduced further' answer.")

    # 7-9. apply, measure, baselines, correctness
    if a.apply or a.evaluate:
        vdir = outdir / "variants"
        vdir.mkdir(exist_ok=True)
        triple = info["target_triple"]
        plan = [("none", None)] + ([(p, f"default<{p}>") for p in PRESETS] if a.evaluate else [])
        if pred["outcome"] == "pipeline" and ok:
            plan.append(("model", pred["pipeline"]))
        variants = [run_variant(tools, n, p, module, vdir, triple) for n, p in plan]
        if pred["outcome"] == "no-change":  # the model's answer is "apply nothing": that is the 'none' variant
            variants.append({**variants[0], "variant": "model", "note": "model: cannot be reduced further"})
        if a.check is not None:
            driver = Path(a.check).resolve() if a.check else None
            if driver is None and not info["has_main"]:
                raise Failure("--check without a driver needs a module that defines main")
            differential_test(tools, variants, driver, vdir, triple)
        result["variants"] = variants
        print_table(variants)
        by = {v["variant"]: v for v in variants}
        model = by.get("model")
        if model and model.get("object_ok"):
            s = {"model_object_size": model["object_size"], "unoptimized_object_size": by["none"].get("object_size"),
                 "improvement_over_unoptimized_pct": pct(by["none"].get("object_size"), model["object_size"])}
            if "Oz" in by and by["Oz"].get("object_ok"):
                s["Oz_object_size"] = by["Oz"]["object_size"]
                s["improvement_over_Oz_pct"] = pct(by["Oz"]["object_size"], model["object_size"])
            s["predicted_size_before"], s["predicted_size_after"] = (pred["predicted_size_before"],
                                                                     pred["predicted_size_after"])
            result["summary"] = s
            print(f"\nModel pipeline: measured {s['model_object_size']} bytes (model predicted "
                  f"{pred['predicted_size_after']}); unoptimized {s['unoptimized_object_size']} (model predicted "
                  f"{pred['predicted_size_before']}); improvement over unoptimized "
                  f"{s['improvement_over_unoptimized_pct']}%"
                  + (f", over -Oz {s['improvement_over_Oz_pct']}%" if "improvement_over_Oz_pct" in s else ""))
        else:
            ok = False
    (outdir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nArtifacts: {outdir}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="LLVM IR module (.ll), sent whole and unchanged, or a C file (.c)")
    ap.add_argument("--apply", action="store_true", help="apply the prediction with LLVM 17 and measure object size")
    ap.add_argument("--evaluate", action="store_true", help="--apply plus baselines: none, default<O1|O2|O3|Os|Oz>")
    ap.add_argument("--check", nargs="?", const="", metavar="DRIVER.c",
                    help="differential test of every variant: link DRIVER.c (or use the module's own main) and "
                         "compare exit code and stdout with the unoptimized module")
    ap.add_argument("--expand", action="store_true", help="print the pass-by-pass expansion of the pipeline")
    ap.add_argument("--out", help="artifact directory (default: <input-stem>.llmc next to the input)")
    ap.add_argument("--frontend", choices=list(FRONTEND_FLAGS), default="listing4",
                    help="C input only: clang flags for unoptimized IR (default: listing4 = -Oz -Xclang "
                         "-disable-llvm-passes, the form of the paper's Listing 4 input)")
    ap.add_argument("--target", help="C input only: target triple for clang (default: clang's host target)")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--function", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.function:
        ap.error("--function was removed: the 2024 workflow predicts for the whole module, never for one function")
    if a.check is not None and not (a.apply or a.evaluate):
        ap.error("--check needs --apply or --evaluate")
    try:
        return run(a)
    except Failure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
