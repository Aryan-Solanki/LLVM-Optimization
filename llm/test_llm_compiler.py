"""Smoke test for Meta LLM Compiler 7B FTD served by llama-server (stdlib only).

Sends the paper's own flag-tuning example (arXiv:2407.02524, Listing 4: a whole module) with the prompt from
get_passes.build_prompt, the single definition of the prompt, and prints the complete reply, without stopping
after the pass list. The paper's answer for this input: "object file size of 65 which can be reduced to 53".
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import get_passes as gp  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
MODULE, _ = gp.read_module(HERE / "tests" / "inputs" / "listing4_add_two.ll")

payload = {"model": "llm-compiler-7b-ftd", "prompt": gp.build_prompt(MODULE), "temperature": 0.0, "max_tokens": 768}
req = urllib.request.Request(f"{BASE}/v1/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})

t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as resp:
    body = json.load(resp)
elapsed = time.time() - t0

assert body.get("object") == "text_completion", f"unexpected object: {body.get('object')}"
text = body["choices"][0]["text"]
usage = body.get("usage", {})
timings = body.get("timings", {})

print("OK - valid text_completion JSON")
print(f"model:   {body.get('model')}")
print(f"finish:  {body['choices'][0].get('finish_reason')}")
print(f"tokens:  prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
if timings:
    print(f"speed:   {timings.get('predicted_per_second', 0):.1f} tok/s generation")
print(f"latency: {elapsed:.1f}s")
print("-" * 60)
print(text.strip())
print("-" * 60)
print("parsed:", gp.parse_reply(text))
