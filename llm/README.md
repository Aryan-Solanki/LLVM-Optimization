# llm/: LLM Compiler pass prediction

This folder holds the part of the project that asks **Meta's LLM Compiler 7B FTD** for an optimization pass
pipeline. The project uses that pipeline as the **LLM** level next to O0 … Oz (see the main README). You normally
don't run anything here directly: `python run_all.py` and the dashboard do it for you.

```
llm/
├── get_passes.py           the method: prompt, greedy decoding, LLVM 17 validation, object-size measurement
├── templates/llm-compiler.jinja   Meta's [INST] … [/INST] chat template for llama-server
├── test_llm_compiler.py    live smoke test: the paper's Listing 4 example against a running server
├── tests/                  offline unit tests (stub server, real LLVM 17 tools) and their input modules
├── models/                 the GGUF model            (download, see models/README.md)
├── llama.cpp/              llama-server binaries     (download, see llama.cpp/README.md)
└── tools/llvm17/           LLVM 17.0.6 opt           (download, see tools/llvm17/README.md)
```

## Method (`get_passes.py`)

It follows Cummins et al., *Meta Large Language Model Compiler* (arXiv:2407.02524, 2024): flag tuning for object
size (Sec. 3.1, Fig. 3).

| Step | What happens | Paper |
|---|---|---|
| Input | A `.c` file is compiled by the project clang 17.0.6 with `-Oz -Xclang -disable-llvm-passes`: the form of the paper's Listing 4 input (`minsize optsize` attributes, no optimization applied). The whole module is sent unchanged. | Sec. 3.1, Listing 4 |
| Prompt | `[INST] Tell me how to optimize this LLVM-IR for object file size:\n\n<code>{module}</code> [/INST]` | Listing 4 |
| Context | Prompt plus 200 reply tokens must fit the 16,384-token context the model was trained with. Otherwise: error, nothing is truncated or split. | Sec. 4 |
| Decoding | Greedy (temperature 0); generation stops after the pass list. | Fig. 3 |
| Parsing | The pipeline from ``opt -p '…'``, verbatim, and the model's claimed sizes. | Listings 4, 5 |
| Validation | `opt -p '<pipeline>' -print-pipeline-passes`. An invalid pipeline is an error; nothing is substituted. | Sec. 2.2 |
| Application | `opt in.ll -S -o out.ll -p '<pipeline>'`. IR written by the model itself is never used. | Listing 1 |
| Object size | `clang -c -x ir out.ll`, then `llvm-size`: **text + data**. Gives 65 → 53 bytes for Listing 4, the paper's numbers. | Sec. 2.2 |
| Baselines | no passes, `default<O1|O2|O3|Os|Oz>` with the same commands. | Sec. 5.1 |

Differences from the paper: 4-bit (Q4_K_M) quantization; Windows COFF target (`x86_64-w64-windows-gnu`) instead of
mostly Linux ELF; no splitting of oversized modules; no PassListEval.

All tools come from the project's single LLVM 17.0.6 toolchain (`../toolchain.json`): `opt` from `tools/llvm17`,
clang and `llvm-size` from `../toolchains/llvm-mingw-17.0.6-ucrt`. Each is checked to report 17.0.6 before a run;
nothing is taken from `PATH`.

## Using it directly (optional)

A llama-server with the model must be running (`python run_all.py` starts and stops one automatically). By hand:

```powershell
.\llama.cpp\llama-server.exe -m models\llm-compiler-7b-ftd.Q4_K_M.gguf -ngl 20 -c 16384 --temp 0 `
    --jinja --chat-template-file templates\llm-compiler.jinja --host 127.0.0.1 --port 8080
```

```powershell
python get_passes.py prog.c                       # predict a pipeline for the whole module
python get_passes.py prog.c --evaluate            # + apply it and measure object size vs none and O1 … Oz
python get_passes.py prog.c --evaluate --check    # + run every variant and compare its output
python test_llm_compiler.py                       # smoke test with the paper's Listing 4 example
```

Artifacts go to `--out DIR` (default `<input>.llmc/`): `prompt.txt`, `reply.txt`, `request.json`, `response.json`,
`variants/*.ll|.o` and `result.json`.

## Tests

`python -m unittest discover -s tests -v` (from this folder). Offline: a stub server records the exact request; the
real LLVM 17.0.6 `opt`, clang and `llvm-size` are used.
