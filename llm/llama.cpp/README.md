# llm/llama.cpp

Put the llama.cpp server binaries here (not stored in git). The project starts `llama-server.exe` from this folder.

Tested build: **b12421** (Windows x64, CUDA 12.4), from https://github.com/ggml-org/llama.cpp/releases/tag/b12421

1. `llama-b12421-bin-win-cuda-12.4-x64.zip` (llama-server.exe and the ggml/llama DLLs)
2. `cudart-llama-bin-win-cuda-12.4-x64.zip` (CUDA runtime DLLs)

Extract both zips **directly into this folder**, so that `llm/llama.cpp/llama-server.exe` exists.

* Needs an NVIDIA GPU with a recent driver. With 6 GB of VRAM the project offloads 20 layers (`-ngl 20`).
* Without an NVIDIA GPU use the CPU build of the same release (`llama-b12421-bin-win-cpu-x64.zip`); it is slower.
* A newer build usually works as well; `python run_all.py --check` shows whether the server can be started.
