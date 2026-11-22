# toolchains/llvm-mingw-17.0.6-ucrt

Project-local LLVM 17.0.6 toolchain used by BOTH workflows (see ../toolchain.json).

| | |
|---|---|
| Source | https://github.com/mstorsjo/llvm-mingw/releases/tag/20231128 ("llvm-mingw 20231128 with LLVM 17.0.6") |
| Asset | `llvm-mingw-20231128-ucrt-x86_64.zip`, 134,865,419 bytes (matches GitHub's asset size; GitHub publishes no checksum) |
| SHA-256 (as downloaded 2026-11-22) | `7a344dafa6942de2c1f4643b3eb5c5ce5317fbab671a887e4d39f326b331798f` |
| Contents | the zip's folder `llvm-mingw-20231128-ucrt-x86_64/`, extracted unchanged and renamed to this folder |
| Versions | clang 17.0.6 (llvm-project 6009708b4367171ccdbf4b5905cb6a803753fe18), target x86_64-w64-windows-gnu; llvm-size 17.0.6; LLD 17.0.6 |
| Runtime | mingw-w64 with the Windows UCRT (same C runtime flavour as the previous toolchain) |
| Not included | `opt` (llvm-mingw does not ship it). The project's `opt` is `llm/tools/llvm17/bin/opt.exe`, LLVM 17.0.6 |

It is not on PATH and is not registered with WinGet; the project finds it only through `toolchain.json`.
