# llm/tools/llvm17

LLVM 17.0.6 `opt` (not stored in git). `toolchain.json` expects `bin/opt.exe` here; llvm-mingw (the clang 17 in
`toolchains/`) does not ship `opt`.

Source: conda-forge package **llvm-tools 17.0.6** (build `h2a44499_3`, win-64) and its two DLL dependencies.

```powershell
# with conda / mamba / micromamba installed:
conda create -p .\_llvm17 -c conda-forge llvm-tools=17.0.6 --yes
New-Item -ItemType Directory -Force llm\tools\llvm17\bin
copy .\_llvm17\Library\bin\opt.exe     llm\tools\llvm17\bin\
copy .\_llvm17\Library\bin\zlib.dll    llm\tools\llvm17\bin\
copy .\_llvm17\Library\bin\libzstd.dll llm\tools\llvm17\bin\
if (Test-Path .\_llvm17\Library\bin\zstd.dll) { copy .\_llvm17\Library\bin\zstd.dll llm\tools\llvm17\bin\ }
Remove-Item -Recurse -Force .\_llvm17
```

Files used here (SHA-256):

| File | SHA-256 |
|---|---|
| `opt.exe` (llvm-tools 17.0.6 h2a44499_3) | `c1c70789b890dfe92cd3adcb24ca7be3d08e3eaced39abf36c01d6af86309213` |
| `zlib.dll` (libzlib 1.3.1) | `7d86de8659d728d0cc22615ea37248ca83b24af64a9fc19ecc4548105308ee0a` |
| `libzstd.dll` / `zstd.dll` (zstd 1.5.6) | `626adec52d89c6275e5d58427f6c44a76c94ea2fe700a663d6df5610065fbdb4` |

`python run_all.py --check` verifies that `opt --version` reports exactly 17.0.6.
