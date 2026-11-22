"""TEST DOUBLE - not a compiler.

Mimics just enough of clang's command line to exercise the project's plumbing
(file naming, timing, counting, CSV, charts) when no real LLVM is installed.
The "IR" and "assembly" it writes are canned text with a different number of
instructions per level.  Numbers produced with it are meaningless and must never
be reported as experimental results.

The project refuses any clang that is not the required version/target (src/toolchain.py), so by default the double
reports the required ones; set FAKE_CLANG_VERSION / FAKE_CLANG_TARGET to test the refusal.
"""
import os
import sys

VERSION = os.environ.get("FAKE_CLANG_VERSION", "17.0.6")
TARGET = os.environ.get("FAKE_CLANG_TARGET", "x86_64-w64-windows-gnu")
args = sys.argv[1:]
if "--version" in args:
    print(f"clang version {VERSION} (test double, not a compiler)\nTarget: {TARGET}\nThread model: posix")
    sys.exit(0)
if "-dumpmachine" in args:
    print(TARGET)
    sys.exit(0)

LEVELS = {"-O0": 12, "-O1": 8, "-O2": 6, "-O3": 6, "-Os": 7, "-Oz": 5}
level = next((a for a in args if a in LEVELS), "-O0")
n = LEVELS[level]
out = args[args.index("-o") + 1]
src = [a for a in args if a.endswith(".c")]
if not src:
    print("fake clang: no input", file=sys.stderr)
    sys.exit(1)

if "-S" in args and "-emit-llvm" in args:
    body = "".join(f"  %v{i} = add i32 1, {i}\n" for i in range(n - 1)) + "  ret i32 0\n"
    text = f"; fake\ndeclare i32 @puts(ptr)\ndefine i32 @main() {{\nentry:\n{body}}}\n"
elif "-S" in args:
    text = "\t.text\nmain:\n" + "\tnop\n" * n + "\t.addrsig\n"
else:
    text = "FAKE EXE " * (100 + n)
with open(out, "w") as fh:
    fh.write(text)
