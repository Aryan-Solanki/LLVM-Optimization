"""Instruction counters for LLVM IR (``.ll``) and native assembly (``.s``).

The two counters measure *different things* and their numbers must never be
mixed up:

* **LLVM IR instructions** - target-independent instructions inside the body of
  every ``define`` in a ``.ll`` file (``alloca``, ``load``, ``add``, ``br``...).
* **Assembly instructions** - real machine instructions for the target CPU
  (``movl``, ``addl``, ``ret``... on x86; ``ldr``, ``add``... on ARM) in a
  ``.s`` file.

Both counters are line based state machines (no LLVM libraries needed), so they
work on files produced by any Clang version.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "IRCount",
    "AsmCount",
    "count_ir_instructions",
    "count_ir_file",
    "count_asm_instructions",
    "count_asm_file",
]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _strip_ir_comment(line: str) -> str:
    """Remove a ``;`` comment from an IR line, ignoring ``;`` inside strings."""
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_str = not in_str
        elif ch == ";" and not in_str:
            return line[:i]
    return line


def _bracket_delta(text: str) -> int:
    """Net change of ``(``/``[`` nesting on a line, ignoring quoted strings.

    Used to detect instructions that continue on the next line, e.g. a
    ``switch`` with its ``[ ... ]`` case table.
    """
    delta = 0
    in_str = False
    for ch in text:
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in "([":
                delta += 1
            elif ch in ")]":
                delta -= 1
    return delta


# --------------------------------------------------------------------------
# LLVM IR
# --------------------------------------------------------------------------
@dataclass
class IRCount:
    """Result of counting LLVM IR instructions."""

    total: int = 0
    per_function: Dict[str, int] = field(default_factory=dict)
    opcodes: Counter = field(default_factory=Counter)
    debug_intrinsics_ignored: int = 0

    @property
    def functions(self) -> int:
        return len(self.per_function)


# A basic-block label: ``entry:``, ``for.body:``, ``12:``, ``"odd name":``
_IR_LABEL = re.compile(r'^(?:[A-Za-z$._\-][\w$.\-]*|\d+|"[^"]*")\s*:')
# The ``%x = `` / ``%"quoted name" = `` prefix of a value-producing instruction.
_IR_RESULT = re.compile(r'^(?:%[\w$.\-]+|%"[^"]*")\s*=\s*')
_IR_FN_NAME = re.compile(r'@("[^"]+"|[\w$.\-]+)\s*\(')
_IR_TAIL_MARKERS = {"tail", "musttail", "notail"}
# Continuation lines printed by LLVM for ``landingpad`` clauses.
_IR_LANDINGPAD_CLAUSE = re.compile(r"^(catch|filter|cleanup)\b")


def count_ir_instructions(text: str, ignore_debug_intrinsics: bool = True) -> IRCount:
    """Count the instructions in LLVM IR text.

    What counts: every instruction inside the ``{ ... }`` body of a ``define``.

    What is ignored: blank lines, ``;`` comments, ``declare`` lines, global
    variables, ``attributes`` groups, ``!metadata`` nodes, basic-block labels,
    the ``define`` header and closing brace, ``#dbg_*`` debug records and
    (by default) calls to ``@llvm.dbg.*`` intrinsics.

    Multi-line instructions (``switch``/``indirectbr`` tables, calls with
    arguments split over lines, ``landingpad`` clauses) count once.
    """
    result = IRCount()
    in_function = False
    in_header = False          # a ``define`` line whose ``{`` is on a later line
    current_fn = ""
    depth = 0                  # open ( or [ carried over from previous lines

    for raw in text.splitlines():
        line = _strip_ir_comment(raw).strip()
        if not line:
            continue

        # ---- outside any function body ---------------------------------
        if not in_function:
            if in_header:
                if "{" in line:
                    in_header, in_function = False, True
                continue
            if line.startswith("define"):
                m = _IR_FN_NAME.search(line)
                current_fn = m.group(1).strip('"') if m else f"<anon{len(result.per_function)}>"
                result.per_function.setdefault(current_fn, 0)
                if line.rstrip().endswith("{") or "{" in line.split(")")[-1]:
                    in_function = True
                else:
                    in_header = True
            # declare, globals, attributes, metadata, target lines: ignored
            continue

        # ---- inside a function body ------------------------------------
        if depth > 0:                      # continuation of a multi-line instr.
            depth = max(0, depth + _bracket_delta(line))
            continue
        if line == "}":
            in_function = False
            continue
        if _IR_LABEL.match(line):          # basic-block label
            rest = _IR_LABEL.sub("", line, count=1).strip()
            if not rest:
                continue
            line = rest                    # (defensive) instruction after label
        if line.startswith("#dbg_") or line.startswith("uselistorder"):
            depth = max(0, depth + _bracket_delta(line))
            continue
        if line.startswith("!"):           # stray metadata attachment line
            continue
        if _IR_LANDINGPAD_CLAUSE.match(line):
            continue                       # part of the previous landingpad

        body = _IR_RESULT.sub("", line, count=1)
        tokens = body.split()
        if not tokens:
            continue
        opcode = tokens[0]
        if opcode in _IR_TAIL_MARKERS and len(tokens) > 1:
            opcode = tokens[1]

        depth = max(0, _bracket_delta(line))

        if ignore_debug_intrinsics and "@llvm.dbg." in body:
            result.debug_intrinsics_ignored += 1
            continue

        result.total += 1
        result.per_function[current_fn] += 1
        result.opcodes[opcode] += 1

    return result


def count_ir_file(path: Path, ignore_debug_intrinsics: bool = True) -> IRCount:
    return count_ir_instructions(
        Path(path).read_text(encoding="utf-8", errors="replace"),
        ignore_debug_intrinsics,
    )


# --------------------------------------------------------------------------
# Native assembly
# --------------------------------------------------------------------------
@dataclass
class AsmCount:
    """Result of counting machine instructions in an assembly file."""

    total: int = 0
    mnemonics: Counter = field(default_factory=Counter)


# Leading label(s): ``main:``  ``.LBB0_1:``  ``_foo:``  ``"?bar@@YAXXZ":``  ``1:``
_ASM_LABEL = re.compile(r'^(?:"[^"]*"|[\w$.@?\-]+)\s*:')
_ASM_COMMENT_PREFIXES = ("#", "//", ";", "@", "|")


def count_asm_instructions(text: str) -> AsmCount:
    """Count machine instructions in assembly text (AT&T/GAS-style output).

    Ignored: blank lines, comment-only lines (``#``, ``//``, ``;``, ``@``,
    ``/* ... */``), labels (also when followed by an instruction on the same
    line, which *is* counted), and assembler directives (anything starting with
    ``.``, e.g. ``.text``, ``.cfi_startproc``, ``.seh_endprologue``).

    Limitations: Intel-syntax assembly with MASM-style directives (``PROC``,
    ``END``) is not recognised; multiple ``;``-separated statements on one line
    count as one.
    """
    result = AsmCount()
    in_block_comment = False

    for raw in text.splitlines():
        line = raw.strip()

        # /* ... */ block comments (may span lines)
        if in_block_comment:
            end = line.find("*/")
            if end == -1:
                continue
            line = line[end + 2:].strip()
            in_block_comment = False
        while "/*" in line:
            start = line.find("/*")
            end = line.find("*/", start + 2)
            if end == -1:
                line = line[:start].strip()
                in_block_comment = True
                break
            line = (line[:start] + " " + line[end + 2:]).strip()

        if not line or line.startswith(_ASM_COMMENT_PREFIXES):
            continue

        # peel off any leading labels
        while True:
            m = _ASM_LABEL.match(line)
            if not m:
                break
            line = line[m.end():].strip()
        if not line or line.startswith(_ASM_COMMENT_PREFIXES):
            continue
        if line.startswith("."):           # directive
            continue

        mnemonic = line.split()[0].lower()
        result.total += 1
        result.mnemonics[mnemonic] += 1

    return result


def count_asm_file(path: Path) -> AsmCount:
    return count_asm_instructions(Path(path).read_text(encoding="utf-8", errors="replace"))
