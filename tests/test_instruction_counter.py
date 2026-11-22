"""Unit tests for the instruction counters.

The IR / assembly snippets below are HAND-WRITTEN FIXTURES that exercise the
parsers (comments, labels, declares, metadata, multi-line switch...).  They are
not output of the experiment and say nothing about optimization results.

Run:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.instruction_counter import count_asm_instructions, count_ir_instructions  # noqa: E402

IR_FIXTURE = r"""
; ModuleID = 'x.c'
source_filename = "x.c"
target datalayout = "e-m:w-p270:32:32"
target triple = "x86_64-pc-windows-msvc"

@.str = private unnamed_addr constant [9 x i8] c"a; b %d\0A\00", align 1   ; semicolon inside string
@g = global i32 5, align 4

declare i32 @printf(ptr, ...)   ; declaration: ignored

; a comment line

define dso_local i32 @add(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4          ; 1
  %b.addr = alloca i32, align 4          ; 2
  store i32 %a, ptr %a.addr, align 4     ; 3
  store i32 %b, ptr %b.addr, align 4     ; 4
  %0 = load i32, ptr %a.addr, align 4    ; 5
  %1 = load i32, ptr %b.addr, align 4    ; 6
  %add = add nsw i32 %0, %1              ; 7
  ret i32 %add                           ; 8
}

define i32 @sw(i32 %x) {
entry:
  switch i32 %x, label %def [
    i32 0, label %a
    i32 1, label %b
  ]
a:                                                ; preds = %entry
  br label %end
b:
  br label %end
def:
  br label %end
"quoted label":
  br label %end
end:
  %r = phi i32 [ 1, %a ], [ 2, %b ], [ 3, %def ]
  %c = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %r)
  call void asm sideeffect "nop; nop", ""()
  call void @llvm.dbg.value(metadata i32 %r, metadata !9, metadata !DIExpression()), !dbg !10
  ret i32 %r
}

attributes #0 = { noinline nounwind optnone uwtable }
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 2}
"""
# add: 8.  sw: switch(1) + br x4 + phi + call + inline-asm call + ret = 1+4+1+1+1+1 = 9
# (the dbg.value intrinsic call is ignored by default)


class IRCounterTests(unittest.TestCase):
    def test_total_and_per_function(self):
        r = count_ir_instructions(IR_FIXTURE)
        self.assertEqual(r.per_function["add"], 8)
        self.assertEqual(r.per_function["sw"], 9)
        self.assertEqual(r.total, 17)
        self.assertEqual(r.functions, 2)

    def test_debug_intrinsics_can_be_counted(self):
        r = count_ir_instructions(IR_FIXTURE, ignore_debug_intrinsics=False)
        self.assertEqual(r.total, 18)

    def test_opcode_histogram(self):
        r = count_ir_instructions(IR_FIXTURE)
        self.assertEqual(r.opcodes["alloca"], 2)
        self.assertEqual(r.opcodes["store"], 2)
        self.assertEqual(r.opcodes["switch"], 1)
        self.assertEqual(r.opcodes["call"], 2)      # tail call + inline asm call
        self.assertEqual(r.opcodes["br"], 4)

    def test_declarations_globals_metadata_ignored(self):
        text = 'declare i32 @f(i32)\n@x = global i32 1\n!0 = !{i32 1}\nattributes #0 = { nounwind }\n'
        self.assertEqual(count_ir_instructions(text).total, 0)

    def test_empty_input(self):
        self.assertEqual(count_ir_instructions("").total, 0)

    def test_landingpad_clauses_are_not_separate_instructions(self):
        text = (
            "define void @f() personality ptr @p {\n"
            "entry:\n"
            "  %lp = landingpad { ptr, i32 }\n"
            "          cleanup\n"
            "          catch ptr null\n"
            "  ret void\n"
            "}\n"
        )
        self.assertEqual(count_ir_instructions(text).total, 2)

    def test_multiline_call(self):
        text = (
            "define void @f() {\nentry:\n"
            "  call void @g(i32 1,\n"
            "               i32 2)\n"
            "  ret void\n}\n"
        )
        self.assertEqual(count_ir_instructions(text).total, 2)

    def test_crlf_line_endings(self):
        self.assertEqual(count_ir_instructions(IR_FIXTURE.replace("\n", "\r\n")).total, 17)


ASM_FIXTURE = r"""
	.text
	.def	 main;
	.scl	2;
	.endef
	.file	"x.c"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.seh_proc main
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	movq	%rsp, %rbp
	/* block
	   comment movl $1, %eax */
	xorl	%eax, %eax        # trailing comment
.LBB0_1:
	addl	$1, %eax
	cmpl	$10, %eax
	jne	.LBB0_1
	popq	%rbp
	retq
	.seh_endproc
foo: nop
	// arm-style comment line
	; another comment style
	.addrsig
"""
# pushq movq xorl addl cmpl jne popq retq + "foo: nop" = 9


class AsmCounterTests(unittest.TestCase):
    def test_count(self):
        r = count_asm_instructions(ASM_FIXTURE)
        self.assertEqual(r.total, 9)

    def test_mnemonics(self):
        r = count_asm_instructions(ASM_FIXTURE)
        self.assertEqual(r.mnemonics["nop"], 1)
        self.assertEqual(r.mnemonics["retq"], 1)

    def test_arm_immediates_are_not_comments(self):
        text = "foo:\n\tmov\tx0, #1\n\tadd\tx0, x0, #2 // comment\n\tret\n"
        self.assertEqual(count_asm_instructions(text).total, 3)

    def test_directives_labels_only(self):
        self.assertEqual(count_asm_instructions("\t.text\nL1:\n.Lfunc_end0:\n\t.size f, .-f\n").total, 0)


if __name__ == "__main__":
    unittest.main()
