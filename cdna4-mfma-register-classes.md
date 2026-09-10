# CDNA4 MFMA register-class experiments

PR [#10337](https://github.com/triton-lang/triton/pull/10337) is checked out on
`pr-10337-amd-register-classes`, rebased onto the latest `origin/main` fetched
for this task. The PR makes it possible to steer native Gluon MFMA operands
through AGPR or VGPR tuples. Direct inline MFMA can enforce the instruction's
register classes. Native MFMA constraints apply at the surrounding asm boundaries;
LLVM may satisfy them with copies.

- Fetched PR head: `8e0131148b` (three commits).
- Rebase base: `4a15f415d8ac4f830ce788c6fca3a5b3b29908e7`.
- Rebased PR tip: `f38b1c391729e09c3b5c7f8043dcf2d746f1d369`.
- Device: AMD Instinct MI350X, `gfx950`, GPU 0.
- Compiler: the repository build using LLVM `b010a18d`.
- LLVM reference source: `~/llvm`, at `4cbc99688044` when inspected.

Rebase conflicts were in `TritonOps.td`, `ElementwiseOpToLLVM.cpp`, and `Ops.cpp`.
The resolution preserves main's memory-descriptor operands and descriptor effect
verification. A descriptor contributes one address even when other arguments are
packed; its vector size must remain one. Positive and negative lit coverage for
that combination is included in the working tree.

## Results

| Experiment | Result |
| --- | --- |
| Native `amd.cdna4.mfma`, A in AGPR or VGPR | Both worked. |
| Native MFMA, B in AGPR or VGPR | Both worked, independently of A. |
| Native MFMA, both C and D constrained to AGPR | An AGPR accumulator/result instruction was emitted in the tested kernels. |
| Native MFMA, both C and D constrained to VGPR | A VGPR accumulator/result instruction was emitted. |
| Only the native result constrained to AGPR | Selected AGPR C/D in the tested 16x16 kernel, with no `v_accvgpr_*` copies. |
| Native C and D constrained to different classes | Numerically correct, but MFMA itself still uses one class for C/D. Copies satisfy the other boundary constraint. |
| Direct inline MFMA, all eight A/B/C-D class combinations | All compiled and passed GPU correctness checks. |
| Direct inline MFMA, C/D in different register classes | Rejected by the assembler in both directions. |
| Four-dword tuple in the PR (`i128`) | Works with `a`, `v`, and `^VA`. |
| Eight/sixteen-dword tuple in the PR (`i256`/`i512`) | LLVM reports that it cannot allocate the output register. |
| The same wide tuples expressed as LLVM vectors | `<8 x i32>` and `<16 x i32>` work with all three constraints in standalone LLVM probes. |

C denotes the input accumulator and D the output. C and D must use the same
register **class**, but need not use the same register **numbers**. A and B each
have an independent AGPR/VGPR selector. LLVM's instruction encoding has separate
`acc(0)` and `acc(1)` bits for A/B and a shared `acc_cd` bit for C/D; see
`~/llvm/llvm/lib/Target/AMDGPU/VOPInstructions.td:543` and the paired instruction
profiles in `VOP3PInstructions.td:1042`.

## Using the PR with native MFMA

The reusable helper is in
[test_cdna4_mfma_register_classes.py](test_cdna4_mfma_register_classes.py), named
`_constrain_mfma_register_class`:

```python
@gluon.jit
def constrain(x, CLASS: gl.constexpr, PACK: gl.constexpr):
    VEC: gl.constexpr = PACK * x.dtype.primitive_bitwidth // 32
    return gl.inline_asm_elementwise(
        "", "=" + CLASS + ",0", [x], dtype=x.dtype,
        is_pure=True, pack=PACK,
        operand_vec_sizes=[VEC], result_vec_sizes=[VEC],
    )

# a and b retain their valid DotOperandLayouts; acc retains its AMDMFMALayout.
a = constrain(a, "a", 8)       # eight FP16/BF16 values -> four dwords
b = constrain(b, "v", 8)
acc = constrain(acc, "a", 4)   # four FP32 values -> four dwords
out = gl.amd.cdna4.mfma(a, b, acc)
out = constrain(out, "a", 4)
```

`"=a,0"` is an empty, tied-output asm: it preserves the bits and requires the
value to pass through an AGPR tuple. `"=v,0"` does the same for VGPRs. These
constraints do not themselves emit a move instruction, although LLVM can insert
copies to satisfy them. Tensor layouts describe element distribution; these
constraints control physical register classes at asm boundaries.

The flexible AMD constraint is `VA`; LLVM IR spells a two-character constraint
with a caret: `"=^VA,0"`. Plain `"=VA,0"` is parsed incorrectly and gives an
allocation diagnostic for `A`. See `SIISelLowering.cpp:19826` and
`~/llvm/llvm/lib/IR/InlineAsm.cpp:198`.

For a 32x32 FP32 accumulator, use `PACK=4`. It inserts four asm invocations per
lane, one for each four-dword group. Using `PACK=16` asks this PR for `i512` and
hits the wide-integer limitation. Grouping can also introduce copies to assemble
the complete contiguous MFMA register tuple.

Representative assembly from the one-instruction 16x16 FP16 cases:

```asm
; A/B in VGPR, C/D in AGPR, no accvgpr copies in this kernel:
v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], v[4:7], a[0:3]

; All four operands in AGPR:
v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[8:11], a[4:7]

; C constrained to VGPR and D constrained to AGPR:
v_mfma_f32_16x16x32_f16 v[0:3], v[0:3], v[8:11], v[4:7]
; Followed by four v_accvgpr_write_b32 operations for the result constraint.
```

Forcing contradictory C/D classes inside a repeated accumulation can cause
copies on every iteration. Even compatible constraints can change register
pressure and generate tuple-reassembly moves. These are correctness and assembly
experiments; they do not establish a performance improvement.

## Direct inline MFMA

`_amd_mfma_inline_register_classes_kernel` demonstrates direct enforcement using
`pack=4`, `operand_vec_sizes=[4, 4, 4]`, and `result_vec_sizes=[4]`. It explicitly
packs pairs of FP16 values into uint32 tensors with a valid distribution so all
inputs and the result have the common shape/layout required by
`inline_asm_elementwise`.

The instruction constraints are `"=&" + DCLASS + "," + ACLASS + "," + BCLASS +
"," + CCLASS`. The early-clobber output keeps the result disjoint from inputs.
All eight legal combinations were checked against `C + A.float() @ B.float()`.
Both mixed-C/D cases fail with `invalid operand for instruction`.

The direct-asm example includes conservative `s_nop` padding before and after
MFMA because LLVM cannot see the instruction hazards inside the asm block. It
is intended to demonstrate register placement, not to supply a tuned GEMM. No
source-level lane IDs, divergent scalars, or invalid layouts are used.

## Wide tuple limitation

The PR's third commit, "Bitcast packed vector types to integer types for NVPTX",
also applies those bitcasts to AMD. `ElementwiseOpToLLVM.cpp:262` converts packed
inputs to integers and the matching return-type code does the same for outputs.
AMD's inline-asm constraint lowering accepts legal vector types and specially
accepts `i128`, but not arbitrary `i256`/`i512` integers. This is a type-lowering
limitation, not a lack of wide AGPR/VGPR register classes.

The standalone LLVM probe matrix covered integer and vector representations at
128, 256, and 512 bits with `a`, `v`, and `^VA`: 12 accepted cases and six expected
rejections. A future fix should keep native vector operands/results for AMD and
limit the NVPTX integer conversion to the backend that needs it. That fix has
not been applied here; the branch retains the rebased PR's behavior.

A failed wide-integer compile can print a diagnostic and still return a Triton
artifact with no MFMA. The experiment driver checks that an MFMA was emitted
before running that artifact, and initializes output tensors to NaN before
correctness checks.

## Reproduction and validation

Run from the Triton root with the gfx950 environment:

```bash
make PYTHON=.venv/bin/python
.venv/bin/pytest -s --tb=short test_cdna4_mfma_register_classes.py python/test/gluon/test_frontend.py::test_inline_asm_shared_amd_compilation
.venv/bin/pytest -s --tb=short third_party/amd/python/test/test_inline_asm_gfx1250.py
```

The standalone MFMA test file passed 184 cases: 176 native MFMA cases and eight
direct inline MFMA cases. The two existing descriptor compilation cases also
passed, for 186 cases in total. Native cases
cover FP16/BF16, 16x16x32 and 32x32x16 instructions, all 16 explicit A/B/C/D
constraint combinations plus six baseline/one-sided/flexible cases, and runtime
loops with one/eight iterations. These use one wave and non-transposed layouts.
The PR's four gfx1250 compile-only cases also passed.

From `build/cmake.linux-x86_64-cpython-3.12`, `ninja triton-opt` and these lit
files passed:

- `test/Conversion/tritongpu_to_llvm.mlir`
- `test/Conversion/amd/tritongpu_to_llvm_gfx1250.mlir`
- `test/Triton/invalid.mlir`
- `test/TritonGPU/invalid.mlir`

Ruff, YAPF, and `git diff --check` passed for the experiment changes.

Raw reproducers and generated TTGIR, LLVM IR, AMD assembly, diagnostic logs, and
JSON summaries are in `/tmp/cdna4-mfma-register-classes/`. The driver there can
be rerun with:

```bash
.venv/bin/python /tmp/cdna4-mfma-register-classes/experiment.py
.venv/bin/python /tmp/cdna4-mfma-register-classes/experiment.py --size 32
.venv/bin/python /tmp/cdna4-mfma-register-classes/experiment.py --mode direct
# Expected compile failures:
.venv/bin/python /tmp/cdna4-mfma-register-classes/experiment.py --size 32 --cpack 16 --classes v,v,a,a --compile-only
.venv/bin/python /tmp/cdna4-mfma-register-classes/experiment.py --mode direct --classes v,v,a,v --compile-only
```

The `dynamic_*` artifacts come from the persistent pytest kernel with a runtime
loop bound. `width_summary.json` records the pinned-LLVM type/constraint probes.
`encoding_summary.json` records 16 assembler probes with eight accepted and
eight rejected C/D class combinations; those auxiliary probes used
`~/build/mlir-debug/bin/llvm-mc`. The direct Triton tests and both mixed-C/D
failures were independently checked with Triton's pinned LLVM.

The experiment tests, descriptor compatibility tests, and this note form a
follow-up to the three rebased PR commits.
