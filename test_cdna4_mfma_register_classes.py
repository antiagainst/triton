"""CDNA4 MFMA register-class experiments for PR #10337."""

import re
from itertools import product

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


@gluon.jit
def _constrain_mfma_register_class(x, CLASS: ttgl.constexpr, PACK: ttgl.constexpr):
    if CLASS != "none":
        VEC: ttgl.constexpr = PACK * x.dtype.primitive_bitwidth // 32
        x = ttgl.inline_asm_elementwise("", "=" + ("^VA" if CLASS == "VA" else CLASS) + ",0", [x], x.dtype,
                                        is_pure=True, pack=PACK, operand_vec_sizes=[VEC], result_vec_sizes=[VEC])
    return x


@gluon.jit(do_not_specialize=["REPS"])
def _amd_mfma_register_classes_kernel(A, B, C, D, ACLASS: ttgl.constexpr, BCLASS: ttgl.constexpr,
                                      CCLASS: ttgl.constexpr, DCLASS: ttgl.constexpr, SIZE: ttgl.constexpr, REPS,
                                      CPACK: ttgl.constexpr):
    K: ttgl.constexpr = 32 if SIZE == 16 else 16
    L: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[SIZE, SIZE, K], transposed=False,
                                               warps_per_cta=[1, 1])
    LA: ttgl.constexpr = ttgl.DotOperandLayout(0, L, 8)
    LB: ttgl.constexpr = ttgl.DotOperandLayout(1, L, 8)
    am = ttgl.arange(0, SIZE, ttgl.SliceLayout(1, LA))
    ak = ttgl.arange(0, K, ttgl.SliceLayout(0, LA))
    bk = ttgl.arange(0, K, ttgl.SliceLayout(1, LB))
    bn = ttgl.arange(0, SIZE, ttgl.SliceLayout(0, LB))
    cm = ttgl.arange(0, SIZE, ttgl.SliceLayout(1, L))
    cn = ttgl.arange(0, SIZE, ttgl.SliceLayout(0, L))
    a = ttgl.load(A + am[:, None] * K + ak[None, :])
    b = ttgl.load(B + bk[:, None] * SIZE + bn[None, :])
    c = ttgl.load(C + cm[:, None] * SIZE + cn[None, :])
    a = _constrain_mfma_register_class(a, ACLASS, 8)
    b = _constrain_mfma_register_class(b, BCLASS, 8)
    for _ in range(REPS):
        c = _constrain_mfma_register_class(c, CCLASS, CPACK)
        c = ttgl.amd.cdna4.mfma(a, b, c)
        c = _constrain_mfma_register_class(c, DCLASS, CPACK)
    ttgl.store(D + cm[:, None] * SIZE + cn[None, :], c)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("size", [16, 32])
@pytest.mark.parametrize("reps", [1, 8])
@pytest.mark.parametrize("in_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "classes",
    list(product(("a", "v"), repeat=4)) + [
        ("none", "none", "none", "none"),
        ("none", "none", "none", "a"),
        ("none", "none", "a", "none"),
        ("none", "none", "none", "v"),
        ("none", "none", "v", "none"),
        ("VA", "VA", "VA", "VA"),
    ])
def test_amd_mfma_register_classes(size, reps, in_dtype, classes):
    torch.manual_seed(4)
    k = 32 if size == 16 else 16
    a = torch.randn((size, k), device="cuda", dtype=in_dtype)
    b = torch.randn((k, size), device="cuda", dtype=in_dtype)
    c = torch.randn((size, size), device="cuda", dtype=torch.float32)
    d = torch.full_like(c, float("nan"))
    # Four dwords fit in i128. A whole 32x32 accumulator becomes i512,
    # which integer packing cannot use as an AMD inline-asm operand.
    compiled = _amd_mfma_register_classes_kernel[(1, )](a, b, c, d, *classes, size, reps, 4, num_warps=1,
                                                        enable_fp_fusion=False)
    assert "v_mfma_f32_" in compiled.asm["amdgcn"]
    ref = c + reps * (a.float() @ b.float())
    torch.testing.assert_close(d, ref, atol=2e-4, rtol=1e-4)


@gluon.jit
def _amd_mfma_inline_register_classes_kernel(A, B, C, D, ACLASS: ttgl.constexpr, BCLASS: ttgl.constexpr,
                                             CCLASS: ttgl.constexpr, DCLASS: ttgl.constexpr):
    L: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=False,
                                               warps_per_cta=[1, 1])
    m = ttgl.arange(0, 16, ttgl.SliceLayout(1, L))[:, None]
    n = ttgl.arange(0, 16, ttgl.SliceLayout(0, L))[None, :]
    # L maps m = 4 * (lane // 16) + register, n = lane % 16.
    # The four registers of each input pack eight adjacent K elements.
    a0 = ttgl.load(A + n * 32 + m * 2).to(ttgl.uint16, bitcast=True).to(ttgl.uint32)
    a1 = ttgl.load(A + n * 32 + m * 2 + 1).to(ttgl.uint16, bitcast=True).to(ttgl.uint32)
    b0 = ttgl.load(B + m * 2 * 16 + n).to(ttgl.uint16, bitcast=True).to(ttgl.uint32)
    b1 = ttgl.load(B + (m * 2 + 1) * 16 + n).to(ttgl.uint16, bitcast=True).to(ttgl.uint32)
    a = a0 | (a1 << 16)
    b = b0 | (b1 << 16)
    c = ttgl.load(C + m * 16 + n)
    # Conservative padding handles hazards hidden inside this asm block.
    d = ttgl.inline_asm_elementwise(
        "s_nop 7\ns_nop 7\ns_nop 7\ns_nop 7\n"
        "v_mfma_f32_16x16x32_f16 $0, $1, $2, $3\n"
        "s_nop 7\ns_nop 7\ns_nop 7\ns_nop 7", "=&" + DCLASS + "," + ACLASS + "," + BCLASS + "," + CCLASS, [a, b, c],
        dtype=ttgl.float32, is_pure=True, pack=4, operand_vec_sizes=[4, 4, 4], result_vec_sizes=[4])
    ttgl.store(D + m * 16 + n, d)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("a_class,b_class,cd_class", list(product(("a", "v"), repeat=3)))
def test_amd_mfma_inline_register_classes(a_class, b_class, cd_class):
    torch.manual_seed(4)
    a = torch.randn((16, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((32, 16), device="cuda", dtype=torch.float16)
    c = torch.randn((16, 16), device="cuda", dtype=torch.float32)
    d = torch.full_like(c, float("nan"))
    compiled = _amd_mfma_inline_register_classes_kernel[(1, )](a, b, c, d, a_class, b_class, cd_class, cd_class,
                                                               num_warps=1)
    operands = ", ".join(rf"{cls}\[\d+:\d+\]" for cls in (cd_class, a_class, b_class, cd_class))
    assert re.search(r"v_mfma_f32_16x16x32_f16 " + operands, compiled.asm["amdgcn"])
    torch.testing.assert_close(d, c + a.float() @ b.float(), atol=1e-4, rtol=1e-4)
