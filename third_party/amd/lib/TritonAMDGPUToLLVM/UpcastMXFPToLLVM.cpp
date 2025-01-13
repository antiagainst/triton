#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <array>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

SmallVector<Value, 4> createInlineAsmUpcast(RewriterBase &rewriter,
                                            UpcastMXFPOp upcastOp, Value vI32) {
  Location loc = upcastOp.getLoc();

  Value allEM = and_(vI32, i32_val(0x77777777));
  Value allS = and_(vI32, i32_val(0x88888888));
  // 0xXXXXXXX[X] -> 0x0000000X
  Value v0EM = and_(allEM, i32_val(0x0000000f));
  Value v0S = lshr(and_(allS, i32_val(0x0000000f)), i32_val(3));
  // 0xXXXXXX[X]X -> 0x00000X00
  Value v1EM = shl(and_(allEM, i32_val(0x000000f0)), i32_val(4));
  Value v1S = shl(and_(allS, i32_val(0x000000f0)), i32_val(4 - 3));
  // 0xXXXXX[X]XX -> 0x000X0000
  Value v2EM = shl(and_(allEM, i32_val(0x00000f00)), i32_val(8));
  Value v2S = shl(and_(allS, i32_val(0x00000f00)), i32_val(8 - 3));
  // 0xXXXX[X]XXX -> 0x0X000000
  Value v3EM = shl(and_(allEM, i32_val(0x0000f000)), i32_val(12));
  Value v3S = shl(and_(allS, i32_val(0x0000f000)), i32_val(12 - 3));
  // 0xXXX[X]XXXX -> 0x0000000X
  Value v4EM = lshr(and_(allEM, i32_val(0x000f0000)), i32_val(16));
  Value v4S = lshr(and_(allS, i32_val(0x000f0000)), i32_val(16 + 3));
  // 0xXX[X]XXXXX -> 0x00000X00
  Value v5EM = lshr(and_(allEM, i32_val(0x00f00000)), i32_val(12));
  Value v5S = lshr(and_(allS, i32_val(0x00f00000)), i32_val(12 + 3));
  // 0xX[X]XXXXXX -> 0x000X0000
  Value v6EM = lshr(and_(allEM, i32_val(0x0f000000)), i32_val(8));
  Value v6S = lshr(and_(allS, i32_val(0x0f000000)), i32_val(8 + 3));
  // 0x[X]XXXXXXX -> 0x0X000000
  Value v7EM = lshr(and_(allEM, i32_val(0xf0000000)), i32_val(4));
  Value v7S = lshr(and_(allS, i32_val(0xf0000000)), i32_val(4 + 3));

  Value v3210EMIndex = or_(v3EM, or_(v2EM, or_(v1EM, v0EM)));
  Value v3210SIndex = or_(v3S, or_(v2S, or_(v1S, v0S)));
  Value v7654EMIndex = or_(v7EM, or_(v6EM, or_(v5EM, v4EM)));
  Value v7654SIndex = or_(v7S, or_(v6S, or_(v5S, v4S)));

  // FP4 has 4 bits: S.EE.M. Bf16 bit patterns for positive values:
  //
  // FP4    | BF16   | Value
  // ------ | ------ | -----
  // 0.00.0 | 0x0000 | + 0.0
  // 0.00.1 | 0x3f00 | + 0.5
  // 0.01.0 | 0x3f80 | + 1.0
  // 0.01.1 | 0x3fc0 | + 1.5
  // 0.10.0 | 0x4000 | + 2.0
  // 0.10.1 | 0x4040 | + 3.0
  // 0.11.0 | 0x4080 | + 4.0
  // 0.11.1 | 0x40c0 | + 6.0
  Value resB10Lo = i32_val(0xc0800000);
  Value resB10Hi = i32_val(0xc0804000);
  Value resB32LoNoS = i32_val(0x3f3f3f00);
  Value resB32HiNoS = i32_val(0x40404040);

  Value resB32LoS = i32_val(0x8000);
  Value resB32HiS = i32_val(0);

  Type i32Ty = rewriter.getI32Type();
  auto permU32FnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty});
  LLVM::LLVMFuncOp funcOp = appendOrGetExternFuncOp(
      rewriter, upcastOp, "llvm.amdgcn.perm", permU32FnTy);

  auto v3210ResB10 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                            {resB10Hi, resB10Lo, v3210EMIndex});
  auto v3210ResB32NoS = LLVM::createLLVMCallOp(
      rewriter, loc, funcOp, {resB32HiNoS, resB32LoNoS, v3210EMIndex});
  auto v3210ResB32S = LLVM::createLLVMCallOp(
      rewriter, loc, funcOp, {resB32HiS, resB32LoS, v3210EMIndex});
  Value v3210ResB32 = or_(v3210ResB32NoS.getResult(), v3210ResB32S.getResult());

  auto v7654ResB10 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                            {resB10Hi, resB10Lo, v7654EMIndex});
  auto v7654ResB32NoS = LLVM::createLLVMCallOp(
      rewriter, loc, funcOp, {resB32HiNoS, resB32LoNoS, v7654EMIndex});
  auto v7654ResB32S = LLVM::createLLVMCallOp(
      rewriter, loc, funcOp, {resB32HiS, resB32LoS, v7654EMIndex});
  Value v7654ResB32 = or_(v7654ResB32NoS.getResult(), v7654ResB32S.getResult());

  auto r10 = LLVM::createLLVMCallOp(
      rewriter, loc, funcOp,
      {v3210ResB32, v3210ResB10.getResult(), i32_val(0x05010400)});
  auto r32 = LLVM::createLLVMCallOp(
      rewriter, loc, funcOp,
      {v3210ResB32, v3210ResB10.getResult(), i32_val(0x07030602)});

  auto r54 = LLVM::createLLVMCallOp(
      rewriter, loc, funcOp,
      {v7654ResB32, v7654ResB10.getResult(), i32_val(0x05010400)});
  auto r76 = LLVM::createLLVMCallOp(
      rewriter, loc, funcOp,
      {v7654ResB32, v7654ResB10.getResult(), i32_val(0x07030602)});

  return {r10.getResult(), r32.getResult(), r54.getResult(), r76.getResult()};
}

SmallVector<Value> upcast8xMxfp4ToBf16(RewriterBase &rewriter,
                                       UpcastMXFPOp upcastOp,
                                       ArrayRef<Value> values) {
  Location loc = upcastOp.getLoc();
  SmallVector<Value> results;
  MLIRContext *ctx = rewriter.getContext();
  assert(values.size() % 4 == 0);
  for (int i = 0; i < values.size(); i += 4) {
    Value v0 = values[i];
    Value v1 = values[i + 1];
    Value v2 = values[i + 2];
    Value v3 = values[i + 3];
    Value packedVec = undef(vec_ty(i8_ty, 4));
    packedVec = insert_element(packedVec, v0, i32_val(0));
    packedVec = insert_element(packedVec, v1, i32_val(1));
    packedVec = insert_element(packedVec, v2, i32_val(2));
    packedVec = insert_element(packedVec, v3, i32_val(3));
    SmallVector<Type> rets(4, i32_ty);
    SmallVector<Value, 4> ret =
        createInlineAsmUpcast(rewriter, upcastOp, bitcast(packedVec, i32_ty));
    for (int i = 0; i < 4; i++) {
      Value vecbf16 = bitcast(ret[i], vec_ty(bf16_ty, 2));
      results.push_back(extract_element(vecbf16, i32_val(0)));
      results.push_back(extract_element(vecbf16, i32_val(1)));
    }
  }
  return results;
}

SmallVector<Value> convertMxfp4x2ToFp16x2(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> values) {
  SmallVector<Value> results;
  for (auto v : values) {
    auto em0 = and_(v, i8_val(0x7));
    auto em1 = and_(v, i8_val(0x70));
    // FP16 bits: sign = 1, exponent = 5, mantissa = 10
    Value v0 = or_(shl(zext(i16_ty, em0), i16_val(10 - 1)),
                   shl(zext(i16_ty, and_(v, i8_val(0x8))), i16_val(12)));
    Value v1 = or_(shl(zext(i16_ty, em1), i16_val(10 - 1 - 4)),
                   shl(zext(i16_ty, and_(v, i8_val(0x80))), i16_val(8)));

    // Three cases:
    // 1) x is normal and non-zero: Correct bias
    v0 = select(icmp_ne(and_(em0, i8_val(0x6)), i8_val(0)),
                add(v0, i16_val((15 - 1) << 10)), v0);
    v1 = select(icmp_ne(and_(em1, i8_val(0x60)), i8_val(0)),
                add(v1, i16_val((15 - 1) << 10)), v1);

    // 2) x is subnormal (x == 0bs001 where s is the sign): Map to fp16 +-0.5
    v0 = bitcast(select(icmp_eq(em0, i8_val(0x1)),
                        or_(i16_val(0x3800), and_(v0, i16_val(0x8000))), v0),
                 f16_ty);
    v1 = bitcast(select(icmp_eq(em1, i8_val(0x10)),
                        or_(i16_val(0x3800), and_(v1, i16_val(0x8000))), v1),
                 f16_ty);
    // 3) x is zero, nothing to do
    results.push_back(v0);
    results.push_back(v1);
  }
  return results;
}

Value mxfpScaleFp16(RewriterBase &rewriter, Location loc, Value v, Value scale,
                    bool fastMath) {
  Value scaleF32 = bitcast(shl(zext(i32_ty, scale), i32_val(23)), f32_ty);
  Value scaleF16 =
      LLVM::AMD::cvtFp32ToFp16(loc, rewriter, scaleF32, RoundingMode::RTNE);
  Value mulF16 = fmul(v, scaleF16);
  if (fastMath)
    return mulF16;
  // Account for NaN in the scale as per the mxfp specification.
  Value scaleIsNan = icmp_eq(scale, i8_val(0xff));
  Value nanF16 = bitcast(i16_val(0x7c01), f16_ty);
  return select(scaleIsNan, nanF16, bitcast(mulF16, f16_ty));
};

// Scales the given bf16 v using the given scale factor without relying on bf16
// multiplication.
//
// In gfx9 architectures, we don't have bf16 VALU ops. So instead this function
// handles v * scale multiplication using fp32 VALU ops. LLVM backend can do it
// for us, just with unnecessary overheads.
Value mxfpScaleBf16ViaF32(RewriterBase &rewriter, Location loc, Value v,
                          Value scale, bool fastMath) {
  Value c16 = i32_val(16);
  Value vF32 = bitcast(shl(zext(i32_ty, bitcast(v, i16_ty)), c16), f32_ty);
  Value scaleF32 = bitcast(shl(zext(i32_ty, scale), i32_val(23)), f32_ty);
  Value mulF32 = fmul(vF32, scaleF32);
  Value mulI16 = trunc(i16_ty, lshr(bitcast(mulF32, i32_ty), c16));
  Value mulBf16 = bitcast(mulI16, bf16_ty);
  if (fastMath)
    return mulBf16;
  // Account for NaN in the scale as per the mxfp specification.
  Value scaleIsNan = icmp_eq(scale, i8_val(0xff));
  Value nanBf16 = bitcast(i16_val(0x7fff), bf16_ty);
  return select(scaleIsNan, nanBf16, mulBf16);
};

class UpcastMXFPOpPattern : public ConvertOpToLLVMPattern<UpcastMXFPOp> {
private:
  const TargetInfoBase &targetInfo;

public:
  UpcastMXFPOpPattern(LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(UpcastMXFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fpType = op.getFpType();
    bool isPacked = fpType == ScaleDotElemType::E2M1;
    if (!(isPacked || fpType == ScaleDotElemType::E4M3 ||
          fpType == ScaleDotElemType::E5M2))
      return rewriter.notifyMatchFailure(op, "NYI: non-mxfp4/mxfp8 cases");

    Location loc = op.getLoc();
    auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);
    LDBG("x: " << xVals.size() << " x " << xVals.front().getType());
    LDBG("scale: " << scaleVals.size() << " x " << scaleVals.front().getType());

    // When we lower scaled dot op, we made sure to distribute K only on one
    // warp. MXFP spec mandates 1 scale value for every 32 onsecutive values
    // along the K dimension. So in total each thread should read 32x main
    // element values.
    if (xVals.size() != scaleVals.size() * (isPacked ? 16 : 32))
      return rewriter.notifyMatchFailure(op, "unsupported problem size");

    auto dotEncoding =
        cast<DotOperandEncodingAttr>(op.getSrc().getType().getEncoding());
    auto mfmaEncoding = dyn_cast<AMDMfmaEncodingAttr>(dotEncoding.getParent());
    if (!mfmaEncoding)
      return rewriter.notifyMatchFailure(op, "NYI: non-mfma dot operand");
    LDBG("mfma: " << mfmaEncoding);

    int mDim = mfmaEncoding.getMDim();
    if (mDim != 32 && mDim != 16)
      return rewriter.notifyMatchFailure(op, "NYI: non-mfma32/16 intrinsics");

    int numThreads = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    Value warpSize = i32_val(numThreads);
    Value tid = tid_val();
    Value warpId = udiv(tid, warpSize);
    Value laneId = urem(tid, warpSize);

    bool useFp16 = op.getType().getElementType().isF16();
    if (isPacked) {
      xVals = useFp16 ? convertMxfp4x2ToFp16x2(rewriter, loc, xVals)
                      : upcast8xMxfp4ToBf16(rewriter, op, xVals);
    }

    // Given that MFMA layout for the A tensor arranges thread in a column-major
    // manner, for the current tid, it's at row (tid % mDim). When we set up
    // blocked layout for the A scale tensor, we made sure that it has a
    // threadsPerWarp = [M=mDim, K=64/mDim]. So the threads holding scale values
    // for the current thread starts at ((tid % mDim) * (64 / mDim)).
    Value offset = mul(urem(laneId, i32_val(mDim)), i32_val(numThreads / mDim));

    if (mDim == 32) {
      // One mfma32 intrinsic processes a 32x8 A tensor slice. Due to how we
      // tile, the same warp owns the whole K dim. Inside a warp, each thread
      // only holds 4 consecutive elements along K--a 1x4 vector. We need to
      // tile the warp 4 times to cover 32 values along K. So for a thread, the
      // first 4 1x4 vectors it holds shares the first scale value at row (tid %
      // mDim). the second 4 1x4 vectors shares the second scale value at row
      // (tid % mDim); and so forth.
      std::array<Value, 2> scaleThreads = {offset, add(offset, i32_val(1))};

      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        std::array<Value, 2> si = {
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[1]),
        };

        for (int j = 0; j < 32; ++j) {
          int index = 32 * i + j;
          xVals[index] =
              useFp16 ? mxfpScaleFp16(rewriter, loc, xVals[index], si[j / 16],
                                      op.getFastMath())
                      : mxfpScaleBf16ViaF32(rewriter, loc, xVals[index],
                                            si[j / 16], op.getFastMath());
        }
      }
    } else {
      assert(mDim == 16);
      // One mfma16 intrinsic processes a 16x16 A tensor slice. Similarly, we
      // need to tile the warp 2 times to cover 32 valeus. So for a thread, the
      // first 2 1x4 vectors shares the first scale value at row (tid % mDim).
      std::array<Value, 4> scaleThreads = {offset, add(offset, i32_val(1)),
                                           add(offset, i32_val(2)),
                                           add(offset, i32_val(3))};

      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        auto si = std::array<Value, 4>{
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[1]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[2]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[3]),
        };

        for (int j = 0; j < 32; ++j) {
          int index = 32 * i + j;
          xVals[index] = useFp16
                             ? mxfpScaleFp16(rewriter, loc, xVals[index],
                                             si[j / 16], op.getFastMath())
                             : mxfpScaleBf16ViaF32(rewriter, loc, xVals[index],
                                                   si[j / 8], op.getFastMath());
        }
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<UpcastMXFPOpPattern>(typeConverter, targetInfo, benefit);
}
