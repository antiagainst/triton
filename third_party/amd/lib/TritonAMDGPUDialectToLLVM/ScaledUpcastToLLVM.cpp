#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

template <typename ConvertOp>
SmallVector<Value, 4> upcast8xMxfp4CDNA4(RewriterBase &rewriter, Location loc,
                                         ArrayRef<Value> xVals, int idx,
                                         Value scale) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  for (int i : llvm::seq(4))
    packedVec = b.insert_element(packedVec, xVals[idx + i], b.i32_val(i));
  packedVec = b.bitcast(packedVec, i32_ty);
  Type retElemType = bf16_ty;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp4Op>)
    retElemType = f16_ty;
  Type resType = vec_ty(retElemType, 2);
  if (scale.getType().isBF16())
    scale = b.bitcast(
        b.shl(b.zext(i32_ty, b.bitcast(scale, i16_ty)), b.i32_val(16)), f32_ty);
  SmallVector<Value, 4> results;
  // Intentionally swap the byte indices 1 and 2 to align with how the LLVM
  // backend accesses them
  for (int srcSelIndex : {0, 2, 1, 3})
    results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec, scale,
                                                 srcSelIndex));
  return results;
}

template <typename ConvertOp>
SmallVector<Value, 2> upcast4xMxfp8CDNA4(RewriterBase &rewriter, Location loc,
                                         ArrayRef<Value> xVals, int idx,
                                         Value scale) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  for (int i : llvm::seq(4))
    packedVec = b.insert_element(packedVec, xVals[idx + i], b.i32_val(i));
  packedVec = b.bitcast(packedVec, i32_ty);
  Type retElemType = bf16_ty;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp8Op> ||
                std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Bf8Op>)
    retElemType = f16_ty;
  if (scale.getType().isBF16())
    scale = b.bitcast(
        b.shl(b.zext(i32_ty, b.bitcast(scale, i16_ty)), b.i32_val(16)), f32_ty);
  Type resType = vec_ty(retElemType, 2);
  SmallVector<Value, 2> results;
  results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec, scale,
                                               /*srcLoHiSel=*/false));
  results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec, scale,
                                               /*srcLoHiSel=*/true));
  return results;
}

struct ScaledUpcastFp4Pattern
    : ConvertOpToLLVMPattern<amdgpu::ScaledUpcastFp4Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(amdgpu::ScaledUpcastFp4Op upcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = upcastOp.getLoc();
    auto elemType = upcastOp.getType().getElementType();

    auto inputVals = unpackLLElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);

    assert(inputVals.size() % 4 == 0);
    SmallVector<Value> results;
    results.reserve(inputVals.size() * 2);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (int i = 0; i < inputVals.size(); i += 4) {
      SmallVector<Value, 4> v4i32;
      if (elemType.isBF16()) {
        v4i32 = upcast8xMxfp4CDNA4<ROCDL::CvtScaleF32PkBf16Fp4Op>(
            rewriter, loc, inputVals, i, scaleVals[i * 2]);
      } else {
        v4i32 = upcast8xMxfp4CDNA4<ROCDL::CvtScaleF32PkF16Fp4Op>(
            rewriter, loc, inputVals, i, scaleVals[i * 2]);
      }
      for (int j = 0; j < 4; j++) {
        Value elements = b.bitcast(v4i32[j], vec_ty(elemType, 2));
        results.push_back(b.extract_element(elements, b.i32_val(0)));
        results.push_back(b.extract_element(elements, b.i32_val(1)));
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  upcastOp.getType());
    rewriter.replaceOp(upcastOp, result);
    return success();
  }
};

struct ScaledUpcastFp8Pattern
    : ConvertOpToLLVMPattern<amdgpu::ScaledUpcastFp8Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(amdgpu::ScaledUpcastFp8Op upcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = upcastOp.getLoc();
    auto elemType = upcastOp.getType().getElementType();

    auto inputVals = unpackLLElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);

    assert(inputVals.size() % 4 == 0);
    SmallVector<Value> results;
    results.reserve(inputVals.size() * 2);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (int i = 0; i < inputVals.size(); i += 4) {
      SmallVector<Value, 4> v2i32;
      if (elemType.isBF16()) {
        v2i32 = upcast4xMxfp8CDNA4<ROCDL::CvtScaleF32PkBf16Fp8Op>(
            rewriter, loc, inputVals, i, scaleVals[i]);
      } else {
        v2i32 = upcast4xMxfp8CDNA4<ROCDL::CvtScaleF32PkF16Fp8Op>(
            rewriter, loc, inputVals, i, scaleVals[i]);
      }
      for (int j = 0; j < 2; j++) {
        Value elements = b.bitcast(v2i32[j], vec_ty(elemType, 2));
        results.push_back(b.extract_element(elements, b.i32_val(0)));
        results.push_back(b.extract_element(elements, b.i32_val(1)));
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  upcastOp.getType());
    rewriter.replaceOp(upcastOp, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::AMD::populateScaledUpcastOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ScaledUpcastFp4Pattern>(typeConverter, benefit);
}
