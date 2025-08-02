#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

// Goes through the RHS of mul op to figure out the BF16/FP32 scale value.
TypedValue<RankedTensorType> getUpcastFp32Scale(arith::MulFOp mulOp) {
  Value scale = mulOp.getRhs();
  auto scaleTy = dyn_cast<RankedTensorType>(scale.getType());
  if (!scaleTy)
    return {};

  // First check whether the scale is encoded as a BF16 value.
  if (scaleTy.getElementType().isBF16())
    return cast<TypedValue<RankedTensorType>>(scale);

  // Otherwise, for FP16 case, we need to figure out the original FP32 value.
  if (!scaleTy.getElementType().isF16())
    return {};
  // Go through various shape manipulation ops.

  arith::TruncFOp truncOp = nullptr;
  while (scale) {
    llvm::TypeSwitch<Operation *>(scale.getDefiningOp())
        .Case<tt::BroadcastOp, tt::ExpandDimsOp, tt::ReshapeOp, tt::TransOp>(
            [&](auto op) { scale = op.getSrc(); })
        .Case<arith::TruncFOp>([&](auto op) { truncOp = op, scale = nullptr; })
        .Default([&](auto op) { scale = nullptr; });
  }
  if (!truncOp)
    return {};
  auto inType = dyn_cast<RankedTensorType>(truncOp.getIn().getType());
  if (!inType || !inType.getElementType().isF32())
    return {};
  return cast<TypedValue<RankedTensorType>>(truncOp.getIn());
}

class UseScaledUpcastFp4 : public OpRewritePattern<arith::MulFOp> {
public:
  UseScaledUpcastFp4(MLIRContext *context, const AMD::TargetInfo &targetInfo,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                PatternRewriter &rewriter) const override {
    if (targetInfo.getISAFamily() != AMD::ISAFamily::CDNA4)
      return failure();

    auto fp4ToFpOp = mulOp.getLhs().getDefiningOp<ttg::Fp4ToFpOp>();
    if (!fp4ToFpOp)
      return failure();

    auto scale = getUpcastFp32Scale(mulOp);
    if (!scale)
      return failure();

    rewriter.replaceOpWithNewOp<amdgpu::ScaledUpcastFp4Op>(
        mulOp, mulOp.getType(), fp4ToFpOp.getSrc(), scale, fp4ToFpOp.getAxis());
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class UseScaledUpcastFp8 : public OpRewritePattern<arith::MulFOp> {
public:
  UseScaledUpcastFp8(MLIRContext *context, const AMD::TargetInfo &targetInfo,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                PatternRewriter &rewriter) const override {
    if (targetInfo.getISAFamily() != AMD::ISAFamily::CDNA4)
      return failure();

    auto fpToFpOp = mulOp.getLhs().getDefiningOp<FpToFpOp>();
    if (!fpToFpOp)
      return failure();

    auto scale = getUpcastFp32Scale(mulOp);
    if (!scale)
      return failure();

    rewriter.replaceOpWithNewOp<amdgpu::ScaledUpcastFp8Op>(
        mulOp, mulOp.getType(), fpToFpOp.getSrc(), scale);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

} // namespace

namespace mlir::triton::AMD {

void populatePeepholeOptimizationPatterns(RewritePatternSet &patterns,
                                          const AMD::TargetInfo &targetInfo,
                                          PatternBenefit benefit) {
  patterns.add<UseScaledUpcastFp4, UseScaledUpcastFp8>(patterns.getContext(),
                                                       targetInfo, benefit);
}

} // namespace mlir::triton::AMD
