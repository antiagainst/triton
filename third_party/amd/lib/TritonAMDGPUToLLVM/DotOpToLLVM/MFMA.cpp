/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "../PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::LinearEncodingAttr;

using ValueTable = std::map<std::array<int, 3>, Value>;

/// Get matrix format flag passed through BLGP/CBSZ args in V_MFMA_*_F8F6F4
/// instructions.
///
/// Values:
/// - 0: E4M3(FP8)
/// - 1: E5M2(BF8)
/// - 2: E2M3(FP6)
/// - 3: E3M2(BF6)
/// - 4: E2M1(FP4)
static inline std::optional<ROCDL::MatrixFormat>
getMfmaF8F6F4MatrixFormat(Type t) {
  return llvm::TypeSwitch<Type, std::optional<ROCDL::MatrixFormat>>(t)
      .Case<Float8E4M3FNType>(
          [](Type) { return ROCDL::MatrixFormat::fp8_e4m3; })
      .Case<Float8E5M2Type>([](Type) { return ROCDL::MatrixFormat::fp8_e5m2; })
      .Case<Float6E3M2FNType>(
          [](Type) { return ROCDL::MatrixFormat::fp6_e3m2; })
      .Case<Float6E2M3FNType>(
          [](Type) { return ROCDL::MatrixFormat::fp6_e2m3; })
      .Case<Float4E2M1FNType>(
          [](Type) { return ROCDL::MatrixFormat::fp4_e2m1; })
      .Default([](Type) { return std::nullopt; });
}

// Name of the attribute set by the `cd_regclass` argument of Gluon's AMD
// `mfma` / `mfma_scaled`.
constexpr llvm::StringLiteral kCDRegClassAttrName = "amdg.cd_regclass";

// Register classes to pin an MFMA tile's accumulator input (C) and result
// (D) to: "a" = AGPRs, "v" = VGPRs, empty = no pin.
struct CDPins {
  StringRef c;
  StringRef d;
};

// The pins requested by `kCDRegClassAttrName` on `op`, whose accumulator
// input is `c`. A constant accumulator (e.g. zeros) is not pinned on C:
// LLVM folds it into the MFMA as an immediate, which a pin would prevent.
FailureOr<CDPins> getCDPins(Operation *op, Value c) {
  Attribute attr = op->getAttr(kCDRegClassAttrName);
  if (!attr)
    return CDPins{};
  auto regClass = dyn_cast<StringAttr>(attr);
  if (!regClass || (regClass.getValue() != "a" && regClass.getValue() != "v")) {
    op->emitError() << kCDRegClassAttrName << " must be \"a\" or \"v\", got "
                    << attr;
    return failure();
  }
  StringRef cls = regClass.getValue();
  return CDPins{matchPattern(c, m_Constant()) ? StringRef() : cls, cls};
}

// Emits the LLVM ops that the MFMA dot lowerings are built from.
struct MFMAEmitter {
  ConversionPatternRewriter &rewriter;
  const LLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx;

  MFMAEmitter(ConversionPatternRewriter &rewriter,
              const LLVMTypeConverter *typeConverter, Location loc)
      : rewriter(rewriter), typeConverter(typeConverter), loc(loc),
        ctx(rewriter.getContext()) {}

  Value generateMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                       Value valC, int cbsz = 0, int abid = 0,
                       int blgp = 0) const {
    assert(cbsz >= 0 && cbsz <= 4);
    assert(abid >= 0 && abid <= 15);
    assert(blgp >= 0 && blgp <= 7);

    auto resType = valC.getType();
    OperationState loweredOp(loc, intrinsicName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC});
    loweredOp.addAttribute("cbsz", rewriter.getI32IntegerAttr(cbsz));
    loweredOp.addAttribute("abid", rewriter.getI32IntegerAttr(abid));
    // For `blgp`: f64 MFMA uses negation flags, while other MFMA ops use B-lane
    // permutation flags.
    auto vecTy = cast<VectorType>(resType);
    if (vecTy.getElementType().isF64()) {
      loweredOp.addAttribute(
          "blgp", ROCDL::MFMANegModifierAttr::get(
                      ctx, static_cast<ROCDL::MFMANegModifier>(blgp)));
    } else {
      loweredOp.addAttribute(
          "blgp",
          ROCDL::MFMAPermBAttr::get(ctx, static_cast<ROCDL::MFMAPermB>(blgp)));
    }
    return rewriter.create(loweredOp)->getResult(0);
  }

  Value generateScaledMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                             Value valC, Value valScaleA, Value valScaleB,
                             Type elemTypeA, Type elemTypeB, int opSelA,
                             int opSelB) const {
    auto resType = valC.getType();
    OperationState loweredOp(loc, intrinsicName);
    auto cbsz = getMfmaF8F6F4MatrixFormat(elemTypeA);
    auto blgp = getMfmaF8F6F4MatrixFormat(elemTypeB);
    assert(cbsz && blgp);
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, valScaleA, valScaleB});
    loweredOp.addAttribute("cbsz", ROCDL::MatrixFormatAttr::get(ctx, *cbsz));
    loweredOp.addAttribute("blgp", ROCDL::MatrixFormatAttr::get(ctx, *blgp));
    loweredOp.addAttribute("opselA", rewriter.getI32IntegerAttr(opSelA));
    loweredOp.addAttribute("opselB", rewriter.getI32IntegerAttr(opSelB));
    return rewriter.create(loweredOp)->getResult(0);
  }

  // Wraps `acc` in an empty inline asm whose output is tied to its input
  // (constraint "=<regClass>,0"). No instruction is emitted, but the value has
  // to sit in the given register class ("a" = AGPRs, "v" = VGPRs) at this
  // point, which steers the MFMA accumulator into that register file. Returns
  // `acc` unchanged when `regClass` is empty.
  Value pinRegClass(Value acc, StringRef regClass) const {
    if (regClass.empty())
      return acc;
    std::string constraints = ("=" + regClass + ",0").str();
    return LLVM::InlineAsmOp::create(
               rewriter, loc, acc.getType(), ValueRange{acc},
               /*asm_string=*/"", constraints, /*has_side_effects=*/false,
               /*is_align_stack=*/false, LLVM::TailCallKind::None,
               LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
               /*operand_attrs=*/ArrayAttr())
        ->getResult(0);
  }

  /// Dot operand layout minimal tile is kDimInstrSize elements across
  /// K dimension. If dot operand K dimension is smaller, layout
  /// assigns tensor elements to multiple different hardware locations.
  /// In this case mfma instruction adds elements in accumulator
  /// multiple times.
  ///
  /// Let say A=[1,2]; B=[3,4], C = A*B = 1*3+2*4 = 11
  /// Consider instruction K size is 4,
  /// in this case operands will be duplicated:
  /// A' = [1,2,1,2] B' = [3,4,3,4]
  /// C' = (1*3+2*4) + (1*3+2*4) = 22
  ///
  /// Following code adjusts accumulator values in such cases.
  /// If accumulator is integer, shift accumulator right by
  /// log2(duplicationRate). If accumulator is float, multiply accum
  /// with 1/duplicationRate constant.
  void adjustAccForSmallKDim(SmallVector<Value> &fc, Value &acc, Type dstElemTy,
                             int b, int m, int n, int64_t numRepM,
                             int64_t numRepN, int64_t kDimInstrSize,
                             int64_t kDimOperandSize,
                             unsigned elemsPerVec) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    for (unsigned v = 0; v < elemsPerVec; ++v) {
      Value accElem = tb.extract_element(dstElemTy, acc, tb.i32_val(v));
      if (kDimInstrSize > kDimOperandSize) {
        assert(kDimInstrSize % kDimOperandSize == 0);
        int duplicationRate = kDimInstrSize / kDimOperandSize;
        assert(llvm::isPowerOf2_32(duplicationRate));
        if (dstElemTy.isInteger()) {
          auto shiftSize = llvm::Log2_32(duplicationRate);
          assert(!accElem.getType().isUnsignedInteger() &&
                 "MFMA uses signed accumulator");
          accElem = tb.ashr(accElem, tb.i32_val(shiftSize));
        } else {
          auto multiplierAttr =
              rewriter.getFloatAttr(dstElemTy, 1.0 / duplicationRate);
          auto multiplierVal = LLVM::ConstantOp::create(
              rewriter, loc, dstElemTy, multiplierAttr);
          accElem = tb.fmul(accElem, multiplierVal);
        }
      }
      auto linearIdx = b * numRepM * numRepN * elemsPerVec +
                       m * numRepN * elemsPerVec + n * elemsPerVec + v;
      fc[linearIdx] = accElem;
    }
  }

  /// Process the elements in rawElems and prepare a vector for mfma input.
  /// rawElems is a vector of kBase elements. Each element is of the raw
  /// element type from the input. We need to prepare a vector of kBase
  /// elements of appropriate element type required by mfma instructions.
  Value prepareOperands(Value rawElems, int kBase, Type type,
                        bool preserveBF16) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value results;

    // Construct a vector type of kBase elements with desired type
    auto vecTy = vec_ty(type, kBase);
    if (type.isBF16() && !preserveBF16)
      vecTy = vec_ty(i16_ty, kBase);
    Value vec = b.undef(vecTy);

    // For each element in rawElems, extract the element as the desired type,
    // bitcast it if needed, and insert it into vec.
    for (int elemId = 0; elemId < kBase; ++elemId) {
      auto val = b.extract_element(type, rawElems, b.i32_val(elemId));
      if (type.isBF16() && !preserveBF16) {
        // rocdl.mfma.f32.32x32x8bf16.1k calls for input of i16 type
        auto cast = b.bitcast(val, i16_ty);
        vec = b.insert_element(vecTy, vec, cast, b.i32_val(elemId));
      } else {
        vec = b.insert_element(vecTy, vec, val, b.i32_val(elemId));
      }
    }

    // Now we have a vector of kBase elements of desired type.
    // Then we need to prepare vec for results.
    if (type.getIntOrFloatBitWidth() == 8) {
      if (1 == kBase)
        // This is only for the scale operands of scaled mfma on CDNA4
        results = b.zext(i32_ty, b.bitcast(vec, i8_ty));
      if (2 == kBase)
        // This case can occur during scale tensor packing when there aren't
        // enough elements to fill all 4 opSel slots. For example, with an A
        // tensor of size 16x256 and using 16x16x128 block sizes, we end up with
        // only 2 elements to pack,  resulting in a kBase of 2.
        results = b.zext(i32_ty, b.bitcast(vec, i16_ty));
      if (4 == kBase)
        // This is for int8 on pre- CDNA3 GPUs and scale tensors on CDNA4 GPUs
        results = b.bitcast(vec, i32_ty);
      if (8 == kBase)
        results = b.bitcast(vec, i64_ty);
      if (16 == kBase)
        // This is only for the operands of scaled mfma on CDNA4
        results = b.bitcast(vec, vec_ty(i32_ty, 4));
      if (32 == kBase)
        results = b.bitcast(vec, vec_ty(i32_ty, 8));
    } else {
      results = vec;
    }
    return results;
  }

  /// Converts dot operand structure to value table and converts types
  /// appropriate for mfma instructions
  ValueTable getValuesFromDotOperandLayoutStruct(Value value, int batch,
                                                 int nonKRep, int kRepInKWidth,
                                                 int kWidth, int kBase,
                                                 RankedTensorType tensorType,
                                                 bool allowXF32,
                                                 bool preserveBF16) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    Type type = tensorType.getElementType();
    auto elems = unpackTensorElements(loc, value, rewriter, tensorType);
    if (type.isF32() && allowXF32) {
      // Quiet NaNs before XF32 discards their low 13 mantissa bits.
      for (Value &elem : elems)
        elem = LLVM::createLLVMIntrinsicCallOp(
                   rewriter, loc, "llvm.canonicalize", elem.getType(), elem)
                   ->getResult(0);
    }
    // number of kBase-element vectors
    int numVecInKBase = kRepInKWidth * kWidth / kBase;
    if (numVecInKBase == 0) {
      numVecInKBase = 1;
      nonKRep /= kBase / (kRepInKWidth * kWidth);
      assert(nonKRep > 0 && "nonKrep too small");
    }
    ValueTable dotOpVals;

    SmallVector<int64_t> strides =
        computeStrides({batch, nonKRep, numVecInKBase, kBase});
    for (int b = 0; b < batch; ++b) {
      for (int nonK = 0; nonK < nonKRep; nonK++) {
        for (int kBaseVec = 0; kBaseVec < numVecInKBase; kBaseVec++) {
          // For each kBase-element vector

          // Step 1: construct each kBase-element vector by
          //         - extracting kBase elements from elems and
          //         - putting them into a kBase-element vector, i.e. rawElems
          Type elemTy = typeConverter->convertType(type);
          Type ty = vec_ty(elemTy, kBase);
          Value rawElems = tb.undef(ty);
          for (int k = 0; k < kBase; ++k) {
            auto index = linearize({b, nonK, kBaseVec, k}, strides);
            rawElems =
                tb.insert_element(ty, rawElems, elems[index], tb.i32_val(k));
          }

          // Step 2: process rawElems based on element type
          // Note that for f32 and XF32 is not allowed or f64, nothing needs
          // to be done and rawElems is inserted into the ValueTable directly
          if ((type.isF32() && !allowXF32) || type.isF64()) {
            dotOpVals[{b, nonK, kBaseVec}] =
                tb.extract_element(type, rawElems, tb.i32_val(0));
          } else {
            Value vals;
            if (type.isF32() && allowXF32) {
              vals = prepareOperands(rawElems, kBase, f32_ty, preserveBF16);
            } else if (elemTy.getIntOrFloatBitWidth() == 8) {
              vals = prepareOperands(rawElems, kBase, i8_ty, preserveBF16);
            } else if (type.isBF16()) {
              vals = prepareOperands(rawElems, kBase, bf16_ty, preserveBF16);
            } else {
              assert(type.isF16() && "Unsupported data type");
              vals = prepareOperands(rawElems, kBase, f16_ty, preserveBF16);
            }

            // Step 3: Insert the processed vals into the ValueTable
            dotOpVals[{b, nonK, kBaseVec}] = vals;
          }
        }
      }
    }
    return dotOpVals;
  }
};

} // namespace

namespace mlir::triton::AMD {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto aTensorTy = cast<RankedTensorType>(op.getA().getType());
  auto bTensorTy = cast<RankedTensorType>(op.getB().getType());
  auto dTensorTy = cast<RankedTensorType>(op.getD().getType());
  assert(isa<DotOperandEncodingAttr>(aTensorTy.getEncoding()) &&
         isa<DotOperandEncodingAttr>(bTensorTy.getEncoding()) &&
         "Both A and B should be DotOperand layout.");
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(dTensorTy.getEncoding());

  auto loc = op.getLoc();
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  MFMAEmitter emitter(rewriter, typeConverter, loc);
  // Check if this dot has come with priority set by setprio.
  auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

  auto mnkDim = mfmaLayout.getInstrShape();
  auto mDim = mnkDim[0];
  auto nDim = mnkDim[1];
  auto kDim = mnkDim[2];
  auto mfmaVersion = mfmaLayout.getVersion();

  FailureOr<CDPins> cdPins = getCDPins(op.getOperation(), op.getC());
  if (failed(cdPins))
    return failure();

  auto elemTyA = aTensorTy.getElementType();
  auto elemTyB = bTensorTy.getElementType();

  const auto kDimOperandSize = aTensorTy.getShape().back();

  bool allowXF32 =
      op.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
  // When mfmaLayout is transposed, operands will be swapped later (op1, op2
  // swap). For asymmetric types (e.g., bf8_fp8), we need to select the
  // intrinsic with swapped types so the hardware interprets the data
  // correctly after the swap.
  Type intrinsicElemTyA = elemTyA;
  Type intrinsicElemTyB = elemTyB;
  if (mfmaLayout.getIsTransposed() && elemTyA != elemTyB) {
    std::swap(intrinsicElemTyA, intrinsicElemTyB);
  }
  FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic =
      MfmaIntrinsic::get(loc, mfmaVersion, mDim, nDim, kDim, intrinsicElemTyA,
                         intrinsicElemTyB, /*withScale=*/false, allowXF32);
  if (failed(maybeMfmaIntrinsic))
    return op.emitError("no matching matrix core intrinsic ")
           << "for mfma version " << mfmaVersion << " with instruction shape ["
           << mDim << ", " << nDim << ", " << kDim
           << "] and element types A=" << elemTyA << ", B=" << elemTyB
           << ". Check whether the mfma version,"
           << " instruction shape, and data types "
           << "are supported on the current AMD GPU architecture.";

  StringRef intrinsicName = maybeMfmaIntrinsic->name;
  unsigned kBase = maybeMfmaIntrinsic->kBase;

  auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  int kWidth = aEncoding.getKWidth();

  const auto kDimInstrSize = mfmaLayout.getInstrShapeForOperand(kWidth, 0)[1];

  auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
  auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);

  assert(repA[2] == repB[1]);

  Value loadedA = adaptor.getA();
  Value loadedB = adaptor.getB();
  Value loadedC = adaptor.getC();

  auto numRepM = repA[1];
  auto numRepN = repB[2];
  auto numRepK = repA[2];
  auto numRepB = repA[0];
  assert(repA[0] == repB[0]);

  int numBroadcastA = 1;
  int numBroadcastB = 1;
  int numRepKA = numRepK;
  int numRepKB = numRepK;
  if ((mDim == 64 && nDim == 4)) {
    numBroadcastB = 16;
    numRepKA *= 16;
  }
  if ((mDim == 4 && nDim == 64)) {
    numBroadcastA = 16;
    numRepKB *= 16;
  }
  numRepK = std::max(numRepKA, numRepKB);

  bool preserveBF16 = intrinsicName.contains(".bf16") && mfmaVersion >= 4;
  auto operandA = emitter.getValuesFromDotOperandLayoutStruct(
      loadedA, numRepB, numRepM, numRepKA, kWidth, kBase, aTensorTy, allowXF32,
      preserveBF16);
  auto operandB = emitter.getValuesFromDotOperandLayoutStruct(
      loadedB, numRepB, numRepN, numRepKB, kWidth, kBase, bTensorTy, allowXF32,
      preserveBF16);

  int warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
  int elemsPerVec = mDim * nDim / warpSize;
  int numVecInKBase = numRepK * kWidth / kBase;

  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackTensorElements(loc, loadedC, rewriter, op.getC().getType());
  SmallVector<int64_t> fcStrides =
      computeStrides({numRepB, numRepM, numRepN, elemsPerVec});

  Value firstMfma;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);
  for (int b = 0; b < numRepB; ++b) {
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        Value acc = tb.undef(vecTy);

        for (int v = 0; v < elemsPerVec; ++v) {
          int linearIdx = linearize({b, m, n, v}, fcStrides);
          Value c = fc[linearIdx];
          acc = tb.insert_element(vecTy, acc, c, tb.i32_val(v));
        }
        // Pin C right before the tile's first MFMA.
        acc = emitter.pinRegClass(acc, cdPins->c);

        for (int k = 0; k < numVecInKBase; ++k) {
          Value op1 = operandA[{b, m, k}];
          Value op2 = operandB[{b, n, k}];
          int cbsz = 0;
          int abid = 0;

          if (numBroadcastA > 1) {
            assert(!mfmaLayout.getIsTransposed());
            cbsz = llvm::Log2_32(numBroadcastA);
            abid = k % numBroadcastA;
            op1 = operandA[{b, m, k / numBroadcastA}];
          }

          if (numBroadcastB > 1) {
            assert(numBroadcastA == 1);
            assert(mfmaLayout.getIsTransposed());
            cbsz = llvm::Log2_32(numBroadcastB);
            abid = k % numBroadcastB;
            op2 = operandB[{b, n, k / numBroadcastB}];
          }

          if (mfmaLayout.getIsTransposed())
            std::swap(op1, op2);

          acc =
              emitter.generateMFMAOp(intrinsicName, op1, op2, acc, cbsz, abid);

          if (!firstMfma)
            firstMfma = acc;
        }
        // Pin D right after the tile's last MFMA.
        acc = emitter.pinRegClass(acc, cdPins->d);

        emitter.adjustAccForSmallKDim(fc, acc, dstElemTy, b, m, n, numRepM,
                                      numRepN, kDimInstrSize, kDimOperandSize,
                                      elemsPerVec);
      }
    }
  }

  // Originally, setprio (high) is set to the high-level dot op. After dot is
  // being lowered to the series of mfma operations, it should be moved next
  // to the first mfma leaving the first mfma staying at the low priority. In
  // this way, incoming warp can be effectively waiting on the first mfma
  // instruction (low priority) while the other warp is executing mfma with
  // high priority. Otherwise, incoming warp can break the cluster.
  if (setPrioOp && firstMfma)
    setPrioOp->moveAfter(firstMfma.getDefiningOp());

  Value res =
      packTensorElements(loc, typeConverter, fc, rewriter, op.getType());
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) {
  auto aTensorTy = op.getA().getType();
  auto bTensorTy = op.getB().getType();
  auto dTensorTy = op.getD().getType();
  assert(isa<DotOperandEncodingAttr>(aTensorTy.getEncoding()) &&
         isa<DotOperandEncodingAttr>(bTensorTy.getEncoding()) &&
         "Both lhs and rhs should be in DotOperand layout.");
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(dTensorTy.getEncoding());

  auto aScale = op.getAScale();
  auto bScale = op.getBScale();

  // If the tt.dot_scaled is transformed from a tt.dot, both scales are None. In
  // this case, both scales remain None in this method and we will generate a
  // mfma instruction with the scale operand to be 0. Then there's an
  // optimization pass in the LLVM backend to convert such V_MFMA_SCALE_*_F8F6F4
  // instruction to V_MFMA_*_F8F6F4 to avoid LD_SCALE.
  //
  // If the tt.dot_scaled is not from a tt.dot but native, we support 0, 1, 2
  // scales and treat them in different ways:
  //
  // 1. #scales = 0: Just like those transformed from tt.dot, both scales remain
  // None.
  // 2. #scales = 1: The upstream transform guarantees to create constant
  // scales for the absent.
  // 2. #scales = 2: Both scales should exist.

  // Thus in this pass, there shouldn't be a single scale present.
  if ((aScale && !bScale) || (!aScale && bScale)) {
    llvm::report_fatal_error("Single scale is not supported\n");
  }

  bool existBothScales = aScale && bScale;
  assert(
      (!existBothScales ||
       (isa<LinearEncodingAttr>(aScale.getType().getEncoding()) &&
        isa<LinearEncodingAttr>(bScale.getType().getEncoding()))) &&
      "If scales exist, both LhsScale and RhsScale should be linear layout.");

  auto loc = op.getLoc();
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  MFMAEmitter emitter(rewriter, typeConverter, loc);
  // Check if this dot has come with priority set by setprio.
  auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

  auto mnkDim = mfmaLayout.getInstrShape();
  auto mDim = mnkDim[0];
  auto nDim = mnkDim[1];
  auto kDim = mnkDim[2];
  auto mfmaVersion = mfmaLayout.getVersion();

  FailureOr<CDPins> cdPins = getCDPins(op.getOperation(), op.getC());
  if (failed(cdPins))
    return failure();

  bool isAScaleConstant =
      aScale && isa<arith::ConstantOp, triton::SplatOp>(aScale.getDefiningOp());
  bool isBScaleConstant =
      bScale && isa<arith::ConstantOp, triton::SplatOp>(bScale.getDefiningOp());
  auto elemTyA = aTensorTy.getElementType();
  auto elemTyB = bTensorTy.getElementType();
  ScaleDotElemType aElemType = op.getAElemType();
  ScaleDotElemType bElemType = op.getBElemType();

  auto supportsTypes = [](ScaleDotElemType elemType) {
    return elemType == ScaleDotElemType::E2M1 ||
           elemType == ScaleDotElemType::E4M3 ||
           elemType == ScaleDotElemType::E5M2;
  };

  if (!supportsTypes(aElemType) || !supportsTypes(bElemType)) {
    llvm::report_fatal_error("NYI: mxfp6\n");
  }

  int64_t kDimOperandSize = aTensorTy.getShape().back();

  auto ctx = op.getContext();
  constexpr bool allowXF32 = false;
  FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic =
      MfmaIntrinsic::get(loc, mfmaVersion, mDim, nDim, kDim,
                         scaleDotElemTypeToMLIRType(ctx, aElemType),
                         scaleDotElemTypeToMLIRType(ctx, bElemType),
                         /*withScale=*/true, allowXF32);
  if (failed(maybeMfmaIntrinsic))
    return op.emitError("no matching matrix core intrinsic ")
           << "for mfma version " << mfmaVersion << " with instruction shape ["
           << mDim << ", " << nDim << ", " << kDim
           << "] and element types A=" << elemTyA << ", B=" << elemTyB
           << ". Check whether the mfma version,"
           << " instruction shape, and data types "
           << "are supported on the current AMD GPU architecture.";

  StringRef intrinsicName = maybeMfmaIntrinsic->name;
  unsigned kBase = maybeMfmaIntrinsic->kBase;
  // Two fp4 are packed into an uint8.
  unsigned aKBase = aElemType == ScaleDotElemType::E2M1 ? kBase / 2 : kBase;
  unsigned bKBase = bElemType == ScaleDotElemType::E2M1 ? kBase / 2 : kBase;

  int aKWidth = aKBase;
  int bKWidth = bKBase;

  const auto kDimInstrSize = mfmaLayout.getInstrShapeForOperand(aKBase, 0)[1];

  auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), aKWidth, 0);
  auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), bKWidth, 1);
  assert(repA[2] == repB[1]);

  // For fp4 scaled mfma, each thread takes 1 element from scale. Will have
  // better way to get it when adapting other data types. Similar to
  // scaleKBase
  constexpr int scaleKWidth = 1;
  Value loadedA = adaptor.getA();
  Value loadedB = adaptor.getB();
  Value loadedAScale = adaptor.getAScale();
  Value loadedBScale = adaptor.getBScale();
  Value loadedC = adaptor.getC();

  auto numRepM = repA[1];
  auto numRepN = repB[2];
  auto numRepK = repA[2];
  auto numRepB = repA[0];
  assert(repA[0] == repB[0]);

  // Scaled MFMA instructions expect scale operands as 32-bit values,
  // even though each individual scale is only 8 bits. To reduce register
  // usage, we pack 4 scales into a single 32-bit value and use the opSel
  // field to select the appropriate byte during execution. Packing is done
  // along the K dimension first; if there aren’t enough values in K, we
  // continue along the non-K dimension.
  // TODO: Support opSel selection for constant scales stored in SGPRs.
  const int scaleAKBase =
      isAScaleConstant ? 1 : std::min(4, static_cast<int>(numRepK * numRepM));
  const int scaleBKBase =
      isBScaleConstant ? 1 : std::min(4, static_cast<int>(numRepK * numRepN));

  int akPackedVals =
      isAScaleConstant ? 1 : std::min(4, static_cast<int>(numRepK));
  int bkPackedVals =
      isBScaleConstant ? 1 : std::min(4, static_cast<int>(numRepK));

  assert(scaleAKBase % akPackedVals == 0 && scaleBKBase % bkPackedVals == 0);
  int aNonKPackedVals = scaleAKBase / akPackedVals;
  int bNonKPackedVals = scaleBKBase / bkPackedVals;

  auto operandA = emitter.getValuesFromDotOperandLayoutStruct(
      loadedA, numRepB, numRepM, numRepK, aKWidth, aKBase, aTensorTy, allowXF32,
      /*preserveBF16=*/false);
  auto operandB = emitter.getValuesFromDotOperandLayoutStruct(
      loadedB, numRepB, numRepN, numRepK, bKWidth, bKBase, bTensorTy, allowXF32,
      /*preserveBF16=*/false);

  // Scales have the same replica distributions as their corresponding
  // operands.
  ValueTable operandAScale;
  ValueTable operandBScale;
  if (existBothScales) {
    operandAScale = emitter.getValuesFromDotOperandLayoutStruct(
        loadedAScale, numRepB, numRepM, numRepK, scaleKWidth, scaleAKBase,
        aScale.getType(), allowXF32, /*preserveBF16=*/false);
    operandBScale = emitter.getValuesFromDotOperandLayoutStruct(
        loadedBScale, numRepB, numRepN, numRepK, scaleKWidth, scaleBKBase,
        bScale.getType(), allowXF32, /*preserveBF16=*/false);
  }

  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackTensorElements(loc, loadedC, rewriter, op.getC().getType());

  unsigned warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
  // Number of output elements that each thread holds for one MFMA instruction.
  auto elemsPerVec = mDim * nDim / warpSize;
  int numVecInKBase = numRepK * aKWidth / aKBase;

  Value firstMfma;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);

  // 2-step pingpong got local_loads + dot_scaled in the dot cluster
  // from the first step in the transform pingpong pass.
  // Here, in the second step, it splits operations into two clusters
  // The first cluster has local_load with mfma from the first half of K
  // and the second cluster with the other half K of mfma.
  // By splitting in K dim, we can retire registers used by the
  // first half of mfma, backend compiler is supposed to schedule it.
  int halfPoint = numVecInKBase * numRepB * numRepM * numRepN / 2;
  int currIter = 0;
  bool is2Step = false;
  int innerK = 0, outerK = 0, innerKBound = 1, outerKBound = 1;
  // In order to split mfma by K, change the outermost loop iterates
  // over the K in emitting the mfma operations.
  if (auto pingpongUnitAttr = op->getAttr("pingpong_2step")) {
    is2Step = true;
    outerKBound = numVecInKBase;
  } else
    innerKBound = numVecInKBase;

  for (outerK = 0; outerK < outerKBound; outerK++) {
    for (int b = 0; b < numRepB; ++b) {
      for (int m = 0; m < numRepM; ++m) {
        for (int n = 0; n < numRepN; ++n) {
          // Insert pingpong cluster barrier when needed.
          if (is2Step && currIter++ == halfPoint) {
            ROCDL::SchedBarrier::create(rewriter, loc,
                                        ROCDL::SchedGroupMask::none);
            ROCDL::SBarrierOp::create(rewriter, loc);
            ROCDL::SchedBarrier::create(rewriter, loc,
                                        ROCDL::SchedGroupMask::none);
          }
          Value acc = tb.undef(vecTy);
          for (unsigned v = 0; v < elemsPerVec; ++v) {
            acc = tb.insert_element(
                vecTy, acc,
                fc[b * numRepM * numRepN * elemsPerVec +
                   m * numRepN * elemsPerVec + n * elemsPerVec + v],
                tb.i32_val(v));
          }
          // Pin C right before this pass's MFMAs (all K steps, or one K step
          // with pingpong_2step). Only the first pass reads the op's C; later
          // passes read the previous pass's result.
          acc = emitter.pinRegClass(acc, outerK == 0 ? cdPins->c : cdPins->d);
          for (innerK = 0; innerK < innerKBound; innerK++) {
            int k = is2Step ? outerK : innerK;
            Value valA = operandA[{b, m, k}];
            Value valB = operandB[{b, n, k}];
            Value scaleA, scaleB;
            int opSelA = 0, opSelB = 0;
            if (existBothScales) {
              scaleA =
                  operandAScale[{b, m / aNonKPackedVals, k / akPackedVals}];
              scaleB =
                  operandBScale[{b, n / bNonKPackedVals, k / bkPackedVals}];
              opSelA = (m * numRepK + k) % (aNonKPackedVals * akPackedVals);
              opSelB = (n * numRepK + k) % (bNonKPackedVals * bkPackedVals);
            } else {
              // If both scales are constant 0, the LLVM backend will use
              // V_MFMA_*_F8F6F4 instructions instead of V_MFMA_SCALE_*_F8F6F4
              // to reduce memory access.
              scaleA = scaleB = tb.i32_val(0);
            }
            Type elemTypeA = maybeMfmaIntrinsic->aElementType;
            Type elemTypeB = maybeMfmaIntrinsic->bElementType;
            if (mfmaLayout.getIsTransposed()) {
              std::swap(valA, valB);
              std::swap(scaleA, scaleB);
              std::swap(elemTypeA, elemTypeB);
              std::swap(opSelA, opSelB);
            }
            acc = emitter.generateScaledMFMAOp(intrinsicName, valA, valB, acc,
                                               scaleA, scaleB, elemTypeA,
                                               elemTypeB, opSelA, opSelB);
            if (!firstMfma)
              firstMfma = acc;
          }
          // Pin D right after this pass's MFMAs.
          acc = emitter.pinRegClass(acc, cdPins->d);
          emitter.adjustAccForSmallKDim(fc, acc, dstElemTy, b, m, n, numRepM,
                                        numRepN, kDimInstrSize, kDimOperandSize,
                                        elemsPerVec);
        }
      }
    }
  }

  // Originally, setprio (high) is set to the high-level dot op. After dot is
  // being lowered to the series of mfma operations, it should be moved next
  // to the first mfma leaving the first mfma staying at the low priority. In
  // this way, incoming warp can be effectively waiting on the first mfma
  // instruction (low priority) while the other warp is executing mfma with
  // high priority. Otherwise, incoming warp can break the cluster.
  if (setPrioOp && firstMfma)
    setPrioOp->moveAfter(firstMfma.getDefiningOp());

  Value res =
      packTensorElements(loc, typeConverter, fc, rewriter, op.getType());
  rewriter.replaceOp(op, res);
  return success();
}
} // namespace mlir::triton::AMD
