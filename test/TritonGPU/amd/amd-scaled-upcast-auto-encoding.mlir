// RUN: triton-opt %s -split-input-file --gluon-resolve-auto-encodings | FileCheck %s

// Resolve gluon auto encodings across AMD scaled_upcast ops. These ops
// implement the UpcastFpOpInterface, so the resolver must use the per-operand
// interface methods rather than the index-agnostic infer*Encoding helpers
// (which intentionally bail out on UpcastFpOpInterface ops).

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: [[$BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @upcast_fp8_resolve_backward
  tt.func public @upcast_fp8_resolve_backward(%fs: f8E4M3FN, %ss: i8) -> tensor<16x64xbf16, #blocked> {
    // CHECK-NOT: auto_encoding
    %in = tt.splat %fs : f8E4M3FN -> tensor<16x64xf8E4M3FN, #gluon.auto_encoding>
    %scale = tt.splat %ss : i8 -> tensor<16x64xi8, #gluon.auto_encoding>
    // fp8 upcast is encoding-preserving, so anchoring the result back-propagates
    // the same layout to both the input and the scale operand.
    // CHECK: amdg.scaled_upcast_fp8 {{.*}} : tensor<16x64xf8E4M3FN, [[$BLOCKED]]>, tensor<16x64xi8, [[$BLOCKED]]> -> tensor<16x64xbf16, [[$BLOCKED]]>
    %0 = amdg.scaled_upcast_fp8 %in scale %scale : tensor<16x64xf8E4M3FN, #gluon.auto_encoding>, tensor<16x64xi8, #gluon.auto_encoding> -> tensor<16x64xbf16, #gluon.auto_encoding>
    %1 = gluon.set_auto_layout %0 : tensor<16x64xbf16, #gluon.auto_encoding> -> tensor<16x64xbf16, #blocked>
    tt.return %1 : tensor<16x64xbf16, #blocked>
  }
}

// -----

// Forward propagation: seed the operands and resolve the (dead) upcast result
// purely by forward inference through the op interface.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: [[$BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @upcast_fp8_resolve_forward
  tt.func public @upcast_fp8_resolve_forward(%fs: f8E4M3FN, %ss: i8) -> (tensor<16x64xf8E4M3FN, #blocked>, tensor<16x64xi8, #blocked>) {
    // CHECK-NOT: auto_encoding
    %in = tt.splat %fs : f8E4M3FN -> tensor<16x64xf8E4M3FN, #gluon.auto_encoding>
    %inc = gluon.set_auto_layout %in : tensor<16x64xf8E4M3FN, #gluon.auto_encoding> -> tensor<16x64xf8E4M3FN, #blocked>
    %scale = tt.splat %ss : i8 -> tensor<16x64xi8, #gluon.auto_encoding>
    %scalec = gluon.set_auto_layout %scale : tensor<16x64xi8, #gluon.auto_encoding> -> tensor<16x64xi8, #blocked>
    // The result encoding is inferred forward from the seeded operands.
    // CHECK: amdg.scaled_upcast_fp8 {{.*}} -> tensor<16x64xbf16, [[$BLOCKED]]>
    %0 = amdg.scaled_upcast_fp8 %in scale %scale : tensor<16x64xf8E4M3FN, #gluon.auto_encoding>, tensor<16x64xi8, #gluon.auto_encoding> -> tensor<16x64xbf16, #gluon.auto_encoding>
    tt.return %inc, %scalec : tensor<16x64xf8E4M3FN, #blocked>, tensor<16x64xi8, #blocked>
  }
}

// -----

// Packed fp4 upcast: the packed i8 input (axis dim halved) needs a different
// encoding from the scale, which shares the result's encoding. The resolver
// must query each operand index separately.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: [[$BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @upcast_fp4_resolve_backward
  tt.func public @upcast_fp4_resolve_backward(%fs: i8, %ss: i8) -> tensor<16x16xbf16, #blocked> {
    // CHECK-NOT: auto_encoding
    %in = tt.splat %fs : i8 -> tensor<16x8xi8, #gluon.auto_encoding>
    %scale = tt.splat %ss : i8 -> tensor<16x16xi8, #gluon.auto_encoding>
    // Scale (operand 1) takes the result encoding; the packed input (operand 0)
    // is resolved through the fp4 packing transform.
    // CHECK: amdg.scaled_upcast_fp4 {{.*}} scale {{.*}} {axis = 1 : i32} : tensor<16x8xi8, #{{.*}}>, tensor<16x16xi8, [[$BLOCKED]]> -> tensor<16x16xbf16, [[$BLOCKED]]>
    %0 = amdg.scaled_upcast_fp4 %in scale %scale {axis = 1 : i32} : tensor<16x8xi8, #gluon.auto_encoding>, tensor<16x16xi8, #gluon.auto_encoding> -> tensor<16x16xbf16, #gluon.auto_encoding>
    %1 = gluon.set_auto_layout %0 : tensor<16x16xbf16, #gluon.auto_encoding> -> tensor<16x16xbf16, #blocked>
    tt.return %1 : tensor<16x16xbf16, #blocked>
  }
}
