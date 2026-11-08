// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=COMMON,ONE_L0
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=COMMON,ONE_L0
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=COMMON,WGP,WGP11
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=COMMON,WGP,WGP11
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=COMMON,WGP,WGP12
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=COMMON,ONE_L0
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1100 cu-mode=true" --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=COMMON,ONE_L0
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1200 cu-mode=true" --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=COMMON,ONE_L0

// CTA barriers order only the address spaces they name, on every target.
// COMMON-DAG: [[$LOCAL_MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: llvm.func @lower_barrier
  tt.func @lower_barrier() {
    // COMMON: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // COMMON-NEXT: rocdl.s.barrier
    // COMMON-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    ttg.barrier local
    tt.return
  }

  // COMMON-LABEL: llvm.func @lower_warp_barrier
  tt.func @lower_warp_barrier() {
    // COMMON-NOT: rocdl.s.barrier
    // COMMON: llvm.fence syncscope("wavefront") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // COMMON-NEXT: rocdl.wave.barrier
    // COMMON-NEXT: llvm.fence syncscope("wavefront") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // COMMON-NOT: rocdl.s.barrier
    // COMMON: llvm.return
    ttg.barrier warp local
    tt.return
  }
}

// -----

// A scalar atomic runs in one thread, and the barrier around it only orders
// LDS. In WGP mode the waves span two CUs, each with its own L0, so every wave
// waits for its own global stores before a release atomic's barrier and
// invalidates its own L0 after an acquire atomic's barrier. With one L0 per
// workgroup the issuing thread's fences already cover every wave.
// COMMON-DAG: [[$LOCAL_MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">
// WGP-DAG: [[$GLOBAL_MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"global">
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: llvm.func @scalar_atomic_release_cta
  tt.func @scalar_atomic_release_cta(%ptr: !tt.ptr<i32>, %val: i32) {
    // gfx11 uses an agent-scope fence (only vmcnt/vscnt waits there) so that
    // InstCombine does not merge it into the workgroup fence that follows.
    // WGP11: llvm.fence syncscope("agent") release {llvm.mmra = [[$GLOBAL_MMRA_TAG]]}
    // WGP11-NEXT: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // WGP12: rocdl.s.wait.storecnt 0
    // WGP12-NEXT: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // ONE_L0-NOT: llvm.fence
    // ONE_L0-NOT: rocdl.s.wait
    // ONE_L0: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // COMMON-NEXT: rocdl.s.barrier
    // COMMON: llvm.atomicrmw add {{.*}} syncscope("workgroup") release
    %0 = tt.atomic_rmw add, release, cta, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    // COMMON: llvm.return
    tt.return
  }

  // The result is unused, so the barrier after the atomic carries the acquire.
  // COMMON-LABEL: llvm.func @scalar_atomic_acquire_unused
  tt.func @scalar_atomic_acquire_unused(%ptr: !tt.ptr<i32>, %val: i32) {
    // COMMON: llvm.atomicrmw add {{.*}} syncscope("agent") acquire
    // COMMON: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // COMMON-NEXT: rocdl.s.barrier
    // COMMON-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // WGP-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$GLOBAL_MMRA_TAG]]}
    // ONE_L0-NOT: llvm.fence
    %0 = tt.atomic_rmw add, acquire, gpu, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    // COMMON: llvm.return
    tt.return
  }

  // The barrier that broadcasts the result carries the acquire.
  // COMMON-LABEL: llvm.func @scalar_atomic_acquire_broadcast
  tt.func @scalar_atomic_acquire_broadcast(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    // COMMON: llvm.atomicrmw add {{.*}} syncscope("agent") acquire
    // COMMON: llvm.store
    // COMMON: rocdl.s.barrier
    // COMMON-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // WGP-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$GLOBAL_MMRA_TAG]]}
    // ONE_L0-NOT: llvm.fence
    // COMMON: llvm.load
    %0 = tt.atomic_rmw add, acquire, gpu, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // A relaxed result broadcast stays LDS-only.
  // COMMON-LABEL: llvm.func @scalar_atomic_relaxed_broadcast
  tt.func @scalar_atomic_relaxed_broadcast(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    // COMMON: llvm.atomicrmw add {{.*}} monotonic
    // COMMON: rocdl.s.barrier
    // COMMON-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // COMMON-NOT: llvm.fence
    // COMMON: llvm.return
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // Only the polling thread fences on success; the rendezvous carries it.
  // COMMON-LABEL: llvm.func @scalar_atomic_poll_acquire
  tt.func @scalar_atomic_poll_acquire(%ptr: !tt.ptr<i32>, %expected: i32) {
    // COMMON: llvm.fence syncscope("agent") acquire
    // COMMON: rocdl.s.barrier
    // COMMON-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // WGP-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$GLOBAL_MMRA_TAG]]}
    // ONE_L0-NOT: llvm.fence
    %0 = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    // COMMON: llvm.return
    tt.return
  }
}
