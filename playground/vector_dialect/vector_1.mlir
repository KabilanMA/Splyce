// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Element-wise Multiply — Hand-vectorized using MLIR Vector Dialect
//
//  Kernel:  result[i] = ((((a[i]*b[i] + a[i])*b[i] + a[i])*b[i] + a[i])
//                                                    *b[i] + a[i]) * b[i]
//
//  Strategy: Vectorized co-iteration (W = 8 lanes)
//  ─────────────────────────────────────────────────────────────────────────
//  For each iteration of the outer scf.while:
//
//    ┌── Load W coords + values from A window ──────────────────────────────┐
//    │   ca = [c0, c1, c2, c3, c4, c5, c6, c7]  (sorted, unique)           │
//    │   va = [v0, v1, v2, v3, v4, v5, v6, v7]                             │
//    └──────────────────────────────────────────────────────────────────────┘
//    ┌── Load W coords + values from B window ──────────────────────────────┐
//    │   cb = [c0, c1, c2, c3, c4, c5, c6, c7]  (sorted, unique)           │
//    │   vb = [v0, v1, v2, v3, v4, v5, v6, v7]                             │
//    └──────────────────────────────────────────────────────────────────────┘
//
//    ┌── Shuffle-based all-pairs intersection (the key insight) ────────────┐
//    │  For each A lane k:                                                  │
//    │    broadcast(ca[k]) → compare with all 8 lanes of cb                │
//    │    hit_k  = OR-reduce(eq_k)      ← is ca[k] anywhere in cb?         │
//    │    vb_k   = ADD-reduce(vb & eq_k) ← pull the matching B value       │
//    │             (valid because coords are unique: exactly 1 bit set)     │
//    └──────────────────────────────────────────────────────────────────────┘
//
//    ┌── Vectorized compute kernel ─────────────────────────────────────────┐
//    │   All 8 lanes computed simultaneously via vector arith ops.          │
//    │   Non-matching lanes produce garbage — suppressed at scatter.        │
//    └──────────────────────────────────────────────────────────────────────┘
//
//    ┌── Masked scatter to dense output ────────────────────────────────────┐
//    │   vector.scatter %output[0], %ca, %match_a, %result                 │
//    │   Only lanes where hit_k == true write to memory.                   │
//    └──────────────────────────────────────────────────────────────────────┘
//
//    ┌── Pointer advance via frontier ──────────────────────────────────────┐
//    │   frontier = min(max valid coord in A window,                        │
//    │                  max valid coord in B window)                        │
//    │   adv_a = popcount(ca[k] ≤ frontier ∧ mask_a)                       │
//    │   adv_b = popcount(cb[k] ≤ frontier ∧ mask_b)                       │
//    │   This guarantees no intersection is missed across window boundaries.│
//    └──────────────────────────────────────────────────────────────────────┘
//
//  Lowering target:  --convert-vector-to-llvm with --enable-avx2 (W=8)
//                    or --enable-avx512 (promote W to 16)
//
//  Compile + run:
//    mlir-opt sparse_mul_vectorized.mlir              \
//      --sparsification                               \
//      --sparse-tensor-conversion                     \
//      --convert-vector-to-llvm="enable-avx2"        \
//      --convert-scf-to-cf                           \
//      --convert-arith-to-llvm                       \
//      --convert-func-to-llvm                        \
//      --reconcile-unrealized-casts                  \
//    | mlir-cpu-runner                               \
//        --entry-point-result=void                   \
//        --shared-libs=libmlir_runner_utils.so,libmlir_c_runner_utils.so
// ═══════════════════════════════════════════════════════════════════════════

#sparse = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

module {
  func.func private @rtclock() -> f64
  func.func private @printF64(f64)
  llvm.mlir.global internal constant @filename("test_data.tns\00") {addr_space = 0 : i32}

  // ─────────────────────────────────────────────────────────────────────────
  //  @sparse_mul_vec  —  vectorized kernel, W = 8
  // ─────────────────────────────────────────────────────────────────────────
  func.func @sparse_mul_vec(
      %arg0: tensor<10485760xf32, #sparse>,
      %arg1: tensor<10485760xf32, #sparse>) -> tensor<10485760xf32> {

    // ── Scalar constants ────────────────────────────────────────────────────
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c8  = arith.constant 8 : index
    %f0  = arith.constant 0.0 : f32

    // ── Vector constants ────────────────────────────────────────────────────
    %vf_zero  = arith.constant dense<0.0>   : vector<8xf32>    // f32 passthrough
    %vi_zero  = arith.constant dense<0>     : vector<8xindex>  // index passthrough
    %vb_false = arith.constant dense<false> : vector<8xi1>     // empty mask seed

    // ── Output buffer (dense, zero-initialised) ─────────────────────────────
    %out_t  = tensor.empty() : tensor<10485760xf32>
    %out_f  = linalg.fill ins(%f0 : f32) outs(%out_t : tensor<10485760xf32>)
                -> tensor<10485760xf32>
    %output = bufferization.to_buffer %out_f
                : tensor<10485760xf32> to memref<10485760xf32>

    // ── Unpack sparse storage arrays ─────────────────────────────────────────
    //    positions[0..1] = [start, end] of the nonzero range
    //    coordinates     = sorted nonzero column indices
    //    values          = corresponding float values
    %vals_a    = sparse_tensor.values %arg0
                   : tensor<10485760xf32, #sparse> to memref<?xf32>
    %coords_a  = sparse_tensor.coordinates %arg0 {level = 0 : index}
                   : tensor<10485760xf32, #sparse> to memref<?xindex>
    %pos_a     = sparse_tensor.positions  %arg0 {level = 0 : index}
                   : tensor<10485760xf32, #sparse> to memref<?xindex>

    %vals_b    = sparse_tensor.values %arg1
                   : tensor<10485760xf32, #sparse> to memref<?xf32>
    %coords_b  = sparse_tensor.coordinates %arg1 {level = 0 : index}
                   : tensor<10485760xf32, #sparse> to memref<?xindex>
    %pos_b     = sparse_tensor.positions  %arg1 {level = 0 : index}
                   : tensor<10485760xf32, #sparse> to memref<?xindex>

    %start_a = memref.load %pos_a[%c0] : memref<?xindex>
    %end_a   = memref.load %pos_a[%c1] : memref<?xindex>
    %start_b = memref.load %pos_b[%c0] : memref<?xindex>
    %end_b   = memref.load %pos_b[%c1] : memref<?xindex>

    // ═════════════════════════════════════════════════════════════════════════
    //  Main vectorized co-iteration loop
    //  Loop-carried state: (%i, %j)  — current position in A and B
    // ═════════════════════════════════════════════════════════════════════════
    %_:2 = scf.while (%i = %start_a, %j = %start_b) : (index, index) -> (index, index) {
      // Continue while BOTH sides have remaining nonzeros.
      // (Tail of whichever side runs out first cannot match anything.)
      %more_a = arith.cmpi ult, %i, %end_a : index
      %more_b = arith.cmpi ult, %j, %end_b : index
      %more   = arith.andi %more_a, %more_b : i1
      scf.condition(%more) %i, %j : index, index

    } do {
    ^bb0(%i: index, %j: index):

      // ── Step 1 ─ Compute valid lane counts for this window ─────────────────
      //   rem_x = how many nonzeros are left in X; cnt_x = min(rem_x, 8).
      //   cnt_x drives the validity mask so we don't read past the array end.
      %rem_a = arith.subi %end_a, %i  : index
      %rem_b = arith.subi %end_b, %j  : index
      %cnt_a = arith.minui %rem_a, %c8 : index
      %cnt_b = arith.minui %rem_b, %c8 : index

      // ── Step 2 ─ Build validity masks ──────────────────────────────────────
      //   vector.create_mask N → [true×N, false×(8−N)]
      //   These prevent reads/writes past the array end on the final chunk.
      %mask_a = vector.create_mask %cnt_a : vector<8xi1>
      %mask_b = vector.create_mask %cnt_b : vector<8xi1>

      // ── Step 3 ─ Masked load of coordinates and values ─────────────────────
      //   Invalid lanes receive the passthrough (0 / 0.0) and will never
      //   generate a coordinate match because no valid coord can be 0 twice.
      %ca = vector.maskedload %coords_a[%i], %mask_a, %vi_zero
              : memref<?xindex>, vector<8xi1>, vector<8xindex> -> vector<8xindex>
      %cb = vector.maskedload %coords_b[%j], %mask_b, %vi_zero
              : memref<?xindex>, vector<8xi1>, vector<8xindex> -> vector<8xindex>
      %va = vector.maskedload %vals_a[%i], %mask_a, %vf_zero
              : memref<?xf32>, vector<8xi1>, vector<8xf32> -> vector<8xf32>
      %vb = vector.maskedload %vals_b[%j], %mask_b, %vf_zero
              : memref<?xf32>, vector<8xi1>, vector<8xf32> -> vector<8xf32>

      // ── Step 4 ─ Shuffle-based all-pairs intersection ──────────────────────
      //
      //   For each A lane k (0..7):
      //     (a) broadcast ca[k] to a full vector and compare with cb  → eq_k
      //     (b) mask eq_k with mask_b to suppress invalid B lanes     → eq_k_v
      //     (c) OR-reduce eq_k_v to get scalar hit_k  (true = match found)
      //     (d) SELECT vb where eq_k_v, else 0.0; ADD-reduce → vb_k
      //         ↑ This works because sparse coords are UNIQUE:
      //           at most one lane of eq_k_v is true, so the sum is
      //           exactly the one matching B value (or 0 if no match).
      //
      //   Hardware mapping (after vector-to-LLVM lowering):
      //     AVX2:    vpbroadcastd + vpcmpeqd + vptest  (per lane)
      //     AVX-512: vpbroadcastd + vpcmpeqd + kortestw (whole mask register)
      //     ARM SVE: DUP + CMPEQ + PTEST               (predicate registers)

      // ·· Lane 0 ··
      %a0_s  = vector.extract %ca[0] : index from vector<8xindex>
      %a0_bc = vector.broadcast %a0_s : index to vector<8xindex>
      %eq0   = arith.cmpi eq, %a0_bc, %cb : vector<8xindex>
      %eq0_v = arith.andi %eq0, %mask_b    : vector<8xi1>
      %hit0  = vector.reduction <or>,  %eq0_v : vector<8xi1> into i1
      %sel0  = arith.select %eq0_v, %vb, %vf_zero : vector<8xi1>, vector<8xf32>
      %vb0   = vector.reduction <add>, %sel0 : vector<8xf32> into f32

      // ·· Lane 1 ··
      %a1_s  = vector.extract %ca[1] : index from vector<8xindex>
      %a1_bc = vector.broadcast %a1_s : index to vector<8xindex>
      %eq1   = arith.cmpi eq, %a1_bc, %cb : vector<8xindex>
      %eq1_v = arith.andi %eq1, %mask_b    : vector<8xi1>
      %hit1  = vector.reduction <or>,  %eq1_v : vector<8xi1> into i1
      %sel1  = arith.select %eq1_v, %vb, %vf_zero : vector<8xi1>, vector<8xf32>
      %vb1   = vector.reduction <add>, %sel1 : vector<8xf32> into f32

      // ·· Lane 2 ··
      %a2_s  = vector.extract %ca[2] : index from vector<8xindex>
      %a2_bc = vector.broadcast %a2_s : index to vector<8xindex>
      %eq2   = arith.cmpi eq, %a2_bc, %cb : vector<8xindex>
      %eq2_v = arith.andi %eq2, %mask_b    : vector<8xi1>
      %hit2  = vector.reduction <or>,  %eq2_v : vector<8xi1> into i1
      %sel2  = arith.select %eq2_v, %vb, %vf_zero : vector<8xi1>, vector<8xf32>
      %vb2   = vector.reduction <add>, %sel2 : vector<8xf32> into f32

      // ·· Lane 3 ··
      %a3_s  = vector.extract %ca[3] : index from vector<8xindex>
      %a3_bc = vector.broadcast %a3_s : index to vector<8xindex>
      %eq3   = arith.cmpi eq, %a3_bc, %cb : vector<8xindex>
      %eq3_v = arith.andi %eq3, %mask_b    : vector<8xi1>
      %hit3  = vector.reduction <or>,  %eq3_v : vector<8xi1> into i1
      %sel3  = arith.select %eq3_v, %vb, %vf_zero : vector<8xi1>, vector<8xf32>
      %vb3   = vector.reduction <add>, %sel3 : vector<8xf32> into f32

      // ·· Lane 4 ··
      %a4_s  = vector.extract %ca[4] : index from vector<8xindex>
      %a4_bc = vector.broadcast %a4_s : index to vector<8xindex>
      %eq4   = arith.cmpi eq, %a4_bc, %cb : vector<8xindex>
      %eq4_v = arith.andi %eq4, %mask_b    : vector<8xi1>
      %hit4  = vector.reduction <or>,  %eq4_v : vector<8xi1> into i1
      %sel4  = arith.select %eq4_v, %vb, %vf_zero : vector<8xi1>, vector<8xf32>
      %vb4   = vector.reduction <add>, %sel4 : vector<8xf32> into f32

      // ·· Lane 5 ··
      %a5_s  = vector.extract %ca[5] : index from vector<8xindex>
      %a5_bc = vector.broadcast %a5_s : index to vector<8xindex>
      %eq5   = arith.cmpi eq, %a5_bc, %cb : vector<8xindex>
      %eq5_v = arith.andi %eq5, %mask_b    : vector<8xi1>
      %hit5  = vector.reduction <or>,  %eq5_v : vector<8xi1> into i1
      %sel5  = arith.select %eq5_v, %vb, %vf_zero : vector<8xi1>, vector<8xf32>
      %vb5   = vector.reduction <add>, %sel5 : vector<8xf32> into f32

      // ·· Lane 6 ··
      %a6_s  = vector.extract %ca[6] : index from vector<8xindex>
      %a6_bc = vector.broadcast %a6_s : index to vector<8xindex>
      %eq6   = arith.cmpi eq, %a6_bc, %cb : vector<8xindex>
      %eq6_v = arith.andi %eq6, %mask_b    : vector<8xi1>
      %hit6  = vector.reduction <or>,  %eq6_v : vector<8xi1> into i1
      %sel6  = arith.select %eq6_v, %vb, %vf_zero : vector<8xi1>, vector<8xf32>
      %vb6   = vector.reduction <add>, %sel6 : vector<8xf32> into f32

      // ·· Lane 7 ··
      %a7_s  = vector.extract %ca[7] : index from vector<8xindex>
      %a7_bc = vector.broadcast %a7_s : index to vector<8xindex>
      %eq7   = arith.cmpi eq, %a7_bc, %cb : vector<8xindex>
      %eq7_v = arith.andi %eq7, %mask_b    : vector<8xi1>
      %hit7  = vector.reduction <or>,  %eq7_v : vector<8xi1> into i1
      %sel7  = arith.select %eq7_v, %vb, %vf_zero : vector<8xi1>, vector<8xf32>
      %vb7   = vector.reduction <add>, %sel7 : vector<8xf32> into f32

      // ── Step 5 ─ Pack per-lane scalars back into vectors ───────────────────
      //
      //   match_a[k] = hit_k  ∧  mask_a[k]
      //   vb_matched[k] = the B value aligned to A lane k (0.0 if no match)
      //
      //   NOTE: This scalar→vector round-trip is the one part that the pass
      //   can avoid by building the mask directly from the eq_k vectors.
      //   It is written explicitly here to make the logic transparent.
      %m0 = vector.insert %hit0, %vb_false[0] : i1 into vector<8xi1>
      %m1 = vector.insert %hit1, %m0[1]       : i1 into vector<8xi1>
      %m2 = vector.insert %hit2, %m1[2]       : i1 into vector<8xi1>
      %m3 = vector.insert %hit3, %m2[3]       : i1 into vector<8xi1>
      %m4 = vector.insert %hit4, %m3[4]       : i1 into vector<8xi1>
      %m5 = vector.insert %hit5, %m4[5]       : i1 into vector<8xi1>
      %m6 = vector.insert %hit6, %m5[6]       : i1 into vector<8xi1>
      %m7 = vector.insert %hit7, %m6[7]       : i1 into vector<8xi1>
      %match_a = arith.andi %m7, %mask_a : vector<8xi1>   // suppress invalid A lanes

      %b0 = vector.insert %vb0, %vf_zero[0] : f32 into vector<8xf32>
      %b1 = vector.insert %vb1, %b0[1]      : f32 into vector<8xf32>
      %b2 = vector.insert %vb2, %b1[2]      : f32 into vector<8xf32>
      %b3 = vector.insert %vb3, %b2[3]      : f32 into vector<8xf32>
      %b4 = vector.insert %vb4, %b3[4]      : f32 into vector<8xf32>
      %b5 = vector.insert %vb5, %b4[5]      : f32 into vector<8xf32>
      %b6 = vector.insert %vb6, %b5[6]      : f32 into vector<8xf32>
      %vb_matched = vector.insert %vb7, %b6[7] : f32 into vector<8xf32>

      // ── Step 6 ─ Vectorized compute kernel ─────────────────────────────────
      //
      //   All 8 lanes execute in parallel using vector<8xf32> arith ops.
      //   The kernel is the same polynomial as the scalar version:
      //     result = ((((va*vb+va)*vb+va)*vb+va)*vb+va)*vb
      //
      //   Non-matching lanes will compute with vb_matched[k]=0.0, giving
      //   garbage — but they are suppressed by match_a at the scatter below.
      //   The compiler can eliminate the dead computation with DCE.
      //
      //   Hardware: These lower to 8× vpfmadd213ps in a single AVX2 block.
      %k0 = arith.mulf %va, %vb_matched : vector<8xf32>   // va * vb
      %k1 = arith.addf %k0, %va         : vector<8xf32>   // + va
      %k2 = arith.mulf %k1, %vb_matched : vector<8xf32>   // * vb
      %k3 = arith.addf %k2, %va         : vector<8xf32>   // + va
      %k4 = arith.mulf %k3, %vb_matched : vector<8xf32>   // * vb
      %k5 = arith.addf %k4, %va         : vector<8xf32>   // + va
      %k6 = arith.mulf %k5, %vb_matched : vector<8xf32>   // * vb
      %k7 = arith.addf %k6, %va         : vector<8xf32>   // + va
      %k8 = arith.mulf %k7, %vb_matched : vector<8xf32>   // * vb  ← final

      // ── Step 7 ─ Masked scatter to dense output ─────────────────────────────
      //
      //   vector.scatter base[0], indices=%ca, mask=%match_a, values=%k8
      //
      //   Each lane k writes k8[k] to output[ca[k]], but ONLY when match_a[k]
      //   is true.  This is a single scatter instruction on AVX-512 (vscatterqps).
      //   On AVX2 it expands to 8 predicated scalar stores — still branch-free.
      //
      //   Base index is %c0 because ca already holds absolute output coordinates.
      vector.scatter %output[%c0], %ca, %match_a, %k8
        : memref<10485760xf32>, vector<8xindex>, vector<8xi1>, vector<8xf32>

      // ── Step 8 ─ Advance pointers past the "frontier" ───────────────────────
      //
      //   frontier = min(max valid coord in A window, max valid coord in B window)
      //
      //   Invariant:  any coord > frontier in EITHER window COULD still match
      //               a coord in the other's NEXT window, so we keep it.
      //   Invariant:  any coord ≤ frontier in EITHER window has already been
      //               fully compared against the other window, so we discard it.
      //
      //   adv_x = popcount(cx[k] ≤ frontier  ∧  mask_x[k])
      //         = number of elements to consume from X
      //
      //   popcount trick: zero-extend i1 mask to index, then add-reduce.

      // Replace invalid lanes with 0 before taking the max
      // (invalid lanes already have coord 0 from the masked load passthrough)
      %max_a = vector.reduction <umax>, %ca : vector<8xindex> into index
      %max_b = vector.reduction <umax>, %cb : vector<8xindex> into index
      // ↑ Safe because:  invalid A lanes = 0 (can never be the max since
      //   valid coords are positive and we have at least 1 valid lane here).
      //   If ALL lanes are invalid we never enter the loop body.

      %frontier   = arith.minui %max_a, %max_b : index
      %frontier_v = vector.broadcast %frontier : index to vector<8xindex>

      %le_a = arith.cmpi ule, %ca, %frontier_v : vector<8xindex>
      %le_b = arith.cmpi ule, %cb, %frontier_v : vector<8xindex>
      %adv_mask_a = arith.andi %le_a, %mask_a : vector<8xi1>
      %adv_mask_b = arith.andi %le_b, %mask_b : vector<8xi1>

      %adv_a_i = arith.extui %adv_mask_a : vector<8xi1> to vector<8xindex>
      %adv_b_i = arith.extui %adv_mask_b : vector<8xi1> to vector<8xindex>
      %adv_a   = vector.reduction <add>, %adv_a_i : vector<8xindex> into index
      %adv_b   = vector.reduction <add>, %adv_b_i : vector<8xindex> into index

      %new_i = arith.addi %i, %adv_a : index
      %new_j = arith.addi %j, %adv_b : index
      scf.yield %new_i, %new_j : index, index

    } // end scf.while

    %result = bufferization.to_tensor %output
                : memref<10485760xf32> to tensor<10485760xf32>
    return %result : tensor<10485760xf32>
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  @main  —  benchmark harness (1000 iterations, same as scalar reference)
  //
  //  Prints total elapsed seconds.  To get per-iteration time, divide by 1000.
  //  Compare against the scalar SCF lowering to measure vectorization gain.
  // ─────────────────────────────────────────────────────────────────────────
  func.func @main() {
    %c0    = arith.constant 0    : index
    %c1    = arith.constant 1    : index
    %iters = arith.constant 1000 : index

    // Load sparse input from .tns file
    %ptr = llvm.mlir.addressof @filename : !llvm.ptr
    %sv  = sparse_tensor.new %ptr : !llvm.ptr to tensor<10485760xf32, #sparse>

    // ── Start timer ─────────────────────────────────────────────────────────
    %t0 = func.call @rtclock() : () -> f64

    // ── Benchmark loop ───────────────────────────────────────────────────────
    scf.for %iter = %c0 to %iters step %c1 {
      %r = func.call @sparse_mul_vec(%sv, %sv)
             : (tensor<10485760xf32, #sparse>,
                tensor<10485760xf32, #sparse>) -> tensor<10485760xf32>
      bufferization.dealloc_tensor %r : tensor<10485760xf32>
    }

    // ── Stop timer and report ────────────────────────────────────────────────
    %t1      = func.call @rtclock() : () -> f64
    %elapsed = arith.subf %t1, %t0 : f64

    // Prints total seconds for 1000 iterations.
    // Scalar reference is in sparse_mul_scalar.mlir — run both and compare.
    func.call @printF64(%elapsed) : (f64) -> ()
    return
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  COMPARISON NOTES
// ───────────────────────────────────────────────────────────────────────────
//
//  Scalar (original SCF lowering)       │  Vectorized (this file, W=8)
//  ─────────────────────────────────────┼─────────────────────────────────────
//  1 element compared per loop iter     │  up to 64 pairs per iter (8×8)
//  branch on coord equality             │  branchless predicate mask
//  9 scalar FP ops per match            │  9 vector FP ops = 72 FP ops/iter
//  1 scalar store per match             │  1 masked scatter per iter
//  irregular pointer advance (+1/+0)    │  frontier-based advance (≤ W/iter)
//
//  Expected speedup on dense-ish data:
//    ~4–8× on AVX2,  ~8–16× on AVX-512
//    (Lower if data is very sparse: intersection rate < 1/W kills throughput)
//
//  Profitability threshold (rule of thumb):
//    Speedup ≈ W × intersection_rate / (1 + shuffle_overhead_factor)
//    Worth it when intersection_rate > ~10%  (e.g. 90%+ zero means sparse).
//
//  Dataset suggestions for timing experiments:
//    • SNAP (Stanford Network Analysis Project) social-network edge lists
//    • SuiteSparse Matrix Collection (sparse.tamu.edu) — varying densities
//    • Synthetic: generate with scipy.sparse.random(density=0.01 / 0.1 / 0.5)
// ═══════════════════════════════════════════════════════════════════════════
