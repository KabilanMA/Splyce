// ─────────────────────────────────────────────────────────────────────────────
// coiter_vectorize.mlir  —  FileCheck test for --coiter-vectorize
//
// Run:
//   mlir-opt --coiter-vectorize="vector-width=8" %s | FileCheck %s
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @coiter_basic
// CHECK-NOT:   scf.if
// CHECK:       vector.create_mask
// CHECK:       vector.maskedload
// CHECK:       vector.broadcast
// CHECK:       vector.reduction
// CHECK:       vector.scatter
// CHECK:       scf.yield

func.func @coiter_basic(%coords_a: memref<?xindex>,
                         %coords_b: memref<?xindex>,
                         %vals_a:   memref<?xf32>,
                         %vals_b:   memref<?xf32>,
                         %output:   memref<1024xf32>,
                         %end_a:    index,
                         %end_b:    index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // ── Canonical co-iteration loop ───────────────────────────────────────────
  // This is the pattern the pass must recognize and replace.
  scf.while (%i = %c0, %j = %c0) : (index, index) -> (index, index) {
    %more_a = arith.cmpi ult, %i, %end_a : index
    %more_b = arith.cmpi ult, %j, %end_b : index
    %more   = arith.andi %more_a, %more_b : i1
    scf.condition(%more) %i, %j : index, index

  } do {
  ^bb0(%i: index, %j: index):
    %ca   = memref.load %coords_a[%i] : memref<?xindex>
    %cb   = memref.load %coords_b[%j] : memref<?xindex>

    // Compute min via select pattern (also accept arith.minui).
    %lt   = arith.cmpi ult, %cb, %ca : index
    %min  = arith.select %lt, %cb, %ca : index

    // Equality checks.
    %eq_a = arith.cmpi eq, %ca, %min : index
    %eq_b = arith.cmpi eq, %cb, %min : index
    %both = arith.andi %eq_a, %eq_b : i1

    scf.if %both {
      %va  = memref.load %vals_a[%i] : memref<?xf32>
      %vb  = memref.load %vals_b[%j] : memref<?xf32>

      // Compute kernel: ((va*vb + va)*vb + va)*vb  (≥ 2 FP ops → profitable)
      %m1  = arith.mulf %va, %vb : f32
      %a1  = arith.addf %m1, %va : f32
      %m2  = arith.mulf %a1, %vb : f32
      %a2  = arith.addf %m2, %va : f32
      %m3  = arith.mulf %a2, %vb : f32

      memref.store %m3, %output[%min] : memref<1024xf32>
    }

    // Pointer advance.
    %adv_i = arith.addi %i, %c1 : index
    %adv_j = arith.addi %j, %c1 : index
    %new_i = arith.select %eq_a, %adv_i, %i : index
    %new_j = arith.select %eq_b, %adv_j, %j : index
    scf.yield %new_i, %new_j : index, index
  }

  return
}
