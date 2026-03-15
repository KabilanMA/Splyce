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

// ─────────────────────────────────────────────────────────────────────────────
// N=3 test: three-way co-iteration
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @coiter_3way
// CHECK-NOT:   scf.if
// CHECK:       vector.create_mask
// CHECK:       vector.maskedload
// CHECK:       vector.scatter
func.func @coiter_3way(%ca: memref<?xindex>, %cb: memref<?xindex>,
                        %cc: memref<?xindex>,
                        %va: memref<?xf32>,  %vb: memref<?xf32>,
                        %vc: memref<?xf32>,
                        %out: memref<1024xf32>,
                        %ea: index, %eb: index, %ec: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.while (%i = %c0, %j = %c0, %k = %c0)
      : (index, index, index) -> (index, index, index) {
    %ma  = arith.cmpi ult, %i, %ea : index
    %mb  = arith.cmpi ult, %j, %eb : index
    %mc  = arith.cmpi ult, %k, %ec : index
    %mab = arith.andi %ma, %mb : i1
    %m   = arith.andi %mab, %mc : i1
    scf.condition(%m) %i, %j, %k : index, index, index
  } do {
  ^bb0(%i: index, %j: index, %k: index):
    %coord_a = memref.load %ca[%i] : memref<?xindex>
    %coord_b = memref.load %cb[%j] : memref<?xindex>
    %coord_c = memref.load %cc[%k] : memref<?xindex>

    %min_ab  = arith.minui %coord_a, %coord_b : index
    %min_abc = arith.minui %min_ab,  %coord_c : index

    %eq_a = arith.cmpi eq, %coord_a, %min_abc : index
    %eq_b = arith.cmpi eq, %coord_b, %min_abc : index
    %eq_c = arith.cmpi eq, %coord_c, %min_abc : index
    %eab  = arith.andi %eq_a, %eq_b : i1
    %eabc = arith.andi %eab,  %eq_c : i1

    scf.if %eabc {
      %fa = memref.load %va[%i] : memref<?xf32>
      %fb = memref.load %vb[%j] : memref<?xf32>
      %fc = memref.load %vc[%k] : memref<?xf32>
      // 6 FP ops >= N*2=6 → profitable
      %r0 = arith.mulf %fa, %fb : f32
      %r1 = arith.addf %r0, %fc : f32
      %r2 = arith.mulf %r1, %fb : f32
      %r3 = arith.addf %r2, %fa : f32
      %r4 = arith.mulf %r3, %fc : f32
      %r5 = arith.addf %r4, %fb : f32
      memref.store %r5, %out[%min_abc] : memref<1024xf32>
    }

    %ai = arith.addi %i, %c1 : index
    %aj = arith.addi %j, %c1 : index
    %ak = arith.addi %k, %c1 : index
    %ni = arith.select %eq_a, %ai, %i : index
    %nj = arith.select %eq_b, %aj, %j : index
    %nk = arith.select %eq_c, %ak, %k : index
    scf.yield %ni, %nj, %nk : index, index, index
  }
  return
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative test: single FP op → below arithmetic intensity threshold → NO transform
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @coiter_unprofitable
// CHECK:       scf.while
// CHECK:       scf.if
func.func @coiter_unprofitable(%coords_a: memref<?xindex>,
                                %coords_b: memref<?xindex>,
                                %vals_a:   memref<?xf32>,
                                %vals_b:   memref<?xf32>,
                                %output:   memref<1024xf32>,
                                %end_a:    index,
                                %end_b:    index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.while (%i = %c0, %j = %c0) : (index, index) -> (index, index) {
    %more_a = arith.cmpi ult, %i, %end_a : index
    %more_b = arith.cmpi ult, %j, %end_b : index
    %more   = arith.andi %more_a, %more_b : i1
    scf.condition(%more) %i, %j : index, index
  } do {
  ^bb0(%i: index, %j: index):
    %ca   = memref.load %coords_a[%i] : memref<?xindex>
    %cb   = memref.load %coords_b[%j] : memref<?xindex>
    %lt   = arith.cmpi ult, %cb, %ca : index
    %min  = arith.select %lt, %cb, %ca : index
    %eq_a = arith.cmpi eq, %ca, %min : index
    %eq_b = arith.cmpi eq, %cb, %min : index
    %both = arith.andi %eq_a, %eq_b : i1
    scf.if %both {
      %va = memref.load %vals_a[%i] : memref<?xf32>
      %vb = memref.load %vals_b[%j] : memref<?xf32>
      // Only 1 FP op — below threshold.
      %m1 = arith.mulf %va, %vb : f32
      memref.store %m1, %output[%min] : memref<1024xf32>
    }
    %adv_i = arith.addi %i, %c1 : index
    %adv_j = arith.addi %j, %c1 : index
    %new_i = arith.select %eq_a, %adv_i, %i : index
    %new_j = arith.select %eq_b, %adv_j, %j : index
    scf.yield %new_i, %new_j : index, index
  }
  return
}
