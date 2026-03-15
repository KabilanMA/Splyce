// ─────────────────────────────────────────────────────────────────────────────
// CoIterPasses.td - TableGen declarations for the co-iteration vectorization pass
//
// Generates:
//   PassBase class  -> inherited by CoIterVectorizePass
//   CLI option      -> mlir-opt --splyce
//   Pass registry   -> auto-registered via registerCoIterVectorizePass()
// ─────────────────────────────────────────────────────────────────────────────

#ifndef COITER_SPLYCE_PASSES_TD
#define COITER_SPLYCE_PASSES_TD

include "mlir/Pass/PassBase.td"

def CoIterVectorize : Pass<"splyce"> {
  let summary = "Vectorize scf.while co-iteration loops using Vector dialect";

  let description = [{
    Detects the canonical co-iteration pattern emitted by sparse tensor
    lowering (two sorted-list pointer advances with a branch on coordinate
    equality) and rewrites it into a blocked, branch-free Vector dialect loop.

    The pattern being matched is:

      scf.while (%i, %j) : (index, index) {
        // condition: both pointers in bounds
        %ca = in_bounds(%i, end_a)
        %cb = in_bounds(%j, end_b)
        scf.condition(%ca & %cb) %i, %j
      } do {
        %coord_a = memref.load %coords_a[%i]
        %coord_b = memref.load %coords_b[%j]
        %min     = select min(%coord_a, %coord_b)
        scf.if (%coord_a == %min && %coord_b == %min) {
          // arbitrary compute kernel
        }
        // advance %i if coord_a matched min, %j if coord_b matched min
        scf.yield %new_i, %new_j
      }

    It is replaced by a blocked scf.while that:
      1. Loads W coordinates + values from each side via vector.maskedload
      2. Performs a shuffle-based all-pairs intersection (branchless predicate)
      3. Lifts the compute kernel body to vector<Wxf32> operations
      4. Writes results via vector.scatter
      5. Advances pointers using a frontier-based popcount (no branches)

    Options:
      vector-width  : SIMD width W (default 8; use 16 for AVX-512)
      min-density   : Skip transformation if estimated intersection rate is
                      below this fraction (default 0.0 = always transform)
  }];

  // ── Pass options ───────────────────────────────────────────────────────────
  let options = [
    Option<"vectorWidth", "vector-width", "unsigned",
           /*default=*/"8",
           "SIMD lane count (8=AVX2/NEON, 16=AVX-512, scalable via SVE)">,

    Option<"minDensity", "min-density", "float",
           /*default=*/"0.0",
           "Minimum estimated intersection density; skip if below threshold">,

    Option<"enableTailScalarFallback", "tail-scalar-fallback", "bool",
           /*default=*/"true",
           "Emit a scalar epilogue for the remainder when nnz % W != 0">
  ];

  let constructor = "mlir::createCoIterVectorizePass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // COITER_VECTORIZE_PASSES_TD
