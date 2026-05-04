// Passes.td - TableGen declarations for the co-iteration vectorization pass.
//
// Generates:
//   PassBase class  → inherited by CoIterVectorizePass
//   CLI option      → mlir-opt --splyce
//   Pass registry   → auto-registered via registerCoIterVectorizePass()

#ifndef COITER_SPLYCE_PASSES_TD
#define COITER_SPLYCE_PASSES_TD

include "mlir/Pass/PassBase.td"

def CoIterVectorize : Pass<"splyce"> {
  let summary = "Vectorize scf.while co-iteration loops using Vector dialect";

  let description = [{
    Detects the N-way co-iteration pattern emitted by sparse tensor lowering
    (two sorted-list pointer advances with a branch on coordinate equality)
    and rewrites it into a SIMD fast-lane loop followed by a scalar epilogue.

    The pattern matched is:

      scf.while (%i, %j, %acc) : (index, index, f64) {
        %ca = cmpi ult, %i, end_a
        %cb = cmpi ult, %j, end_b
        scf.condition(%ca & %cb) %i, %j, %acc
      } do {
        %coord_a = memref.load coords_a[%i]
        %coord_b = memref.load coords_b[%j]
        %min     = select min(%coord_a, %coord_b)
        %new_acc = scf.if (%coord_a == %min && %coord_b == %min) {
                     scf.yield acc + a_val * b_val  // arbitrary FP kernel
                   } else { scf.yield %acc }
        scf.yield %new_i, %new_j, %new_acc
      }

    It is replaced by:

      (1) A SIMD fast-lane scf.while that runs while BOTH streams have >= W
          elements remaining (unmasked loads are always safe).  Each iteration:
          - Loads W coordinates and W values per stream (vector.load).
          - Performs a W×W cross-compare: for each driver lane k, broadcasts
            a_coord[k] and compares against all W probe coords.
          - Accumulates per-lane contributions into the loop-carried accumulator.
          - Advances pointers via a branchless frontier (min of last coords).

      (2) The original scalar loop as a scalar epilogue, picking up from the
          pointer positions and accumulator value left by the fast lane.

    Options:
      vector-width    : SIMD width W (default 8; use 16 for AVX-512).
      target-function : Only transform loops inside this named function.
  }];

  let options = [
    Option<"vectorWidth", "vector-width", "unsigned",
           /*default=*/"4",
           "SIMD lane count (4 = SSE/baseline, 8 = AVX2/NEON, 16 = AVX-512)">,

    Option<"targetFunction", "target-function", "std::string",
           /*default=*/"\"\"",
           "Only transform scf.while loops inside this named function. "
           "Empty string (default) matches loops in all functions.">,

    Option<"phaseSelect", "phase-select", "std::string",
           /*default=*/"\"001\"",
           "3-bit phase selector 'XYZ': X=Phase3 (pointer advance), "
           "Y=Phase2 (tensor compute), Z=Phase1 (load + cross-compare). "
           "'1'=vector, '0'=scalar. Default '001' vectorizes Phase 1 only.">
  ];

  let constructor = "mlir::createCoIterVectorizePass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def AddRestrictToTensor : Pass<"splyce-bufferize-restrict"> {
  let summary = "Stamp the restrict attribute on bufferization.to_tensor ops";

  let description = [{
    The MLIR sparse-tensor-conversion pipeline emits

      bufferization.to_tensor %buf : memref<...> to tensor<...>

    without the restrict attribute.  One-shot-bufferize requires restrict to
    establish buffer-ownership: it proves that no other alias writes through the
    same buffer, enabling in-place bufferization of the function return value.

    This pass walks the IR and adds restrict to every to_tensor op that lacks
    it, matching what manual editing would produce.  It is idempotent — already-
    restricted ops are left unchanged.

    To disable the transformation at compile time, comment out the single
    markToTensorRestrict() call inside AddRestrictToTensorPass::runOnOperation.
  }];

  let constructor = "mlir::createAddRestrictToTensorPass()";

  let dependentDialects = ["mlir::bufferization::BufferizationDialect"];
}

#endif // COITER_VECTORIZE_PASSES_TD
