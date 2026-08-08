// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    and rewrites it into a SIMD fast-lane loop followed by a scalar epilogue:

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

    The fast lane runs as long as both streams still have at least W
    elements left, so its loads never need masking: each iteration loads W
    coordinates and values per stream, does a W×W cross-compare to find
    matches, folds the matched contributions into the loop-carried
    accumulator, and advances every stream's pointer past the smallest
    coordinate seen (a branchless frontier). Once a stream runs low, the
    original scalar while loop takes over as an epilogue, picking up right
    where the fast lane's pointers and accumulator left off.

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

def AddFastMathContract : Pass<"splyce-fastmath"> {
  let summary = "Stamp reassoc+contract fastmath flags on floating-point arithmetic ops";

  let description = [{
    Neither the plain sparsification baseline nor Splyce's generated code
    (SIMD fast lane, scalar epilogue) carries any fastmath annotation, so
    strict IEEE-754 semantics force:
      - horizontal reductions (vector.reduction) to lower as a strictly
        sequential sum instead of a parallel/tree reduction, and
      - multiply-accumulate pairs (arith.mulf followed by arith.addf) to stay
        as separate instructions instead of fusing into a single FMA.

    Because this project's kernels compile pre-lowered MLIR/LLVM IR directly
    (mlir-opt/splyce-opt -> splyce-translate/mlir-translate -> clang), Clang's
    -ffast-math / -Ofast never gets a chance to act: those flags only take
    effect when Clang's own C/C++ frontend lowers source to IR, which is
    skipped entirely for .ll input. This pass is the IR-level equivalent,
    scoped to exactly the ops that implement ArithFastMathInterface
    (arith.mulf, arith.addf, vector.reduction, etc.) — the index/pointer
    arithmetic used for sparse coordinate walking is untouched, since it
    doesn't implement the interface.

    Idempotent: ORs reassoc|contract into each op's existing fastmath flags.
  }];

  let options = [
    Option<"targetFunction", "target-function", "std::string",
           /*default=*/"\"\"",
           "Only annotate ops inside this named function. Empty string "
           "(default) matches all functions.">
  ];

  let constructor = "mlir::createAddFastMathContractPass()";

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif // COITER_VECTORIZE_PASSES_TD
