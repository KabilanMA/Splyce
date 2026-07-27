#ifndef TRANSFORMS_SPLYCE_FASTMATH_PREP_PASS_H
#define TRANSFORMS_SPLYCE_FASTMATH_PREP_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

// Creates the pass that stamps `reassoc | contract` fastmath flags onto every
// op implementing arith::ArithFastMathInterface (arith.mulf, arith.addf,
// vector.reduction, ...).
//
// Neither the plain sparsification baseline nor Splyce's own generated code
// (SIMD fast lane, scalar epilogue) carries any fastmath annotation, so
// strict IEEE-754 semantics force horizontal reductions to lower as a
// sequential sum instead of a tree, and block multiply-accumulate pairs from
// fusing into FMA. Because this project compiles pre-lowered MLIR/LLVM IR
// directly, Clang's -ffast-math / -Ofast never gets a chance to act on it —
// those flags only take effect when Clang's own C/C++ frontend lowers source
// to IR, which is skipped entirely for .ll input. This pass is the IR-level
// equivalent, scoped to exactly the ops that implement
// arith::ArithFastMathInterface; index/pointer arithmetic used for sparse
// coordinate walking is untouched, since it doesn't implement the interface.
std::unique_ptr<Pass> createAddFastMathContractPass();

// Registers the pass for --splyce-fastmath CLI availability.
void registerAddFastMathContractPass();

// TableGen-generated base class.
#define GEN_PASS_DECL_ADDFASTMATHCONTRACT
#include "Transforms/Passes.h.inc"

} // namespace mlir

#endif // TRANSFORMS_SPLYCE_FASTMATH_PREP_PASS_H
