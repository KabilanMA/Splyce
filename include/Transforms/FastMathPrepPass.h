// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TRANSFORMS_SPLYCE_FASTMATH_PREP_PASS_H
#define TRANSFORMS_SPLYCE_FASTMATH_PREP_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

// Creates the pass that stamps `reassoc | contract` fastmath flags onto
// every op implementing arith::ArithFastMathInterface, so the co-iteration
// payload's horizontal reduction and multiply-accumulate can lower as a
// tree/FMA instead of strict sequential IEEE-754 ops. This exists because
// we feed pre-lowered MLIR/LLVM IR straight to clang, so -ffast-math never
// gets a chance to run (see the AddFastMathContract pass in Passes.td for
// the full explanation).
std::unique_ptr<Pass> createAddFastMathContractPass();

// Registers the pass for --splyce-fastmath CLI availability.
void registerAddFastMathContractPass();

// TableGen-generated base class.
#define GEN_PASS_DECL_ADDFASTMATHCONTRACT
#include "Transforms/Passes.h.inc"

} // namespace mlir

#endif // TRANSFORMS_SPLYCE_FASTMATH_PREP_PASS_H
