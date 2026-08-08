// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TRANSFORMS_SPLYCE_BUFFERIZATION_PREP_PASS_H
#define TRANSFORMS_SPLYCE_BUFFERIZATION_PREP_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

// Creates the pass that stamps `restrict` onto every bufferization.to_tensor
// op so one-shot-bufferize can bufferize return tensors in place (see the
// AddRestrictToTensor pass in Passes.td for the full rationale). To disable
// it without removing the pass from the pipeline, comment out the
// markToTensorRestrict() call in runOnOperation().
std::unique_ptr<Pass> createAddRestrictToTensorPass();

// Registers the pass for --splyce-bufferize-restrict CLI availability.
void registerAddRestrictToTensorPass();

// TableGen-generated base class.
#define GEN_PASS_DECL_ADDRESTRICTTOTENSOR
#include "Transforms/Passes.h.inc"

} // namespace mlir

#endif // TRANSFORMS_SPLYCE_BUFFERIZATION_PREP_PASS_H
