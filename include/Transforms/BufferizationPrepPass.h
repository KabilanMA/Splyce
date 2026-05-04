#ifndef TRANSFORMS_SPLYCE_BUFFERIZATION_PREP_PASS_H
#define TRANSFORMS_SPLYCE_BUFFERIZATION_PREP_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

// Creates the pass that stamps the restrict attribute on every
// bufferization.to_tensor op found in the IR.
//
// The MLIR sparse lowering pipeline omits restrict from to_tensor ops.
// One-shot-bufferize requires it for in-place bufferization of return tensors.
// This pass inserts the attribute before the bufferization pipeline runs.
//
// To disable the transformation without removing the pass from the pipeline,
// comment out the markToTensorRestrict() call in runOnOperation().
std::unique_ptr<Pass> createAddRestrictToTensorPass();

// Registers the pass for --splyce-bufferize-restrict CLI availability.
void registerAddRestrictToTensorPass();

// TableGen-generated base class.
#define GEN_PASS_DECL_ADDRESTRICTTOTENSOR
#include "Transforms/Passes.h.inc"

} // namespace mlir

#endif // TRANSFORMS_SPLYCE_BUFFERIZATION_PREP_PASS_H
