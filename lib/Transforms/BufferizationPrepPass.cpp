// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BufferizationPrepPass.cpp - stamps `restrict` onto every
// bufferization.to_tensor op that lacks it, ahead of one-shot-bufferize.
// See BufferizationPrepPass.h / Passes.td for why.

#include "Transforms/BufferizationPrepPass.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GEN_PASS_DEF_ADDRESTRICTTOTENSOR
namespace mlir {
#include "Transforms/Passes.h.inc"
}

using namespace mlir;

// Walk rootOp and add the restrict unit-attribute to every to_tensor op that
// does not already carry it.  Idempotent: already-restricted ops are skipped.
static void markToTensorRestrict(Operation *rootOp) {
    rootOp->walk([](bufferization::ToTensorOp op) {
        if (!op.getRestrict())
            op->setAttr("restrict", UnitAttr::get(op->getContext()));
    });
}

namespace {

struct AddRestrictToTensorPass
    : public mlir::impl::AddRestrictToTensorBase<AddRestrictToTensorPass> {

    void runOnOperation() override {
        markToTensorRestrict(getOperation()); // comment this line to disable
    }
};

} // namespace

std::unique_ptr<Pass> mlir::createAddRestrictToTensorPass() {
    return std::make_unique<AddRestrictToTensorPass>();
}

void mlir::registerAddRestrictToTensorPass() {
    PassRegistration<AddRestrictToTensorPass>();
}
