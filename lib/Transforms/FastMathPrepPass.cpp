// FastMathPrepPass.cpp
//
// Stamps `reassoc | contract` fastmath flags onto every op implementing
// arith::ArithFastMathInterface (arith.mulf, arith.addf, vector.reduction,
// ...) so the horizontal reduction and multiply-accumulate in the
// co-iteration payload can lower as a parallel/tree reduction and fuse into
// FMA. See FastMathPrepPass.h for why this can't just be a clang flag.

#include "Transforms/FastMathPrepPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define GEN_PASS_DEF_ADDFASTMATHCONTRACT
namespace mlir {
#include "Transforms/Passes.h.inc"
}

using namespace mlir;

static constexpr arith::FastMathFlags kFlags =
    arith::FastMathFlags::reassoc | arith::FastMathFlags::contract;

// Walk rootOp and OR reassoc|contract into the fastmath flags of every op
// implementing ArithFastMathInterface. If targetFunction is non-empty, only
// ops lexically inside that named function are touched.
static void markFastMathContract(Operation *rootOp, StringRef targetFunction) {
    rootOp->walk([&](arith::ArithFastMathInterface fmi) {
        Operation *op = fmi.getOperation();
        if (!targetFunction.empty()) {
            auto funcOp = op->getParentOfType<func::FuncOp>();
            if (!funcOp || funcOp.getName() != targetFunction)
                return;
        }
        arith::FastMathFlags merged = kFlags;
        if (arith::FastMathFlagsAttr existing = fmi.getFastMathFlagsAttr())
            merged = existing.getValue() | kFlags;
        op->setAttr(fmi.getFastMathAttrName(), arith::FastMathFlagsAttr::get(op->getContext(), merged));
    });
}

namespace {

struct AddFastMathContractPass
    : public mlir::impl::AddFastMathContractBase<AddFastMathContractPass> {

    void runOnOperation() override {
        markFastMathContract(getOperation(), targetFunction);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::createAddFastMathContractPass() {
    return std::make_unique<AddFastMathContractPass>();
}

void mlir::registerAddFastMathContractPass() {
    PassRegistration<AddFastMathContractPass>();
}
