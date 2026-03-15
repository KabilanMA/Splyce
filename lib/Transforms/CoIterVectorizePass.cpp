// CoIterVectorizePass.cpp
//
// Pass entry point.  Wires CoIterPattern (recognition) and
// CoIterVectorBuilder (emission) into the MLIR RewritePattern + Pass
// infrastructure.
//
// Flow:
//
//   CoIterVectorizePass::runOnOperation()
//     └─ applyPatternsGreedily(func, patterns)
//           └─ CoIterVectorizePattern::matchAndRewrite(whileOp, rewriter)
//                 ├─ tryMatchCoIter(whileOp)           // recognize
//                 ├─ isProfitable(desc, minDensity)     // gate
//                 ├─ VectorLoopBuilder::build(loc)      // emit replacement
//                 └─ rewriter.eraseOp(whileOp)          // remove original

#include "Transforms/CoIterVectorizePass.h"
#include "Transforms/CoIterPattern.h"
#include "Transforms/CoIterVectorBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "splyce-vectorize"

// include the TableGen-generated base class definition
#define GEN_PASS_DEF_COITERVECTORIZE
namespace mlir {
#include "Transforms/Passes.h.inc"
}

using namespace mlir;
using namespace mlir::splyce;

// CoIterVectorizePattern
//
// One RewritePattern that:
//  1. Attempts to match an scf.while as a co-iteration loop
//  2. Checks profitability
//  3. Emits the vectorized replacement via VectorLoopBuilder
//  4. Erases the original scf.while
struct CoIterVectorizePattern : public OpRewritePattern<scf::WhileOp> {
    CoIterVectorizePattern(MLIRContext *ctx, unsigned vectorWidth, float minDensity, PatternBenefit benefit = 1) : 
        OpRewritePattern<scf::WhileOp>(ctx, benefit), vectorWidth(vectorWidth), minDensity(minDensity) {}
    
    LogicalResult matchAndRewrite(scf::WhileOp whileOp, PatternRewriter &rewriter) const override {
        // recognition
        auto descOpt = tryMatchCoIter(whileOp);
        if (!descOpt) {
            LLVM_DEBUG(llvm::dbgs() << "[splyce] Pattern did not match on " << whileOp << "\n");
            return failure();
        }

        CoIterDescriptor &desc = *descOpt;

        // profitablity gate
        if (!isProfitable(desc, minDensity)) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] Not profitable - skipping\n");
            return failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Vectorizing co-iteration loop at " << whileOp.getLoc() << " (W-" << vectorWidth << ")\n");

        // emit vectorized replacement
        rewriter.setInsertionPoint(whileOp);
        VectorLoopBuilder builder(desc, vectorWidth, rewriter);
        builder.build(whileOp.getLoc());

        // erase original
        rewriter.eraseOp(whileOp);
        return success();
    }

private:
    unsigned vectorWidth;
    float minDensity;
};

// pass implementation
namespace {

struct CoIterVectorizePass : public mlir::impl::CoIterVectorizeBase<CoIterVectorizePass> {
    // constructors: one for programmatic use (explicit args),
    // one for TableGen-generated CLI parsing (uses the base class options directly)
    CoIterVectorizePass() = default;

    CoIterVectorizePass(unsigned vectorWidth, float minDensity, bool enableTailScalarFallback) {
        this->vectorWidth = vectorWidth;
        this->minDensity = minDensity;
        this->enableTailScalarFallback = enableTailScalarFallback;
    }

    void runOnOperation() override {
        Operation *op = getOperation();

        RewritePatternSet patterns(&getContext());
        populateCoIterVectorizePatterns(patterns, vectorWidth, minDensity);

        // use the greedy driver: keeps applying patterns until a fixed point
        // this handles nested co-iteration loops (e.g. SpGEMM outer/inner)
        if (failed(applyPatternsGreedily(op, std::move(patterns))))
            signalPassFailure();
    }
};
} // anonymous namespace

// public api

void mlir::populateCoIterVectorizePatterns(RewritePatternSet &patterns, unsigned vectorWidth, float minDensity) {
    patterns.add<CoIterVectorizePattern>(patterns.getContext(), vectorWidth, minDensity);
}

std::unique_ptr<Pass> mlir::createCoIterVectorizePass(unsigned vectorWidth, float minDensity, bool tailFallback) {
    return std::make_unique<CoIterVectorizePass>(vectorWidth, minDensity, tailFallback);
}

void mlir::registerCoIterVectorizePass() {
    PassRegistration<CoIterVectorizePass>();
}