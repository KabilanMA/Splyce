// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CoIterVectorizePass.cpp
//
// Pass entry point: wires CoIterPattern (recognition) and SimdFastLaneBuilder
// (emission) into the MLIR RewritePattern + Pass infrastructure.
//
// Flow:
//   CoIterVectorizePass::runOnOperation()
//     |-- applyPatternsGreedily(func, patterns)
//           |-- CoIterVectorizePattern::matchAndRewrite(whileOp, rewriter)
//                 |-- tryMatchCoIter(whileOp)          // recognize
//                 |-- SimdFastLaneBuilder::build(loc)  // SIMD fast-lane loop
//                 |-- rewriter.clone(*whileOp)         // scalar epilogue
//                 |-- rewriter.replaceOp(whileOp, ...)

#include "Transforms/CoIterVectorizePass.h"
#include "Transforms/CoIterPattern.h"
#include "Transforms/SimdFastLaneBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "splyce-vectorize"

#define GEN_PASS_DEF_COITERVECTORIZE
namespace mlir {
#include "Transforms/Passes.h.inc"
}

using namespace mlir;
using namespace mlir::splyce;

// rewrite pattern

// Matches an scf.while as a co-iteration loop and rewrites it into:
//   - A SIMD fast-lane scf.while (W-wide, unmasked, branchless).
//   - The original scf.while as a scalar epilogue (handles remainder 0..W-1).
//
// The scalar epilogue's init values are the SIMD loop's results, so the
// accumulated value and pointer positions chain correctly.
struct CoIterVectorizePattern : public OpRewritePattern<scf::WhileOp> {
    CoIterVectorizePattern(MLIRContext *ctx, unsigned vectorWidth,
                           std::string targetFunction, std::string phaseSelect,
                           PatternBenefit benefit = 1)
        : OpRewritePattern<scf::WhileOp>(ctx, benefit),
          vectorWidth(vectorWidth),
          targetFunction(std::move(targetFunction)),
          cfg(PhaseConfig::parse(phaseSelect)) {}

    LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                  PatternRewriter &rewriter) const override {
        // Skip the scalar epilogue clone we insert below.
        if (whileOp->hasAttr("splyce.scalar_fallback"))
            return failure();

        // Optional function-name filter.
        if (!targetFunction.empty()) {
            auto funcOp = whileOp->getParentOfType<func::FuncOp>();
            if (!funcOp || funcOp.getName() != targetFunction)
                return failure();
        }

        auto descOpt = tryMatchCoIter(whileOp);
        if (!descOpt) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] Pattern did not match\n");
            return failure();
        }
        const CoIterDescriptor &desc = *descOpt;

        Location loc = whileOp.getLoc();
        LLVM_DEBUG(llvm::dbgs()
                   << "[Splyce] Emitting SIMD fast-lane + scalar epilogue at "
                   << loc << " (W=" << vectorWidth << ")\n");

        rewriter.setInsertionPoint(whileOp);

        // SIMD fast-lane
        SimdFastLaneBuilder simdBuilder(desc, vectorWidth, rewriter, cfg);
        scf::WhileOp simdWhile = simdBuilder.build(loc);

        // Scalar epilogue
        // Clone the original while verbatim, then patch its init operands to
        // start from where the SIMD loop stopped.  The attribute prevents the
        // greedy driver from re-applying this transformation to the clone.
        Operation *epilogue = rewriter.clone(*whileOp);
        epilogue->setAttr("splyce.scalar_fallback", rewriter.getUnitAttr());
        for (unsigned i = 0, e = simdWhile.getNumResults(); i < e; ++i)
            epilogue->setOperand(i, simdWhile.getResult(i));

        // The epilogue has the same result types as the original while, so its
        // results are a drop-in replacement (e.g. the memref.store that follows
        // the original while automatically uses the epilogue's accumulator).
        rewriter.replaceOp(whileOp, epilogue->getResults());
        return success();
    }

private:
    unsigned vectorWidth;
    std::string targetFunction;
    PhaseConfig cfg;
};

// pass

namespace {

struct CoIterVectorizePass
    : public mlir::impl::CoIterVectorizeBase<CoIterVectorizePass> {

    CoIterVectorizePass() = default;

    CoIterVectorizePass(unsigned vectorWidth, std::string targetFunction,
                        std::string phaseSelect) {
        this->vectorWidth    = vectorWidth;
        this->targetFunction = std::move(targetFunction);
        this->phaseSelect    = std::move(phaseSelect);
    }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        populateCoIterVectorizePatterns(patterns, vectorWidth, targetFunction,
                                        phaseSelect);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

// public API

void mlir::populateCoIterVectorizePatterns(RewritePatternSet &patterns,
                                           unsigned vectorWidth,
                                           StringRef targetFunction,
                                           StringRef phaseSelect) {
    patterns.add<CoIterVectorizePattern>(patterns.getContext(), vectorWidth,
                                        targetFunction.str(),
                                        phaseSelect.str());
}

std::unique_ptr<Pass> mlir::createCoIterVectorizePass(unsigned vectorWidth,
                                                      StringRef targetFunction,
                                                      StringRef phaseSelect) {
    return std::make_unique<CoIterVectorizePass>(vectorWidth,
                                                 targetFunction.str(),
                                                 phaseSelect.str());
}

void mlir::registerCoIterVectorizePass() {
    PassRegistration<CoIterVectorizePass>();
}
