// CoIterVectorizePass.cpp
//
// Pass entry point.  Wires CoIterPattern (recognition) and
// CoIterVectorBuilder (emission) into the MLIR RewritePattern + Pass
// infrastructure.
//
// Flow (Option 3 - runtime density-aware dispatch):
//
//   CoIterVectorizePass::runOnOperation()
//     └─ applyPatternsGreedily(func, patterns)
//           └─ CoIterVectorizePattern::matchAndRewrite(whileOp, rewriter)
//                 ├─ tryMatchCoIter(whileOp)                // recognize
//                 ├─ isProfitable(desc, minDensity)          // static gate
//                 ├─ emit density check (nnz/dim_size)       // runtime predicate
//                 ├─ scf.if %is_dense {                      // dispatch
//                 │     VectorLoopBuilder::build(loc)        //   vector path
//                 │  } else {
//                 │     rewriter.clone(*whileOp)             //   scalar path
//                 │  }
//                 └─ rewriter.replaceOp(whileOp, ifResults)  // wire & erase

#include "Transforms/CoIterVectorizePass.h"
#include <string>
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
//  1. Matches an scf.while as an N-way co-iteration loop
//  2. Checks static profitability (minDensity gate)
//  3. Emits a runtime density check before the loop
//  4. Wraps both paths in scf.if:
//       then -> vectorized scf.while (VectorLoopBuilder)
//       else -> original scalar scf.while (cloned)
//  5. Replaces the original op with the scf.if results
struct CoIterVectorizePattern : public OpRewritePattern<scf::WhileOp> {
    CoIterVectorizePattern(MLIRContext *ctx, unsigned vectorWidth, float minDensity,
                           float runtimeDensityThreshold, std::string targetFunction,
                           PatternBenefit benefit = 1)
        : OpRewritePattern<scf::WhileOp>(ctx, benefit),
          vectorWidth(vectorWidth),
          minDensity(minDensity),
          runtimeDensityThreshold(runtimeDensityThreshold),
          targetFunction(std::move(targetFunction)) {}

    LogicalResult matchAndRewrite(scf::WhileOp whileOp, PatternRewriter &rewriter) const override {
        // Guard: skip the scalar clone we place in the else block
        if (whileOp->hasAttr("splyce.scalar_fallback"))
            return failure();

        // Function name filter
        if (!targetFunction.empty()) {
            auto funcOp = whileOp->getParentOfType<func::FuncOp>();
            if (!funcOp || funcOp.getName() != targetFunction) {
                // LLVM_DEBUG(llvm::dbgs() << "[Splyce] Skipping: '" << funcOp.getName() << "' not in target function '" << targetFunction << "'\n");
                return failure();
            }
        }

        // LLVM_DEBUG(llvm::dbgs() << "[Splyce] Matching '" << whileOp->getName() << "' in function '" << whileOp->getParentOfType<func::FuncOp>().getName() << "'\n");

        // Recognition
        auto descOpt = tryMatchCoIter(whileOp);
        if (!descOpt) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] Pattern did not match\n");
            return failure();
        }
        CoIterDescriptor &desc = *descOpt;

        // Static profitability gate (compile-time)
        if (!isProfitable(desc, minDensity)) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] Not profitable - skipping\n");
            return failure();
        }

        Location loc = whileOp.getLoc();
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Emitting density-aware dispatch at " << loc
                                << " (W=" << vectorWidth
                                << ", runtime threshold=" << runtimeDensityThreshold << ")\n");

        rewriter.setInsertionPoint(whileOp);

        // Runtime density check 
        //
        // We use the driver stream (stream 0) as the density proxy:
        //   nnz      = end_0 - init_ptr_0        (#nonzeros in the driver fiber)
        //   dim_size = memref.dim(output, 0)      (coordinate space of this level)
        //   is_dense = (nnz * 100 >= dim_size * threshold_pct)
        //
        // Integer arithmetic avoids emitting float ops in the generated IR.
        // threshold_pct is runtimeDensityThreshold * 100, rounded to nearest integer.

        Value initPtr0 = whileOp.getInits()[desc.streams[0].argIndex];
        Value end0 = desc.streams[0].end;
        Value nnz = arith::SubIOp::create(rewriter, loc, end0, initPtr0);
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] initPtr0=" << initPtr0 << "\n");
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] end0=" << end0 << "\n");
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] nnz=" << nnz << "\n");


        // The coiteration coordinate indexes the last dimension of the output
        // (e.g. dim 0 for 1-D outputs, dim 1 for the 2-D MTTKRP output).
        unsigned outRank = cast<MemRefType>(desc.outputMemref.getType()).getRank();
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] outRank=" << outRank << "\n");
        Value cCoordDim = arith::ConstantIndexOp::create(rewriter, loc, outRank - 1);
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] cCoordDim=" << cCoordDim << "\n");
        Value dimSize = memref::DimOp::create(rewriter, loc, desc.outputMemref, cCoordDim);
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] dimSize=" << dimSize << "\n");


        unsigned thresholdPct = static_cast<unsigned>(runtimeDensityThreshold * 100.0f + 0.5f);
        Value cHundred = arith::ConstantIndexOp::create(rewriter, loc, 100);
        Value cThresholdPct = arith::ConstantIndexOp::create(rewriter, loc, thresholdPct);
        Value nnzScaled = arith::MulIOp::create(rewriter, loc, nnz, cHundred);
        Value dimScaled = arith::MulIOp::create(rewriter, loc, dimSize, cThresholdPct);
        Value isDense = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge, nnzScaled, dimScaled);

        // ── scf.if dispatch ─────────────────────────────────────────────────
        //
        //   %r... = scf.if %is_dense -> (result_types...) {
        //       // Vector path: rewritten blocked while loop
        //       %vi... = scf.while ...
        //       scf.yield %vi...
        //   } else {
        //       // Scalar path: original co-iteration loop (cloned)
        //       %si... = scf.while ...
        //       scf.yield %si...
        //   }

        TypeRange resultTypes = whileOp.getResultTypes();
        auto ifOp = scf::IfOp::create(rewriter, loc, resultTypes, isDense,
                                      /*withElseRegion=*/true);

        // ── Vector path (then block) ────────────────────────────────────────
        {
            rewriter.setInsertionPointToStart(ifOp.thenBlock());
            VectorLoopBuilder vecBuilder(desc, vectorWidth, rewriter);
            scf::WhileOp vecWhile = vecBuilder.build(loc);
            scf::YieldOp::create(rewriter, loc, vecWhile.getResults());
        }

        // ── Scalar path (else block) ────────────────────────────────────────
        //
        // Clone the original scf.while verbatim. All operands (init pointers,
        // end bounds, memrefs) are defined outside the dispatch, so they remain
        // valid inside the else block.
        // The "splyce.scalar_fallback" attribute prevents the greedy rewriter
        // from re-applying this same transformation to the clone.
        {
            rewriter.setInsertionPointToStart(ifOp.elseBlock());
            Operation *scalarWhile = rewriter.clone(*whileOp);
            scalarWhile->setAttr("splyce.scalar_fallback", rewriter.getUnitAttr());
            scf::YieldOp::create(rewriter, loc, scalarWhile->getResults());
        }

        // ── Wire results and remove the original ────────────────────────────
        rewriter.replaceOp(whileOp, ifOp.getResults());
        return success();
    }

private:
    unsigned vectorWidth;
    float minDensity;
    float runtimeDensityThreshold;
    std::string targetFunction;
};

// pass implementation
namespace {

struct CoIterVectorizePass : public mlir::impl::CoIterVectorizeBase<CoIterVectorizePass> {
    // constructors: one for programmatic use (explicit args),
    // one for TableGen-generated CLI parsing (uses the base class options directly)
    CoIterVectorizePass() = default;

    CoIterVectorizePass(unsigned vectorWidth, float minDensity, bool enableTailScalarFallback,
                        float runtimeDensityThreshold, std::string targetFunction = "") {
        this->vectorWidth = vectorWidth;
        this->minDensity = minDensity;
        this->enableTailScalarFallback = enableTailScalarFallback;
        this->runtimeDensityThreshold = runtimeDensityThreshold;
        this->targetFunction = std::move(targetFunction);
    }

    void runOnOperation() override {
        Operation *op = getOperation();

        RewritePatternSet patterns(&getContext());
        populateCoIterVectorizePatterns(patterns, vectorWidth, minDensity,
                                        runtimeDensityThreshold, targetFunction);

        // use the greedy driver: keeps applying patterns until a fixed point
        // this handles nested co-iteration loops (e.g. SpGEMM outer/inner)
        if (failed(applyPatternsGreedily(op, std::move(patterns))))
            signalPassFailure();
    }
};
} // anonymous namespace

// public api

void mlir::populateCoIterVectorizePatterns(RewritePatternSet &patterns, unsigned vectorWidth,
                                           float minDensity, float runtimeDensityThreshold,
                                           StringRef targetFunction) {
    patterns.add<CoIterVectorizePattern>(patterns.getContext(), vectorWidth, minDensity,
                                        runtimeDensityThreshold, targetFunction.str());
}

std::unique_ptr<Pass> mlir::createCoIterVectorizePass(unsigned vectorWidth, float minDensity,
                                                      bool tailFallback,
                                                      float runtimeDensityThreshold,
                                                      StringRef targetFunction) {
    return std::make_unique<CoIterVectorizePass>(vectorWidth, minDensity, tailFallback,
                                                runtimeDensityThreshold, targetFunction.str());
}

void mlir::registerCoIterVectorizePass() {
    PassRegistration<CoIterVectorizePass>();
}