// CoIterPattern.cpp - Recognition logic for the N-way co-iteration idiom.

#include "Transforms/CoIterPattern.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#define DEBUG_TYPE "splyce-vectorize"

using namespace mlir;
using namespace mlir::splyce;

// helpers

// Recursively collect `cmpi ult` leaves from a left-associative AND-tree:
//   andi(andi(andi(cmp0, cmp1), cmp2), cmp3)
static void collectUltLeaves(Value v,
                              llvm::SmallVectorImpl<arith::CmpIOp> &out) {
    if (auto andi = v.getDefiningOp<arith::AndIOp>()) {
        collectUltLeaves(andi.getLhs(), out);
        collectUltLeaves(andi.getRhs(), out);
        return;
    }
    if (auto cmp = v.getDefiningOp<arith::CmpIOp>())
        if (cmp.getPredicate() == arith::CmpIPredicate::ult)
            out.push_back(cmp);
}

// Recursively collect `cmpi eq` leaves from an AND-tree.
static void collectEqLeaves(Value v,
                             llvm::SmallVectorImpl<arith::CmpIOp> &out) {
    if (auto andi = v.getDefiningOp<arith::AndIOp>()) {
        collectEqLeaves(andi.getLhs(), out);
        collectEqLeaves(andi.getRhs(), out);
        return;
    }
    if (auto cmp = v.getDefiningOp<arith::CmpIOp>())
        if (cmp.getPredicate() == arith::CmpIPredicate::eq)
            out.push_back(cmp);
}

// Walk arith.addi / arith.select chains to verify a value is derived from
// a specific BlockArgument.
static bool isDerivedFrom(Value v, BlockArgument target) {
    if (v == target)
        return true;
    if (auto sel = v.getDefiningOp<arith::SelectOp>())
        return isDerivedFrom(sel.getTrueValue(), target) ||
               isDerivedFrom(sel.getFalseValue(), target);
    if (auto add = v.getDefiningOp<arith::AddIOp>())
        return isDerivedFrom(add.getLhs(), target) ||
               isDerivedFrom(add.getRhs(), target);
    return false;
}

// condition block

// Extract N StreamDescriptors — one per `cmpi ult` leaf in the AND-tree.
// Fills iterVar, end, and argIndex; coordsMemref / valsMemref / loadedCoord
// are filled later by matchDoBlock.
static bool matchConditionBlock(Block &condBlock, CoIterDescriptor &desc) {
    auto condOp = dyn_cast<scf::ConditionOp>(condBlock.getTerminator());
    if (!condOp) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] No scf.condition\n");
        return false;
    }

    llvm::SmallVector<arith::CmpIOp, 4> ultLeaves;
    collectUltLeaves(condOp.getCondition(), ultLeaves);

    if (ultLeaves.size() < 2) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Fewer than 2 ult leaves ("
                                << ultLeaves.size() << ")\n");
        return false;
    }

    for (auto [idx, cmp] : llvm::enumerate(ultLeaves)) {
        BlockArgument iterVar = dyn_cast<BlockArgument>(cmp.getLhs());
        if (!iterVar || !iterVar.getType().isIndex()) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] ult LHS is not an index BlockArg\n");
            return false;
        }
        StreamDescriptor sd;
        sd.iterVar   = iterVar;
        sd.end       = cmp.getRhs();
        sd.argIndex  = iterVar.getArgNumber();
        desc.streams.push_back(sd);
    }

    // Sort by argIndex so streams[0] is the first iter-arg (driver convention).
    llvm::sort(desc.streams, [](const StreamDescriptor &a,
                                const StreamDescriptor &b) {
        return a.argIndex < b.argIndex;
    });
    return true;
}

// do block 

static bool matchDoBlock(Block &doBlock, CoIterDescriptor &desc) {
    const unsigned N = desc.numStreams();

    // (3) Find N index-typed coordinate loads, one per stream
    llvm::SmallVector<memref::LoadOp, 4> coordLoads(N);
    unsigned matched = 0;
    for (auto &op : doBlock) {
        auto load = dyn_cast<memref::LoadOp>(&op);
        if (!load || !load.getType().isIndex() || load.getIndices().size() != 1)
            continue;
        Value idx = load.getIndices()[0];
        for (StreamDescriptor &sd : desc.streams) {
            unsigned pos = static_cast<unsigned>(&sd - desc.streams.data());
            Value expectedArg = doBlock.getArgument(sd.argIndex);
            if (idx == expectedArg && !coordLoads[pos]) {
                coordLoads[pos]  = load;
                sd.loadedCoord   = load.getResult();
                sd.coordsMemref  = load.getMemRef();
                ++matched;
                LLVM_DEBUG(llvm::dbgs() << "[Splyce] Coord load matched: " << sd);
                break;
            }
        }
    }
    if (matched != N) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Only " << matched << " of " << N
                                << " coord loads matched\n");
        return false;
    }

    // (4) Find the global minimum over all N loaded coordinates
    // Accept chains of arith.minui or arith.select that reference all N coords.
    // Stop at the first scf.if to avoid confusing pointer-advance selects.
    Value globalMin;
    llvm::SmallPtrSet<Value, 8> coordSet;
    for (auto &sd : desc.streams)
        coordSet.insert(sd.loadedCoord);

    for (auto &op : doBlock) {
        if (isa<scf::IfOp>(&op))
            break;
        if (auto minui = dyn_cast<arith::MinUIOp>(&op)) {
            if (coordSet.count(minui.getLhs()) || coordSet.count(minui.getRhs()) ||
                coordSet.count(globalMin)) {
                globalMin = minui.getResult();
                coordSet.insert(globalMin);
            }
        } else if (auto sel = dyn_cast<arith::SelectOp>(&op)) {
            if ((coordSet.count(sel.getTrueValue()) &&
                 coordSet.count(sel.getFalseValue())) ||
                coordSet.count(globalMin)) {
                globalMin = sel.getResult();
                coordSet.insert(globalMin);
            }
        }
    }
    if (!globalMin) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Could not find N-way min\n");
        return false;
    }

    // (5) Find the scf.if whose condition is AND of N "cmpi eq, coord, min" 
    scf::IfOp matchIf;
    for (auto &op : doBlock) {
        auto ifOp = dyn_cast<scf::IfOp>(&op);
        if (!ifOp)
            continue;

        llvm::SmallVector<arith::CmpIOp, 4> eqLeaves;
        collectEqLeaves(ifOp.getCondition(), eqLeaves);
        if (eqLeaves.size() != N)
            continue;

        bool allMatch = llvm::all_of(eqLeaves, [&](arith::CmpIOp cmp) {
            return coordSet.count(cmp.getLhs()) != 0 && cmp.getRhs() == globalMin;
        });
        if (allMatch) {
            matchIf = ifOp;
            break;
        }
    }
    if (!matchIf) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Could not find N-way intersection scf.if\n");
        return false;
    }

    // (6) Find value loads in the scf.if true-block 
    // Each stream contributes one float-typed load indexed by its do-block arg.
    Block &ifBody = matchIf.getThenRegion().front();
    for (auto &op : ifBody) {
        auto load = dyn_cast<memref::LoadOp>(&op);
        if (!load || load.getType().isIndex() || load.getIndices().size() != 1)
            continue;
        Value idx = load.getIndices()[0];
        for (auto &sd : desc.streams) {
            if (idx == doBlock.getArgument(sd.argIndex) && !sd.valsMemref) {
                sd.valsMemref = load.getMemRef();
                if (!desc.elementType)
                    desc.elementType = load.getType();
                break;
            }
        }
    }
    for (auto &sd : desc.streams) {
        if (!sd.valsMemref) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] Stream " << sd.argIndex
                                    << " has no float value load\n");
            return false;
        }
    }

    // (6 cont.) Detect loop-carried accumulator
    // The scf.if must yield a float result that appears as a non-index value
    // in the do-block's scf.yield (acc += kernel(...) pattern).
    if (matchIf.getNumResults() == 0) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] scf.if has no results (expected accumulator)\n");
        return false;
    }
    auto yieldOp = cast<scf::YieldOp>(doBlock.getTerminator());
    bool foundAcc = false;
    for (auto [yieldIdx, yieldedVal] : llvm::enumerate(yieldOp.getResults())) {
        if (!yieldedVal.getType().isIndex() &&
            yieldedVal == matchIf.getResult(0)) {
            desc.accArgIndex = static_cast<unsigned>(yieldIdx);
            foundAcc = true;
            break;
        }
    }
    if (!foundAcc) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] No loop-carried accumulator found\n");
        return false;
    }
    LLVM_DEBUG(llvm::dbgs() << "[Splyce] Detected loop-carried accumulator at argIndex="
                            << desc.accArgIndex << "\n");

    // (7) Verify the scf.yield advances each stream iter var
    if (yieldOp.getResults().size() < N) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] yield has fewer results than streams\n");
        return false;
    }
    for (auto &sd : desc.streams) {
        Value yielded = yieldOp.getResults()[sd.argIndex];
        if (!isDerivedFrom(yielded, doBlock.getArgument(sd.argIndex))) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] yield[" << sd.argIndex
                                    << "] not derived from its iter var\n");
            return false;
        }
    }
    return true;
}

// public API

std::optional<CoIterDescriptor> mlir::splyce::tryMatchCoIter(
    scf::WhileOp whileOp) {

    CoIterDescriptor desc;
    desc.whileOp = whileOp;

    // Require at least 2 index-typed iter args (the stream pointers).
    unsigned indexArgCount = 0;
    for (Type t : whileOp.getOperandTypes())
        if (t.isIndex()) ++indexArgCount;
    if (indexArgCount < 2) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Fewer than 2 index iter args\n");
        return std::nullopt;
    }

    // Only vectorize the innermost co-iteration; skip if a nested loop exists.
    bool hasNestedLoop = false;
    for (Region &region : whileOp->getRegions()) {
        region.walk([&](Operation *op) {
            if (isa<scf::WhileOp, scf::ForOp>(op))
                hasNestedLoop = true;
        });
        if (hasNestedLoop) break;
    }
    if (hasNestedLoop) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Not innermost loop - skipping\n");
        return std::nullopt;
    }

    Block &condBlock = whileOp.getBefore().front();
    Block &doBlock   = whileOp.getAfter().front();

    if (!matchConditionBlock(condBlock, desc)) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Could not match condition block\n");
        return std::nullopt;
    }
    if (!matchDoBlock(doBlock, desc)) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Could not match do block\n");
        return std::nullopt;
    }

    LLVM_DEBUG(llvm::dbgs() << "[Splyce] Matched " << desc.numStreams()
                            << "-way co-iteration\n");
    return desc;
}
