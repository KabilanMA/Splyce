// Recognition logic for the co-iteration idiom

#include "Transforms/CoIterPattern.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#define DEBUG_TYPE "coiter-match"

using namespace mlir;
using namespace mlir::splyce;

// recursively collect `cmpi ult` leaves from an AND-tree.
// The tree is build left-associatively by sparse lowering:
//          andi(andi(andi(comp0, cmp1), cmp2), cmp3) 
static void collectUltLeaves(Value v, llvm::SmallVectorImpl<arith::CmpIOp> &out) {
    //recursive call
    if (auto andi = v.getDefiningOp<arith::AndIOp>()) {
        collectUltLeaves(andi.getLhs(), out);
        collectUltLeaves(andi.getRhs(), out);
        return;
    }

    // base of the recursion
    if (auto cmp = v.getDefiningOp<arith::CmpIOp>())
        if (cmp.getPredicate() == arith::CmpIPredicate::ult)
            out.push_back(cmp);
};

// matchConditionBlock
// extract N StreamDescriptors - one per `cmpi ult` lead in the AND-tree.
// fills iterVar, end, and argIndex. coordsMemreg / valsMemref / loadedCoord are filled later by matchDoBlock
static bool matchConditionBlock(Block &condBlock, CoIterDescriptor &desc) {
    auto condOp = dyn_cast<scf::ConditionOp>(condBlock.getTerminator());
    if (!condOp) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] No scf.condition\n");
        return false;
    }

    llvm::SmallVector<arith::CmpIOp, 4> ultLeaves;
    collectUltLeaves(condOp.getCondition(), ultLeaves);

    if (ultLeaves.size() < 2) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Fewer than 2 ult leaves (" << ultLeaves.size() << ")\n");
        return false;
    }

    for (auto [idx, cmp] : llvm::enumerate(ultLeaves)) {
        BlockArgument iterVar = dyn_cast<BlockArgument>(cmp.getLhs());
        if (!iterVar || !iterVar.getType().isIndex()) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] ult LHS is not an index BlockArg\n");
            return false;
        }
        StreamDescriptor sd;
        sd.iterVar = iterVar;
        sd.end = cmp.getRhs();
        sd.argIndex = iterVar.getArgNumber();
        desc.streams.push_back(sd);
    }

    // sort by argIndex so streams[0] is the first iter-arg (driver convection).
    llvm::sort(desc.streams, [](const StreamDescriptor &a, const StreamDescriptor &b) {
        return a.argIndex < b.argIndex;
    });

    return true;
};

// walk addi/select chains to verify a value is derived from a specific target BlockArgument
static bool isDerivedFrom(Value v, BlockArgument targetArg) {
    // basecase
    if (v == targetArg) {
        return true;
    }

    if (auto sel = v.getDefiningOp<arith::SelectOp>()) {
        return isDerivedFrom(sel.getTrueValue(), targetArg) || isDerivedFrom(sel.getFalseValue(), targetArg);
    }

    if (auto add = v.getDefiningOp<arith::AddIOp>()) {
        return isDerivedFrom(add.getLhs(), targetArg) || isDerivedFrom(add.getRhs(), targetArg);
    }

    // derived from something else like constants
    return false;
}

// // walk addi/select chains to find a root BlockArgument 
// static BlockArgument rootArg(Value v) {
//     if (auto ba = dyn_cast<BlockArgument>(v)) {
//         return ba;
//     }
//     if (auto sel = v.getDefiningOp<arith::SelectOp>()) {
//         if (auto ba = rootArg(sel.getTrueValue())) 
//             return ba;
//     }
//     if (auto add = v.getDefiningOp<arith::AddIOp>()) {
//         if (auto ba = rootArg(add.getRhs()))
//             return ba;
//     }
//     return {};
// }

// recusrively collect all "cmpi eq" leaves from an AND-tree
static void collectEqLeaves(Value v, llvm::SmallVectorImpl<arith::CmpIOp> &out) {
    if (auto andi = v.getDefiningOp<arith::AndIOp>()) {
        collectEqLeaves(andi.getLhs(), out);
        collectEqLeaves(andi.getRhs(), out);
        return;
    }

    // basecase
    if (auto cmp = v.getDefiningOp<arith::CmpIOp>()) {
        if (cmp.getPredicate() == arith::CmpIPredicate())
            out.push_back(cmp);
    }
}

// matchDoBlock
static bool matchDoBlock(Block &doBlock, CoIterDescriptor &desc) {
    const unsigned N = desc.numStreams(); // number of input tensors

    // collect N index-typed coordinate "load"s
    // match each load to a stream by checking which iterVar indexes it
    llvm::SmallVector<memref::LoadOp, 4> coordLoads(N);
    unsigned matched = 0;

    for (auto &op : doBlock) {
        auto load = dyn_cast<memref::LoadOp>(&op);
        if (!load || !load.getType().isIndex() || load.getIndices().size() != 1) 
            continue;
        Value idx = load.getIndices()[0];
        for (auto &sd : desc.streams) {
            if (idx == sd.iterVar && !coordLoads[sd.argIndex]) {
                // argIndex might exceed N if there are non-stream iter vars
                // guard with the stream index instead
                unsigned pos = &sd - &desc.streams[0];
                coordLoads[pos] = load;
                sd.loadedCoord = load.getResult();
                sd.coordsMemref = load.getMemRef();
                ++matched;
                break;
            }
        }
    }

    if (matched != N) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Only " << matched << " of " << N << " coord loads matched\n");
        return false;
    }

    // should find the global minimum over all N loaded coordinates
    // accept any chains of arith.minui or arith.select ops that references all N coord loads
    // then walk forward and pick the last value that depends on all of them
    Value globalMin;
    llvm::SmallPtrSet<Value, 8> coordSet;
    for (auto &sd : desc.streams) 
        coordSet.insert(sd.loadedCoord);
    
    for (auto &op : doBlock) {
        if (auto minui = dyn_cast<arith::MinUIOp>(&op)) {
            if (coordSet.count(minui.getLhs()) || coordSet.count(minui.getRhs()) || coordSet.count(globalMin)) {
                globalMin = minui.getResult();
                coordSet.insert(globalMin);
            }
        } else if (auto sel = dyn_cast<arith::SelectOp>(&op)) {
            // select(cmp, a, b) where a and b are coords or prior min values
            if ((coordSet.count(sel.getTrueValue()) && coordSet.count(sel.getFalseValue())) || coordSet.count(globalMin)) {
                globalMin = sel.getResult();
                coordSet.insert(globalMin);
            }
        }
    }

    if (!globalMin) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Could not find N-way min\n");
        return false;
    }

    // then find the scf.if whose condition is AND of N "cmpi eq, coord_k, min"
    scf::IfOp matchIf;
    for (auto &op : doBlock) {
        auto ifOp = dyn_cast<scf::IfOp>(&op);
        if (!ifOp) 
            continue;
        
        llvm::SmallVector<arith::CmpIOp, 4> eqLeaves;
        collectEqLeaves(ifOp.getCondition(), eqLeaves);
        if (eqLeaves.size() != N) 
            continue;
        
        // every leaf must compare some coord_k against globalmin
        bool allMatch = llvm::all_of(eqLeaves, [&](arith::CmpIOp cmp){
            bool lhsIsCoord = coordSet.count(cmp.getLhs()) != 0;
            bool rhsIsMin = (cmp.getRhs() == globalMin);
            return lhsIsCoord && rhsIsMin;
        });

        if (allMatch) {
            matchIf = ifOp;
            break;
        }
    }

    if (!matchIf) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Could not find N-way match scf.if\n");
        return false;
    }

    // extract kernel ops from the scf.if true-block
    Block &ifBody = matchIf.getThenRegion().front();
    memref::StoreOp resultStore;

    // Map from valsMemref -> stream index, built as we find float loads
    // we identify a float load as belonging to stream k if its index operand equal stream k's iterVar
    for (auto &op : ifBody) {
        if (auto load = dyn_cast<memref::LoadOp>(&op)) {
            if (!load.getType().isIndex() && load.getIndices().size() == 1) {
                Value idx = load.getIndices()[0];
                for (auto &sd : desc.streams) {
                    if (idx == sd.iterVar && !sd.valsMemref) {
                        sd.valsMemref = load.getMemRef();
                        if (!desc.elementType)
                            desc.elementType = load.getType();
                        break;
                    }
                }
            }
        }

        if (auto st = dyn_cast<memref::StoreOp>(&op)) {
            resultStore = st;
            desc.outputMemref = st.getMemRef();
        }
        if (!isa<scf::YieldOp>(op)) {
            desc.kernelOps.push_back(&op);
        }
    }

    // verify all streams found their value memref
    for (auto &sd : desc.streams) {
        if (!sd.valsMemref) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] Stream " << sd.argIndex << " has no float value load\n");
            return false;
        }
    }
    if (!resultStore) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] No result store in if body\n");
        return false;
    }

    // verify scf.yield advances each iter var
    auto yieldOp = cast<scf::YieldOp>(doBlock.getTerminator());
    if (yieldOp.getResults().size() < N) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] yield has fewer results than streams\n");
        return false;
    }

    for (auto &sd : desc.streams) {
        Value yielded = yieldOp.getResults()[sd.argIndex];
        if (isDerivedFrom(yielded, sd.iterVar)) {
            LLVM_DEBUG(llvm::dbgs() << "[Splyce] yield[" << sd.argIndex << "] not derived from its iter var\n");
            return false;
        }
    }

    return true;
}

// public apis
std::optional<CoIterDescriptor> mlir::splyce::tryMatchCoIter(scf::WhileOp whileOp) {
    CoIterDescriptor desc;
    desc.whileOp = whileOp;

    // need atleast 2 index-typed iter args
    unsigned indexArgCount = 0;
    for (Type t : whileOp.getOperandTypes())
        if (t.isIndex()) ++indexArgCount;
    if (indexArgCount < 2) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Fewer than 2 index iter args\n");
        return std::nullopt;
    }

    Block &condBlock = whileOp.getBefore().front();
    Block &doBlock = whileOp.getAfter().front();

    if (!matchConditionBlock(condBlock, desc)) 
        return std::nullopt;
    if (!matchDoBlock(doBlock, desc)) 
        return std::nullopt;

    LLVM_DEBUG(llvm::dbgs() << "[coiter] Matched " << desc.numStreams() << "-way co-iteration\n");
    return desc;
}

bool mlir::splyce::isProfitable(const CoIterDescriptor &desc, float minDensity) {
    if (desc.estiamtedDensity >= 0.0f && desc.estiamtedDensity < minDensity) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Density below threshold\n");
        return false;
    }

    // require atleast N*2 FP-ops: N-way shuffle costs more than 2-way
    unsigned needed = desc.numStreams() * 2;
    unsigned fpOps = 0;
    for (Operation *op : desc.kernelOps) {
        if (isa<arith::MulIOp, arith::AddFOp, arith::SubFOp, arith::DivFOp>(*op))
            ++fpOps;
    }

    if (fpOps < needed) {
        LLVM_DEBUG(llvm::dbgs() << "[Splyce] Only " << fpOps << " FP ops, need " << needed << "\n");
        return false;
    }

    return true;
}

