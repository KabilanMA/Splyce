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
        LLVM_DEBUG(llvm::dbgs() << "[coiter] No scf.condition\n");
        return false;
    }

    llvm::SmallVector<arith::CmpIOp, 4> ultLeaves;
    collectUltLeaves(condOp.getCondition(), ultLeaves);

    if (ultLeaves.size() < 2) {
        LLVM_DEBUG(llvm::dbgs() << "[coiter] Fewer than 2 ult leaves (" << ultLeaves.size() << ")\n");
        return false;
    }

    for (auto [idx, cmp] : llvm::enumerate(ultLeaves)) {
        BlockArgument iterVar = dyn_cast<BlockArgument>(cmp.getLhs());
        if (!iterVar || !iterVar.getType().isIndex()) {
            LLVM_DEBUG(llvm::dbgs() << "[coiter] ult LHS is not an index BlockArg\n");
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

std::optional<CoIterDescriptor> mlir::splyce::tryMatchCoIter(scf::WhileOp whileOp) {
    CoIterDescriptor desc;
    desc.whileOp = whileOp;

    // need atleast 2 index-typed iter args
    unsigned indexArgCount = 0;
    for (Type t : whileOp.getOperandTypes())
        if (t.isIndex()) ++indexArgCount;
    if (indexArgCount < 2) {
        LLVM_DEBUG(llvm::dbgs() << "[coiter] Fewer than 2 index iter args\n");
        return std::nullopt;
    }

    Block &condBlock = whileOp.getBefore().front();
    Block &doBlock = whileOp.getAfter().front();

    if (!matchConditionBlock(condBlock, desc)) return std::nullopt;
    // if (!matchDoBlock(doBlock, desc)) return std::nullopt;

    // the whileOp must carry exactly two index-typed iteration varaibles.
    // it may carry more - e.g. an accumulator - but the first two must be the pointer pair.
    // TODO: extend to search all pairs.
    auto types = whileOp.getOperandTypes();
    // if (types.size() < 2 || types[0].isIndex() || !types)
}

