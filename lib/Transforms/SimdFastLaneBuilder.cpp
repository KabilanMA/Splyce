// SimdFastLaneBuilder.cpp
//
// Implements the W-wide SIMD fast-lane loop for the co-iteration pattern.
// See SimdFastLaneBuilder.h for the overall strategy and phase ordering.

#include "Transforms/SimdFastLaneBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::splyce;

// file-local helpers

// Binary-tree reduction of N scalar f64 values using arith.addf.
// For N=4: addf(addf(v0,v1), addf(v2,v3)).
// Odd trailing element is carried as-is to the next level.
static Value binaryTreeSumF(Location loc, OpBuilder &b,
                             llvm::ArrayRef<Value> vals) {
    ImplicitLocOpBuilder ib(loc, b);
    llvm::SmallVector<Value> cur(vals.begin(), vals.end());
    while (cur.size() > 1) {
        llvm::SmallVector<Value> next;
        for (unsigned i = 0; i + 1 < cur.size(); i += 2)
            next.push_back(ib.create<arith::AddFOp>(cur[i], cur[i + 1]));
        if (cur.size() % 2 == 1)
            next.push_back(cur.back());
        cur = std::move(next);
    }
    return cur[0];
}

// Binary-tree reduction of N scalar index values using arith.addi.
// Same shape as binaryTreeSumF but for index arithmetic.
static Value binaryTreeSumIdx(Location loc, OpBuilder &b,
                               llvm::ArrayRef<Value> vals) {
    ImplicitLocOpBuilder ib(loc, b);
    llvm::SmallVector<Value> cur(vals.begin(), vals.end());
    while (cur.size() > 1) {
        llvm::SmallVector<Value> next;
        for (unsigned i = 0; i + 1 < cur.size(); i += 2)
            next.push_back(ib.create<arith::AddIOp>(cur[i], cur[i + 1]));
        if (cur.size() % 2 == 1)
            next.push_back(cur.back());
        cur = std::move(next);
    }
    return cur[0];
}

// condition block

// Condition: for each stream k, emit (ptr_k + W <= end_k), AND them all.
// The W constant is emitted here (inside the block) so it reads like vec4.
void SimdFastLaneBuilder::emitCondition(Location loc, ValueRange args,
                                         OpBuilder &condB) {
    ImplicitLocOpBuilder cb(loc, condB);
    Value cW = cb.create<arith::ConstantIndexOp>(W);
    Value cond;
    for (const StreamDescriptor &sd : desc.streams) {
        Value ptrPlusW = cb.create<arith::AddIOp>(args[sd.argIndex], cW);
        // ule: ptr + W <= end  ⟺  at least W elements remain
        Value inBound = cb.create<arith::CmpIOp>(arith::CmpIPredicate::ule,
                                                  ptrPlusW, sd.end);
        cond = cond ? cb.create<arith::AndIOp>(cond, inBound) : inBound;
    }
    // Forward ALL iter-args so the do-block sees them as its block arguments.
    cb.create<scf::ConditionOp>(cond, args);
}

// do block

void SimdFastLaneBuilder::emitBody(Location loc, ValueRange args,
                                    OpBuilder &doB) {
    ImplicitLocOpBuilder ib(loc, doB);
    const unsigned N = desc.numStreams();

    // Collect current stream pointers from block args.
    llvm::SmallVector<Value, 4> ptrs(N);
    for (unsigned s = 0; s < N; ++s)
        ptrs[s] = args[desc.streams[s].argIndex];

    // Zero vector constant — emitted inside the block (matches vec4 style).
    Value zeroVF = ib.create<arith::ConstantOp>(
        DenseElementsAttr::get(vecF(),
            llvm::SmallVector<Attribute>(
                W, ib.getFloatAttr(desc.elementType, 0.0))));

    // Phase 1: Unmasked vector loads 
    // Condition guarantees >= W elements, so no mask is needed.
    // Order: A_coords, A_vals, B_coords, B_vals (per-stream: coord then val).
    llvm::SmallVector<Value, 4> allCoords(N), allVals(N);
    for (unsigned s = 0; s < N; ++s) {
        allCoords[s] = ib.create<vector::LoadOp>(
            vecIdx(), desc.streams[s].coordsMemref, ValueRange{ptrs[s]});
        allVals[s] = ib.create<vector::LoadOp>(
            vecF(), desc.streams[s].valsMemref, ValueRange{ptrs[s]});
    }

    // Phase 2: Extract all scalar coordinates 
    // All A coords first, then all B coords (used for cross-compare and advance).
    // coordScalars[stream][lane]
    llvm::SmallVector<llvm::SmallVector<Value, 8>, 4> coordScalars(N);
    for (unsigned s = 0; s < N; ++s) {
        coordScalars[s].reserve(W);
        for (unsigned k = 0; k < W; ++k)
            coordScalars[s].push_back(ib.create<vector::ExtractOp>(
                allCoords[s], ArrayRef<int64_t>{(int64_t)k}));
    }

    // Phase 3: W×W cross-comparisons 
    // For each driver (stream 0) lane k:
    //   row_k = cmpi eq, broadcast(a_coord[k]), b_coords   → vector<W x i1>
    llvm::SmallVector<Value, 8> rows(W);
    for (unsigned k = 0; k < W; ++k) {
        Value aBcast = ib.create<vector::BroadcastOp>(vecIdx(),
                                                       coordScalars[0][k]);
        rows[k] = ib.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                            aBcast, allCoords[1]);
    }

    // Phase 4: Masked B-value selection + add-reduction per driver lane
    // select(row_k, b_vals, zero) collapses to a scalar matched B value
    // (0.0 when no element of B matches a_coord[k]; unique-coord assumption).
    llvm::SmallVector<Value, 8> bMatchVals(W);
    for (unsigned k = 0; k < W; ++k) {
        Value bSel = ib.create<arith::SelectOp>(rows[k], allVals[1], zeroVF);
        bMatchVals[k] = ib.create<vector::ReductionOp>(
            vector::CombiningKind::ADD, bSel);
    }

    // Phase 5: Extract all A values, then compute per-lane contributions
    // All extracts first, then all multiplies — matches vec4 op ordering.
    llvm::SmallVector<Value, 8> aVals(W);
    for (unsigned k = 0; k < W; ++k)
        aVals[k] = ib.create<vector::ExtractOp>(allVals[0],
                                                 ArrayRef<int64_t>{(int64_t)k});

    llvm::SmallVector<Value, 8> contribs(W);
    for (unsigned k = 0; k < W; ++k)
        contribs[k] = ib.create<arith::MulFOp>(aVals[k], bMatchVals[k]);

    // Phase 6: Binary-tree horizontal sum
    // For W=4: addf(addf(c0,c1), addf(c2,c3))  - no intermediate vector.
    Value total = binaryTreeSumF(loc, doB, contribs);

    // Phase 7: Accumulate 
    Value newAcc = ib.create<arith::AddFOp>(args[desc.accArgIndex], total);

    // Phase 8: Branchless scalar pointer advance
    // pivot = min over all streams of the last (largest) coord in the window.
    // The W >= 1 condition guarantees coordScalars[s][W-1] is valid.
    Value pivot = coordScalars[0][W - 1];
    for (unsigned s = 1; s < N; ++s)
        pivot = ib.create<arith::MinUIOp>(pivot, coordScalars[s][W - 1]);

    Value c0 = ib.create<arith::ConstantIndexOp>(0);
    Value c1 = ib.create<arith::ConstantIndexOp>(1);

    llvm::SmallVector<Value, 4> newPtrs(N);
    for (unsigned s = 0; s < N; ++s) {
        // W scalar cmpi ule: coord_k <= pivot?
        llvm::SmallVector<Value, 8> les(W);
        for (unsigned k = 0; k < W; ++k)
            les[k] = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ule,
                                               coordScalars[s][k], pivot);

        // W scalar selects: 1 if advanced, 0 otherwise
        llvm::SmallVector<Value, 8> advances(W);
        for (unsigned k = 0; k < W; ++k)
            advances[k] = ib.create<arith::SelectOp>(les[k], c1, c0);

        // Binary-tree addi to count total advance
        Value step = binaryTreeSumIdx(loc, doB, advances);
        newPtrs[s] = ib.create<arith::AddIOp>(ptrs[s], step);
    }

    // Yield
    // Start from current args (non-stream, non-acc args pass through unchanged).
    llvm::SmallVector<Value> yieldVals(args.begin(), args.end());
    for (unsigned s = 0; s < N; ++s)
        yieldVals[desc.streams[s].argIndex] = newPtrs[s];
    yieldVals[desc.accArgIndex] = newAcc;

    ib.create<scf::YieldOp>(ValueRange(yieldVals));
}

// top-level entry

scf::WhileOp SimdFastLaneBuilder::build(Location loc) {
    ImplicitLocOpBuilder ib(loc, b);

    // Use a non-const local copy: scf::WhileOp wraps Operation*, so the copy
    // is cheap and does not duplicate IR.
    scf::WhileOp origWhile = desc.whileOp;
    TypeRange  iterTypes   = origWhile.getResultTypes();
    ValueRange initVals    = origWhile.getInits();

    return ib.create<scf::WhileOp>(
        iterTypes, initVals,

        // Condition block
        [&](OpBuilder &condB, Location condLoc, ValueRange args) {
            emitCondition(condLoc, args, condB);
        },

        // Do block 
        [&](OpBuilder &doB, Location doLoc, ValueRange args) {
            emitBody(doLoc, args, doB);
        });
}
