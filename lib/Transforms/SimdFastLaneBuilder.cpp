// SimdFastLaneBuilder.cpp
//
// Implements the W-wide SIMD fast-lane loop for the co-iteration pattern.
// See SimdFastLaneBuilder.h for the phase structure and PhaseConfig encoding.

#include "Transforms/SimdFastLaneBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

void SimdFastLaneBuilder::emitCondition(Location loc, ValueRange args,
                                         OpBuilder &condB) {
    ImplicitLocOpBuilder cb(loc, condB);
    Value cW = cb.create<arith::ConstantIndexOp>(W);
    Value cond;
    for (const StreamDescriptor &sd : desc.streams) {
        Value ptrPlusW = cb.create<arith::AddIOp>(args[sd.argIndex], cW);
        Value inBound  = cb.create<arith::CmpIOp>(arith::CmpIPredicate::ule,
                                                   ptrPlusW, sd.end);
        cond = cond ? cb.create<arith::AndIOp>(cond, inBound) : inBound;
    }
    // Forward ALL iter-args so the do-block sees them unchanged.
    cb.create<scf::ConditionOp>(cond, args);
}

// do block

void SimdFastLaneBuilder::emitBody(Location loc, ValueRange args,
                                    OpBuilder &doB) {
    ImplicitLocOpBuilder ib(loc, doB);
    const unsigned N = desc.numStreams();

    llvm::SmallVector<Value, 4> ptrs(N);
    for (unsigned s = 0; s < N; ++s)
        ptrs[s] = args[desc.streams[s].argIndex];

    // ── Phase 1: Load coordinates and values ─────────────────────────────────
    //
    // coordScalars[s][k] and valScalars[s][k] are always populated (scalar).
    // allCoords[s] / allVals[s] are only populated when vecPhase1=true.
    // bMatchVals[k]: matched B value for A lane k; 0.0 when no B coord matches.

    llvm::SmallVector<Value, 4> allCoords(N), allVals(N);
    llvm::SmallVector<llvm::SmallVector<Value, 8>, 4> coordScalars(N), valScalars(N);
    for (unsigned s = 0; s < N; ++s) {
        coordScalars[s].resize(W);
        valScalars[s].resize(W);
    }
    llvm::SmallVector<Value, 8> bMatchVals(W);

    if (cfg.vecPhase1) {
        // Zero vector for masked selects.
        Value zeroVF = ib.create<arith::ConstantOp>(
            DenseElementsAttr::get(vecF(),
                llvm::SmallVector<Attribute>(
                    W, ib.getFloatAttr(desc.elementType, 0.0))));

        // Unmasked vector loads — safe because condition guarantees >= W elements.
        for (unsigned s = 0; s < N; ++s) {
            allCoords[s] = ib.create<vector::LoadOp>(
                vecIdx(), desc.streams[s].coordsMemref, ValueRange{ptrs[s]});
            allVals[s] = ib.create<vector::LoadOp>(
                vecF(), desc.streams[s].valsMemref, ValueRange{ptrs[s]});
        }

        // Extract all scalar coordinates: all stream 0, then all stream 1, ...
        for (unsigned s = 0; s < N; ++s)
            for (unsigned k = 0; k < W; ++k)
                coordScalars[s][k] = ib.create<vector::ExtractOp>(
                    allCoords[s], ArrayRef<int64_t>{(int64_t)k});

        // W cross-comparisons: broadcast(coord[0][k]) eq coordVec[1]
        // Masked B-value selection + add-reduction per driver lane.
        for (unsigned k = 0; k < W; ++k) {
            Value aBcast = ib.create<vector::BroadcastOp>(vecIdx(),
                                                           coordScalars[0][k]);
            Value row    = ib.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                     aBcast, allCoords[1]);
            Value bSel   = ib.create<arith::SelectOp>(row, allVals[1], zeroVF);
            bMatchVals[k] = ib.create<vector::ReductionOp>(
                vector::CombiningKind::ADD, bSel);
        }
    } else {
        // Scalar Phase 1: load W coords+vals per stream at ptr+0, ptr+1, ...
        Value cstZero = ib.create<arith::ConstantOp>(
            ib.getFloatAttr(desc.elementType, 0.0));
        Value c1 = ib.create<arith::ConstantIndexOp>(1);

        for (unsigned s = 0; s < N; ++s) {
            // Chained offset: offsets[0]=ptr, offsets[k]=offsets[k-1]+1.
            llvm::SmallVector<Value, 8> offsets(W);
            offsets[0] = ptrs[s];
            for (unsigned k = 1; k < W; ++k)
                offsets[k] = ib.create<arith::AddIOp>(offsets[k - 1], c1);

            for (unsigned k = 0; k < W; ++k)
                coordScalars[s][k] = ib.create<memref::LoadOp>(
                    desc.streams[s].coordsMemref, ValueRange{offsets[k]});
            for (unsigned k = 0; k < W; ++k)
                valScalars[s][k] = ib.create<memref::LoadOp>(
                    desc.streams[s].valsMemref, ValueRange{offsets[k]});
        }

        // W×W scalar cross-comparisons: rows[k][j] = (coord[0][k] == coord[1][j])
        llvm::SmallVector<llvm::SmallVector<Value, 8>, 8> rows(W);
        for (unsigned k = 0; k < W; ++k) {
            rows[k].resize(W);
            for (unsigned j = 0; j < W; ++j)
                rows[k][j] = ib.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                        coordScalars[0][k],
                                                        coordScalars[1][j]);
        }

        // For each A lane k: select matching B vals, binary-tree addf sum.
        for (unsigned k = 0; k < W; ++k) {
            llvm::SmallVector<Value, 8> selected(W);
            for (unsigned j = 0; j < W; ++j)
                selected[j] = ib.create<arith::SelectOp>(
                    rows[k][j], valScalars[1][j], cstZero);
            bMatchVals[k] = binaryTreeSumF(loc, doB, selected);
        }
    }

    // ── Phase 2: Predicated tensor computation ────────────────────────────────
    //
    // aVals[k]: A value for lane k.  Comes from vector.extract (vecPhase1=true)
    // or from the scalar loads (vecPhase1=false).

    llvm::SmallVector<Value, 8> aVals(W);
    if (cfg.vecPhase1) {
        for (unsigned k = 0; k < W; ++k)
            aVals[k] = ib.create<vector::ExtractOp>(
                allVals[0], ArrayRef<int64_t>{(int64_t)k});
    } else {
        for (unsigned k = 0; k < W; ++k)
            aVals[k] = valScalars[0][k];
    }

    Value newAcc;
    if (cfg.vecPhase2) {
        // Pack scalar bMatchVals and aVals into vectors, multiply, reduce.
        Value bValsVec  = ib.create<vector::FromElementsOp>(vecF(),
                              ValueRange(bMatchVals));
        Value aValsVec  = ib.create<vector::FromElementsOp>(vecF(),
                              ValueRange(aVals));
        Value contribVec = ib.create<arith::MulFOp>(aValsVec, bValsVec);
        Value total      = ib.create<vector::ReductionOp>(
            vector::CombiningKind::ADD, contribVec);
        newAcc = ib.create<arith::AddFOp>(args[desc.accArgIndex], total);
    } else {
        // Scalar: W mulf, binary-tree addf, accumulate.
        llvm::SmallVector<Value, 8> contribs(W);
        for (unsigned k = 0; k < W; ++k)
            contribs[k] = ib.create<arith::MulFOp>(aVals[k], bMatchVals[k]);
        Value total = binaryTreeSumF(loc, doB, contribs);
        newAcc = ib.create<arith::AddFOp>(args[desc.accArgIndex], total);
    }

    // ── Phase 3: Branchless pointer advancement ───────────────────────────────
    //
    // pivot = min over all streams of the last (largest) coord in the window.

    Value pivot = coordScalars[0][W - 1];
    for (unsigned s = 1; s < N; ++s)
        pivot = ib.create<arith::MinUIOp>(pivot, coordScalars[s][W - 1]);

    llvm::SmallVector<Value, 4> newPtrs(N);
    if (cfg.vecPhase3) {
        // Pack coords into vectors for all streams (from_elements if P1=scalar,
        // reuse loaded vectors if P1=vector).  All packing before the broadcast
        // matches the ablation file op ordering.
        llvm::SmallVector<Value, 4> coordVecs(N);
        for (unsigned s = 0; s < N; ++s)
            coordVecs[s] = cfg.vecPhase1
                ? allCoords[s]
                : ib.create<vector::FromElementsOp>(vecIdx(),
                      ValueRange(coordScalars[s]));

        // One broadcast of the pivot, then per-stream advance.
        VectorType vecI64 = VectorType::get({W}, ib.getIntegerType(64));
        Value pivotBcast  = ib.create<vector::BroadcastOp>(vecIdx(), pivot);
        for (unsigned s = 0; s < N; ++s) {
            Value leVec   = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ule,
                                                      coordVecs[s], pivotBcast);
            Value leI64   = ib.create<arith::ExtUIOp>(vecI64, leVec);
            Value stepI64 = ib.create<vector::ReductionOp>(
                vector::CombiningKind::ADD, leI64);
            Value step    = ib.create<arith::IndexCastOp>(
                ib.getIndexType(), stepI64);
            newPtrs[s]    = ib.create<arith::AddIOp>(ptrs[s], step);
        }
    } else {
        // Scalar: W cmpi ule, W select/1/0, binary-tree addi per stream.
        Value c0 = ib.create<arith::ConstantIndexOp>(0);
        Value c1 = ib.create<arith::ConstantIndexOp>(1);
        for (unsigned s = 0; s < N; ++s) {
            llvm::SmallVector<Value, 8> les(W), advances(W);
            for (unsigned k = 0; k < W; ++k)
                les[k] = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ule,
                                                    coordScalars[s][k], pivot);
            for (unsigned k = 0; k < W; ++k)
                advances[k] = ib.create<arith::SelectOp>(les[k], c1, c0);
            Value step = binaryTreeSumIdx(loc, doB, advances);
            newPtrs[s] = ib.create<arith::AddIOp>(ptrs[s], step);
        }
    }

    // Yield: patch stream pointers and accumulator; pass other args unchanged.
    llvm::SmallVector<Value> yieldVals(args.begin(), args.end());
    for (unsigned s = 0; s < N; ++s)
        yieldVals[desc.streams[s].argIndex] = newPtrs[s];
    yieldVals[desc.accArgIndex] = newAcc;
    ib.create<scf::YieldOp>(ValueRange(yieldVals));
}

// top-level entry

scf::WhileOp SimdFastLaneBuilder::build(Location loc) {
    ImplicitLocOpBuilder ib(loc, b);

    // Non-const local copy: scf::WhileOp wraps Operation*, so this is cheap.
    scf::WhileOp origWhile = desc.whileOp;
    TypeRange  iterTypes   = origWhile.getResultTypes();
    ValueRange initVals    = origWhile.getInits();

    return ib.create<scf::WhileOp>(
        iterTypes, initVals,

        [&](OpBuilder &condB, Location condLoc, ValueRange args) {
            emitCondition(condLoc, args, condB);
        },

        [&](OpBuilder &doB, Location doLoc, ValueRange args) {
            emitBody(doLoc, args, doB);
        });
}
