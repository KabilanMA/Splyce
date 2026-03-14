// CoIterVectorBuilder.cpp - N-way vector loop emission

#include "Transforms/CoIterVectorBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::splyce;

// build - top-level entry point
//
// Emits the following structure:
//
//   %cW       = arith.constant W
//   %zeroIdx  = arith.constant 0 : index
//   %zeroF    = arith.constant 0.0 : f32
//   %zeroVF   = arith.constant dense<0.0> : vector<Wxf32>
//   %zeroVIdx = arith.constant dense<0>   : vector<Wxindex>
//   %vbFalse  = arith.constant dense<0>   : vector<Wxi1>
//
//   scf.while (%i = start_a, %j = start_b) {
//     %more_a = cmpi ult, %i, end_a
//     %more_b = cmpi ult, %j, end_b
//     scf.condition(%more_a & %more_b) %i, %j
//   } do {
//     [Step 1+2] validity masks
//     [Step 3]   masked loads
//     [Step 4+5] shuffle intersection
//     [Step 6]   vectorized kernel
//     [Step 7]   scatter
//     [Step 8]   frontier advance
//     scf.yield %new_i, %new_j
//   }

void VectorLoopBuilder::build(Location loc) {
    ImplicitLocOpBuilder ib(loc, b);
    const unsigned N = desc.numStreams();

    // scalar constants
    Value cW = ib.create<arith::ConstantIndexOp>(W);
    Value zeroF = ib.create<arith::ConstantOp>(ib.getFloatAttr(desc.elementType, 0.0f));

    // vector constants
    auto makeZeroVF = [&]() {
        return ib.create<arith::ConstantOp>(
                    DenseElementsAttr::get(vecF(),
                        llvm::SmallVector<Attribute>(
                            W, ib.getFloatAttr(desc.elementType, 0.0f))));
    };
    auto makeZeroVIdx = [&]() {
        return ib.create<arith::ConstantOp>(
                    DenseElementsAttr::get(vecIdx(),
                        llvm::SmallVector<Attribute>(
                            W, ib.getIndexAttr(0))));
    };
    Value zeroVF = makeZeroVF();
    Value zeroVIdx = makeZeroVIdx();

    
    // initial bounds from sparse storage positions
    // Collect initial pointer values (while op's init operands)
    llvm::SmallVector<Value, 4> initPtrs(N);
    scf::WhileOp whileOp = desc.whileOp;
    for (auto &sd : desc.streams) {
        initPtrs[sd.argIndex < N ? sd.argIndex : static_cast<unsigned>(&sd - desc.streams.data())] = whileOp.getInits()[sd.argIndex];
    }

    // build iter types: N index values
    llvm::SmallVector<Type, 4> iterTypes(N, b.getIndexType());
    llvm::SmallVector<Value, 4> initVals(N);
    for (unsigned k = 0; k < N; ++k) {
        initVals[k] = whileOp.getInits()[desc.streams[k].argIndex];
    }

    // vectorized scf.while
    ib.create<scf::WhileOp>(
        TypeRange(iterTypes),
        ValueRange(initVals),

        // condition block - all N pointers still in bounds
        [&](OpBuilder & condB, Location condLoc, Block::BlockArgListType args) {
            ImplicitLocOpBuilder cb(condLoc, condB);
            // build AND-tree of N "cmpi ult" ops
            Value cond;
            for (unsigned k = 0; k < N; ++k) {
                Value inBound = cb.create<arith::CmpIOp>(
                                    arith::CmpIPredicate::ult,
                                    args[k], desc.streams[k].end);
                cond = cond ? cb.create<arith::AndIOp>(cond, inBound) : inBound;
            }
            cb.create<scf::ConditionOp>(cond, ValueRange(args));
        },

        // do block
        [&](OpBuilder & doB, Location doLoc, Block::BlockArgListType args) {
            ImplicitLocOpBuilder db(doLoc, doB);

            // collect current pointers from block args
            llvm::SmallVector<Value, 4> ptrs(N);
            for (unsigned k = 0; k < N; ++k) {
                ptrs[k] = args[k];
            }

            // step 1+2 - validity masks for all N streams
            auto masks = emitValidityMasks(doLoc, ptrs, cW);

            // step 3 - masked loads for all N streams
            auto win = emitMaskedLoads(doLoc, ptrs, masks, zeroVIdx, zeroVF);

            // step 4+5 - N-way shuffle intersection
            auto intersect = emitShuffleIntersection(doLoc, win.coords, win.vals, masks, makeZeroVF());

            // step 6 - vectorized kernel
            Value result = emitVectorKernel(doLoc, win.vals[0], intersect.vbAligned);

            // step 7 - scatter
            emitScatter(doLoc, result, win.coords[0], intersect.matchMask);

            // step 8 - frontier-based advance for all N pointers
            auto newPtrs = emitFrontierAdvance(doLoc, ptrs, win.coords, masks);

            db.create<scf::YieldOp>(ValueRange(newPtrs));
        });
}

// Step 1+2 - Validity masks
llvm::SmallVector<Value, 4> VectorLoopBuilder::emitValidityMasks(Location loc, llvm::ArrayRef<Value> ptrs, Value cW) {
    ImplicitLocOpBuilder b(loc, this->b);
    llvm::SmallVector<Value, 4> masks;
    for (unsigned k = 0; k < desc.numStreams(); ++k) {
        Value rem = b.create<arith::SubIOp>(desc.streams[k].end, ptrs[k]);
        Value cnt = b.create<arith::MinUIOp>(rem, cW);
        masks.push_back(b.create<vector::CreateMaskOp>(vecI1(), ValueRange{cnt}));
    }
    return masks;
}

// step 3 - masked loads
VectorLoopBuilder::WindowVecs VectorLoopBuilder::emitMaskedLoads(Location loc, 
                                                                llvm::ArrayRef<Value> ptrs,
                                                                llvm::ArrayRef<Value> masks,
                                                                Value zeroIdx, 
                                                                Value zeroF) {
    ImplicitLocOpBuilder b(loc, this->b);
    WindowVecs win;
    for (unsigned k = 0; k < desc.numStreams(); ++k) {
        win.coords.push_back(
            b.create<vector::MaskedLoadOp>(vecIdx(), desc.streams[k].coordsMemref, ValueRange{ptrs[k]}, masks[k], zeroIdx)
        );
        win.vals.push_back(
            b.create<vector::MaskedLoadOp>(vecF(), desc.streams[k].valsMemref, ValueRange{ptrs[k]}, masks[k], zeroF)
        );
    }

    return win;
}

// Step 4+5: N-way shuffle intersection
//
// For driver lane k and probe stream p:
//   eq_p    = cmpi eq, broadcast(coords[0][k]), coords[p]  — vector<W x i1>
//   eq_p_v  = andi eq_p, masks[p]
//   hit_p_k = OR-reduce(eq_p_v)                            — i1
//   sel_p   = select(eq_p_v, vals[p], 0.0)
//   val_p_k = add-reduce(sel_p)                            — scalar
//
// hit_k        = AND over all p of hit_p_k
// match_mask[k]= hit_k ∧ masks[0][k]
// vbAligned[p][k] = val_p_k
VectorLoopBuilder::IntersectionResult VectorLoopBuilder::emitShuffleIntersection(Location loc,
                                                                                llvm::ArrayRef<Value> coords,
                                                                                llvm::ArrayRef<Value> vals,
                                                                                llvm::ArrayRef<Value> masks,
                                                                                Value zeroVF) {                                           
    ImplicitLocOpBuilder b(loc, this->b);
    const unsigned N = desc.numStreams();
    
    // accumulator vectors, built lane by lane
    Value allFalse = b.create<arith::ConstantOp>(
        DenseElementsAttr::get(vecI1(), llvm::SmallVector<Attribute>(W, b.getBoolAttr(false)))
    );

    Value hitVec = allFalse;

    // One accumulator per probe stream (index 1..N-1)
    llvm::SmallVector<Value, 4> vbVecs(N-1, zeroVF);

    for (unsigned k = 0; k < W; ++k) {
        // extract driver coordinates at lane k
        Value driverCoord = b.create<vector::ExtractOp>(coords[0], ArrayRef<int64_t>{(int64_t)k});
        Value driverBcast = b.create<vector::BroadcastOp>(vecIdx(), driverCoord);

        // check all probe streams
        Value laneHit;  // i1 - AND of probe hits for this lane
        for (unsigned p = 1; p < N; ++p) {
            // compare broadcast driver coord against all W probe coords
            Value eq = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, driverBcast, coords[p]);
            Value eqMasked = b.create<arith::AndIOp>(eq, masks[p]);

            // hit from this probe: OR-reduce
            Value hitP = b.create<vector::ReductionOp>(vector::CombiningKind::OR, eqMasked);

            // Gather matching value: select then add-reduce (unique coords)
            Value sel = b.create<arith::SelectOp>(eqMasked, vals[p], zeroVF);
            Value valPk = b.create<vector::ReductionOp>(vector::CombiningKind::ADD, sel);

            // AND into lane hit
            laneHit = laneHit ? b.create<arith::AndIOp>(laneHit, hitP) : hitP;
            
            // Insert val into probe's accumulator vector
            vbVecs[p-1] = b.create<vector::InsertOp>(
                valPk, vbVecs[p-1], ArrayRef<int64_t>{(int64_t)k}
            );
        }

        // insert lane hit into hit vector
        hitVec = b.create<vector::InsertOp>(laneHit, hitVec, ArrayRef<int64_t>{(int64_t)k});
    }
    
    // suppress out-of-bounds driver lanes
    Value matchMask = b.create<arith::AndIOp>(hitVec, masks[0]);
    return {matchMask, vbVecs};
}

// Step 6: Vectorized compute kernel
//
// Walk kernelOps, maintaining a scalarToVector substitution table.
// Seed: scalar value loads for stream k -> the corresponding vector.
// For every arith op, clone with vector<W x elemTy> operands/results.
Value VectorLoopBuilder::emitVectorKernel(Location loc, Value vaVec, llvm::ArrayRef<Value> vbAligned) {
    ImplicitLocOpBuilder b(loc, this->b);
    const unsigned N = desc.numStreams();

    DenseMap<Value, Value> scalarToVector;

    // seed the table: find scalar value loads for each stream in kernelOps
    for (Operation *op : desc.kernelOps) {
        auto ld = dyn_cast<memref::LoadOp>(op);
        if (!ld || ld.getType().isIndex())
            continue;
        for (unsigned k = 0; k < N; ++k) {
            if (ld.getMemRef() == desc.streams[k].valsMemref) {
                Value vecVal = (k == 0) ? vaVec : vbAligned[k-1];
                scalarToVector[ld.getResult()] = vecVal;
                break;
            }
        }
    }

    Value lastResult = vaVec;

    for (Operation *op : desc.kernelOps) {
        // skip loads (seeded above) and stores (handled by scatter)
        if (isa<memref::LoadOp, memref::StoreOp>(op))
            continue;
        
        SmallVector<Value> newOperands;
        bool allMapped = true;
        for (Value operand : op->getOperands()) {
            auto it = scalarToVector.find(operand);
            if (it != scalarToVector.end()) {
                newOperands.push_back(it->second);
            } else if (isa<FloatType>(operand.getType())) {
                // Broadcasr scalar constants into a vector
                newOperands.push_back(b.create<vector::BroadcastOp>(vecF(), operand));
            } else {
                allMapped = false;
                break;
            }
        }
        if (!allMapped)
            continue;
        
        // clone op with vector result type
        OperationState state(loc, op->getName());
        state.addOperands(newOperands);
        state.addTypes(vecF());
        state.addAttributes(op->getAttrs());
        Operation *vecOp = b.create(state);
        if (vecOp->getNumResults() == 1) {
            scalarToVector[op->getResult(0)] = vecOp->getResult(0);
            lastResult = vecOp->getResult(0);
        }
    }

    return lastResult;
}

// step 7: masked scatter
void VectorLoopBuilder::emitScatter(Location loc, Value result, Value driverCoords, Value matchMask) {
    ImplicitLocOpBuilder b(loc, this->b);
    Value base = b.create<arith::ConstantIndexOp>(0);
    b.create<vector::ScatterOp>(desc.outputMemref, ValueRange{base}, driverCoords, matchMask, result);
    return;
}

// step 8: N-way frontier advance
//  frontier  = min over all streams of umax(valid coords in window_k)
//  adv_k     = popcount(coords_k[i] <= frontier ∧ mask_k)
//  new_ptr_k = ptr_k + adv_k
llvm::SmallVector<Value, 4> VectorLoopBuilder::emitFrontierAdvance(Location loc,
                                                                  llvm::ArrayRef<Value> ptrs,
                                                                  llvm::ArrayRef<Value> coords,
                                                                  llvm::ArrayRef<Value> masks) {
    
    ImplicitLocOpBuilder b(loc, this->b);
    const unsigned N = desc.numStreams();

    // compute max valid coordinate in each stream's window
    // invalid lanes have coord 0 (from masked load), so umax is safe as long as at least one lane is valid - guaranteed by the while condition
    Value frontier;
    for (unsigned k = 0; k < N; ++k) {
        Value maxK = b.create<vector::ReductionOp>(vector::CombiningKind::MAXUI, coords[k]);
        frontier = frontier ? b.create<arith::MinUIOp>(frontier, maxK) : maxK;
    }

    Value frontierV = b.create<vector::BroadcastOp>(vecIdx(), frontier);

    // for each stream: mask the elements at or before the frontier
    llvm::SmallVector<Value, 4> newPtrs(N);
    for (unsigned k = 0; k < N; ++k) {
        Value le = b.create<arith::CmpIOp>(arith::CmpIPredicate::ule, coords[k], frontierV);
        Value advMask = b.create<arith::AndIOp>(le, masks[k]);
        Value advVec = b.create<arith::ExtUIOp>(vecIdx(), advMask);
        Value adv = b.create<vector::ReductionOp>(vector::CombiningKind::ADD, advVec);
        newPtrs[k] = b.create<arith::AddIOp>(ptrs[k], adv);
    }
    
    return newPtrs;
}