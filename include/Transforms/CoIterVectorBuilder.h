// CoIterVectorBuilder.h
// Emits the vectorized N-way co-iteration loop in Vector dialect.
//
// The builder is parameterised over N (number of streams) and W (SIMD width) so neither is baked in at compile time.
// Given a CoIterDescriptor, emits the vectorized replacement loop using mlir::OpBuilder.  
// Separated from the pass and pattern classes so it can be tested in isolation (and later extended with SVE / RVV support).

#ifndef TRANSFORMS_SPLYCE_VECTOR_BUILDER_H
#define TRANSFORMS_SPLYCE_VECTOR_BUILDER_H

#include "Transforms/CoIterPattern.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace splyce {

// VectorLoopBuilder
//
// Emits the complete vectorized replacement for one matched co-iteration loop.
//
// The general N-way intersection strategy:
//   Driver = streams[0]. Its W coordinates are broadcast one lane at a time.
//   Probes = streams[1..N-1]. For each driver lane k:
//               hit_k = AND over all probe p of
//                         OR-reduce(broadcast(ca[k]) == cb_p ∧ mask_p)
//               val_p_k = matched value from probe p (0 if no match)
//   match_mask[k] = hit_k ∧ mask_driver[k]
//   For the kernel, va = driver values, vb_p = probe p aligned values.

class VectorLoopBuilder {

public:
    VectorLoopBuilder(const CoIterDescriptor &desc, unsigned vectorWidth, OpBuilder &builder)
                    : desc(desc), W(vectorWidth), b(builder) {}

  // Emit the complete vectorized scf.while loop at `builder`'s current insertion point.  
  // Does not erase the original - the caller does that.
  void build(Location loc);

private:
    const CoIterDescriptor &desc;
    unsigned W;
    OpBuilder &b;

    // Vector type helpers
    VectorType vecF() { 
        return VectorType::get({W}, desc.elementType); 
    }

    VectorType vecIdx() { 
        return VectorType::get({W}, b.getIndexType()); 
    }

    VectorType vecI1() {
        return VectorType::get({W}, b.getI1Type());
    }

    // Emission helpers (each corresponds to one numbered step)

    // Step 1+2: ompute validity masks for every stream's current window.
    //   masks[k] = create_mask(min(end_k - ptr_k, W))
    llvm::SmallVector<Value, 4> emitValidityMasks(Location loc, llvm::ArrayRef<Value> ptrs, Value cW);

    // Step 3: Masked load of coordinates and values for every stream.
    struct WindowVecs {
        llvm::SmallVector<Value, 4> coords;    // N X vector<W x index> - coordinates
        llvm::SmallVector<Value, 4> vals;    // N x vector<W x elemTy> - values
    };

    WindowVecs emitMaskedLoads(Location loc,
                                llvm::ArrayRef<Value> ptrs,
                                llvm::ArrayRef<Value> masks,
                                Value zeroIdx, 
                                Value zeroF);

    // Step 4+5: N-way shuffle-based intersection.
    //
    //    For each driver lane k (0..W-1):
    //      For each probe stream p (1..N-1):
    //          eq_p    = cmpi eq, broadcast(ca[k]), coords_p   (vector<W x i1>)
    //          eq_p_v  = andi eq_p, mask_p
    //          hit_p_k = OR-reduce(eq_p_v)
    //          sel_p   = select(eq_p_v, vals_p, 0.0)
    //          val_p_k = add-reduce(sel_p)          ← unique by coord uniqueness
    //      hit_k        = AND of all hit_p_k
    //      val_p_vec[k] = val_p_k  (for each p)
    //    match_mask = AND(hit_vec, mask_driver)
    //
    // Returns the match mask and one aligned-value vector per probe stream.
    struct IntersectionResult {
        Value matchMask;  // vector<W x i1>
        llvm::SmallVector<Value, 4> vbAligned;  // (N-1) x vector<W x elemTy>
    };
    IntersectionResult emitShuffleIntersection(Location loc,
                                                llvm::ArrayRef<Value> coords,
                                                llvm::ArrayRef<Value> vals,
                                                llvm::ArrayRef<Value> masks,
                                                Value zeroVF);

    // Step 6: Clone the compute kernel with vector operand types.
    //  vaVec = driver values (vector<W x elemTy>)
    //  vbAligned = one aligned-value vector per probe stream
    Value emitVectorKernel(Location loc, 
                            Value vaVec, llvm::ArrayRef<Value> vbAligned);

    // Step 7: Masked scatter of results to the output memref.
    //   vector.scatter %output[0], %ca, %matchMask, %result
    void emitScatter(Location loc,
                    Value result, Value driverCoords, Value matchMask);

    // Step 8: Frontier-based N-way pointer advance.
    //  frontier = min over all streams of umax(valid coords in window)
    //  adv_k    = popcount(coords_k[i] <= frontier ∧ mask_k)
    //  new_ptr_k = ptr_k + adv_k
    llvm::SmallVector<Value, 4> emitFrontierAdvance(Location loc,
                                                    llvm::ArrayRef<Value> ptrs,
                                                    llvm::ArrayRef<Value> coords,
                                                    llvm::ArrayRef<Value> masks);
};

} // namespace coiter
} // namespace mlir

#endif // TRANSFORMS_COITER_VECTOR_BUILDER_H