// SimdFastLaneBuilder.h
//
// Emits the SIMD fast-lane loop that precedes the scalar epilogue in the
// SpGEMM co-iteration transformation.
//
// Generated body structure (mirrors spgemm_vec4.mlir):
//
//   Phase 1 – Unmasked vector.load for each stream (coord then val).
//   Phase 2 – Extract all scalar coordinates (all A, then all B).
//   Phase 3 – W cross-comparisons: broadcast(a_coord[k]) == b_coords.
//   Phase 4 – Masked B-value selection + add-reduction per driver lane.
//   Phase 5 – Extract all A scalar values, then compute W contributions.
//   Phase 6 – Binary-tree addf reduction to a single total.
//   Phase 7 – Accumulate: new_acc = old_acc + total.
//   Phase 8 – Branchless scalar advance:
//                pivot = minui(a_coord[W-1], b_coord[W-1])
//                for each stream: W scalar cmpi ule, W scalar select/1/0,
//                                 binary-tree addi, ptr += step.

#ifndef TRANSFORMS_SPLYCE_SIMD_FAST_LANE_BUILDER_H
#define TRANSFORMS_SPLYCE_SIMD_FAST_LANE_BUILDER_H

#include "Transforms/CoIterPattern.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace splyce {

class SimdFastLaneBuilder {
public:
    SimdFastLaneBuilder(const CoIterDescriptor &desc, unsigned vectorWidth,
                        OpBuilder &builder)
        : desc(desc), W(vectorWidth), b(builder) {}

    // Emit the SIMD fast-lane scf.while at the builder's current insertion
    // point. Returns the created op so the caller can feed its results into
    // the scalar epilogue as new init values.
    scf::WhileOp build(Location loc);

private:
    const CoIterDescriptor &desc;
    unsigned W;
    OpBuilder &b;

    VectorType vecF()   { return VectorType::get({W}, desc.elementType); }
    VectorType vecIdx() { return VectorType::get({W}, b.getIndexType()); }

    // Condition block: AND-tree of (ptr_k + W <= end_k) for each stream k.
    // The W constant is emitted inside the block.
    void emitCondition(Location loc, ValueRange args, OpBuilder &condB);

    // Do block: all eight phases in the order matching spgemm_vec4.mlir.
    void emitBody(Location loc, ValueRange args, OpBuilder &doB);
};

} // namespace splyce
} // namespace mlir

#endif // TRANSFORMS_SPLYCE_SIMD_FAST_LANE_BUILDER_H
