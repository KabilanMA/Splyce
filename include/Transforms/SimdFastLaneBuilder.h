// SimdFastLaneBuilder.h
//
// Emits the SIMD fast-lane loop for the co-iteration vectorization pass.
//
// Three computation phases, each independently switchable vector/scalar via
// PhaseConfig:
//
//   Phase 1 – Loading + cross-comparison
//              V: vector.load + broadcast + cmpi eq vector + select + reduce
//              S: scalar memref.load per lane + W×W cmpi eq + select + addf tree
//
//   Phase 2 – Predicated tensor computation
//              V: from_elements + arith.mulf vector + vector.reduction
//              S: W scalar mulf + binary-tree addf
//
//   Phase 3 – Pointer advancement
//              V: (from_elements or reuse coord vector) + broadcast pivot +
//                 cmpi ule vector + extui to i64 + reduction + index_cast
//              S: W scalar cmpi ule + select/1/0 + binary-tree addi
//
// PhaseConfig "XYZ": X=Phase3, Y=Phase2, Z=Phase1; '1'=vector, '0'=scalar.

#ifndef TRANSFORMS_SPLYCE_SIMD_FAST_LANE_BUILDER_H
#define TRANSFORMS_SPLYCE_SIMD_FAST_LANE_BUILDER_H

#include "Transforms/CoIterPattern.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace splyce {

struct PhaseConfig {
    bool vecPhase1 = true;   // bit 0: Phase 1 vectorized
    bool vecPhase2 = false;  // bit 1: Phase 2 vectorized
    bool vecPhase3 = false;  // bit 2: Phase 3 vectorized

    // Parse "XYZ" where X=Phase3, Y=Phase2, Z=Phase1 ('1'=vector, '0'=scalar).
    // Default-constructed config corresponds to "001" (Phase 1 only).
    static PhaseConfig parse(llvm::StringRef s) {
        PhaseConfig cfg;
        cfg.vecPhase3 = (s.size() >= 1 && s[0] == '1');
        cfg.vecPhase2 = (s.size() >= 2 && s[1] == '1');
        cfg.vecPhase1 = (s.size() >= 3 && s[2] == '1');
        return cfg;
    }
};

class SimdFastLaneBuilder {
public:
    SimdFastLaneBuilder(const CoIterDescriptor &desc, unsigned vectorWidth,
                        OpBuilder &builder, PhaseConfig cfg = {})
        : desc(desc), W(vectorWidth), b(builder), cfg(cfg) {}

    // Emit the SIMD fast-lane scf.while at the builder's current insertion
    // point.  Returns the op so the caller can wire its results into the
    // scalar epilogue as new init values.
    scf::WhileOp build(Location loc);

private:
    const CoIterDescriptor &desc;
    unsigned W;
    OpBuilder &b;
    PhaseConfig cfg;

    VectorType vecF()   { return VectorType::get({W}, desc.elementType); }
    VectorType vecIdx() { return VectorType::get({W}, b.getIndexType()); }

    // Condition block: AND-tree of (ptr_k + W <= end_k) for each stream k.
    void emitCondition(Location loc, ValueRange args, OpBuilder &condB);

    // Do block: all three phases in the order matching the ablation files.
    void emitBody(Location loc, ValueRange args, OpBuilder &doB);
};

} // namespace splyce
} // namespace mlir

#endif // TRANSFORMS_SPLYCE_SIMD_FAST_LANE_BUILDER_H
