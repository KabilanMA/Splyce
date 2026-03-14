#ifndef TRANSFORM_SPLYCE_PATTERN_H
#define TRANSFORM_SPLYCE_PATTERN_H

#include <optional>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace splyce {


/**
 * @brief Describes one sparse input stream participating in the co-iteration. There are exactly N of these in the CoIterDescriptor (one per input tensor).
 */
struct StreamDescriptor {
    // Loop-carried index variable (block argument in the scf.while before/after regions)
    BlockArgument iterVar;

    // exclusive upper bound: the pointer must stay below "end" to be valid.
    Value end;

    // The memref<?xindex> holding sorted nonzero coordinates for this stream.
    Value coordsMemref;

    // The memref<?xelemTy> holding corresponding nonzero values.
    Value valsMemref;

    // The scalar coordinates loaded in the do-block this iteration ("memref.load coordsMemref[iterVar]")
    // Captured during matching so the scf.if condition analysis can correlated it to this stream.
    Value loadedCoord;

    // Position of iterVar in the while op's iter-arg list (0-based)
    // used to correlated condition-block args with do-block args.
    unsigned argIndex = 0;
};

/**
 * @brief Everything the rewriter needs to know about a matched co-iteration loop.
 * 
 */
struct CoIterDescriptor {

    // the while loop itself
    scf::WhileOp whileOp;

    // N input stream
    llvm::SmallVector<StreamDescriptor, 4> streams;

    // output memref
    // detected from the memref.store inside the scf.if true-block
    Value outputMemref;

    // compute kernel
    // all ops inside the scf.if true-block in order.
    // Includes N value loads (one per stream) and the final store.
    llvm::SmallVector<Operation *, 8> kernelOps;

    // element type
    // scalar type of stream values (f32, f64, i32, ...)
    Type elementType;

    // estimated intersection density
    // fraction of driver coordinates expected to match ALL probe streams
    // -1.0 = unknown (always transform)
    float estiamtedDensity = -1.0f;


    unsigned numStreams() const {
        return streams.size();
    }

};

// tryMatchCoIter
//
// Attempts to recognise the N-way co-iteration idiom in `whileOp`.
// Returns a populated CoIterDescriptor on success, std::nullopt otherwise.
//
// Recognition criteria (all must hold):
//
//  (1) Condition block terminates with scf.condition.
//
//  (2) The condition value is a left-associative AND-tree of `cmpi ult` ops:
//        %c0  = cmpi ult, %i0, %end0
//        %c1  = cmpi ult, %i1, %end1
//        %c01 = andi %c0, %c1
//        %c2  = cmpi ult, %i2, %end2
//        %c012= andi %c01, %c2  ...
//      Each leaf's LHS must be a BlockArgument. N = leaf count. N >= 2.
//
//  (3) The do-block has exactly N index-typed memref.loads, each indexed
//      by one of the N loop-carried iter vars.
//
//  (4) The do-block contains a global minimum over all N coordinates
//      (chain of arith.minui or arith.select ops covering all loads).
//
//  (5) The do-block has one scf.if whose condition is the AND of N
//      `cmpi eq, coord_k, globalMin` comparisons.
//
//  (6) The scf.if true-block contains exactly N float-typed value loads
//      (one per stream) and exactly one memref.store.
//
//  (7) The do-block's scf.yield carries N index values each derivable
//      from the corresponding iter var via arith.select / arith.addi.
std::optional<CoIterDescriptor> tryMatchCoIter(scf::WhileOp whileOp);


 
// isProfitable
//
//   1. estimatedDensity >= minDensity
//   2. At least (N * 2) FP ops in kernelOps to amortise N-way shuffle cost
bool isProfitable(const CoIterDescriptor &desc, float minDensity);

}   // MLIR
}   // Splyce

#endif //TRANSFORM_SPLYCE_PATTERN_H

