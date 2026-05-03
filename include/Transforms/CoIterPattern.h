#ifndef TRANSFORM_SPLYCE_PATTERN_H
#define TRANSFORM_SPLYCE_PATTERN_H

#include <optional>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace splyce {

// Describes one sparse input stream participating in the co-iteration.
// There are exactly N of these in CoIterDescriptor (one per input tensor).
struct StreamDescriptor {
    // Loop-carried index variable (block argument in the scf.while regions).
    BlockArgument iterVar;

    // Exclusive upper bound: the pointer must stay below this to be valid.
    Value end;

    // memref<?xindex> holding sorted nonzero coordinates for this stream.
    Value coordsMemref;

    // memref<?xelemTy> holding the corresponding nonzero values.
    Value valsMemref;

    // Scalar coordinate loaded in the do-block ("memref.load coordsMemref[iterVar]").
    // Captured so the scf.if condition analysis can correlate it to this stream.
    Value loadedCoord;

    // Position of iterVar in the while op's iter-arg list (0-based).
    unsigned argIndex = 0;

    void print(llvm::raw_ostream &os) const {
        os << "StreamDescriptor (argIndex=" << argIndex << "):\n"
           << "  iterVar:      " << iterVar << "\n"
           << "  end:          " << end << "\n"
           << "  coordsMemref: ";
        if (coordsMemref) os << coordsMemref; else os << "<null>";
        os << "\n  valsMemref:   ";
        if (valsMemref) os << valsMemref; else os << "<null>";
        os << "\n  loadedCoord:  ";
        if (loadedCoord) os << loadedCoord; else os << "<null>";
        os << "\n";
    }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const StreamDescriptor &sd) {
    sd.print(os);
    return os;
}

// Everything the rewriter needs to know about a matched co-iteration loop.
struct CoIterDescriptor {
    // The matched while loop.
    scf::WhileOp whileOp;

    // One StreamDescriptor per input tensor, sorted by argIndex.
    llvm::SmallVector<StreamDescriptor, 4> streams;

    // Scalar element type of the stream values (f32, f64, ...).
    Type elementType;

    // Position of the loop-carried accumulator in the while op's iter-arg list.
    // The accumulator is a float iter-arg whose value is updated by the kernel
    // (e.g. acc += a_val * b_val in SpGEMM) instead of being stored to a memref.
    // ~0u means no accumulator was found (pattern did not match).
    unsigned accArgIndex = ~0u;

    unsigned numStreams() const { return streams.size(); }

    // True when the kernel accumulates into a loop-carried float iter-arg.
    bool hasAccumulator() const { return accArgIndex != ~0u; }
};

// tryMatchCoIter
//
// Attempts to recognise the N-way co-iteration idiom inside `whileOp`.
// Returns a populated CoIterDescriptor on success, std::nullopt otherwise.
//
// Recognition criteria (all must hold):
//
//  (1) Condition block terminates with scf.condition.
//
//  (2) The condition value is a left-associative AND-tree of `cmpi ult` ops:
//        %c0  = cmpi ult, %i0, %end0
//        %c1  = cmpi ult, %i1, %end1
//        %c01 = andi %c0, %c1   ...
//      Each leaf's LHS must be a BlockArgument of index type. N = leaf count, N >= 2.
//
//  (3) The do-block has exactly N index-typed memref.loads, each indexed by
//      one of the N loop-carried stream iter vars.
//
//  (4) The do-block contains a global minimum over all N coordinates
//      (chain of arith.minui or arith.select ops covering all loads).
//
//  (5) The do-block has one scf.if whose condition is the AND of N
//      `cmpi eq, coord_k, globalMin` comparisons.
//
//  (6) The scf.if true-block contains exactly N float-typed value loads
//      (one per stream).  The kernel result is loop-carried: the scf.if
//      yields a float that appears as a non-index value in the do-block's
//      scf.yield (accumulator pattern).
//
//  (7) The do-block's scf.yield carries N index values each derivable
//      from the corresponding iter var via arith.select / arith.addi.
std::optional<CoIterDescriptor> tryMatchCoIter(scf::WhileOp whileOp);

} // namespace splyce
} // namespace mlir

#endif // TRANSFORM_SPLYCE_PATTERN_H
