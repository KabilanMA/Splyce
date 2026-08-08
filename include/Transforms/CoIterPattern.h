// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

    // SSA result of this stream's float value load in the scf.if true-block
    // ("%v = memref.load valsMemref[iterVar]"). Used to classify multiply-
    // chain leaves as stream values vs. an extra loop-invariant scalar
    // factor (see CoIterDescriptor::extraScalarFactor).
    Value loadedVal;

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

    // True when the kernel scatters its result directly into an output
    // memref (e.g. A[coord, j] += ...) instead of threading a loop-carried
    // scalar accumulator through the while loop — the shape produced when
    // the co-iterated dimension is a parallel/output index rather than a
    // reduction index (e.g. SpMMH). Mutually exclusive with
    // hasAccumulator().
    bool isScatterPattern = false;

    // For scatter patterns: the output memref written by the scf.if body,
    // and the "free" index (not one of this loop's iter-args - e.g. an
    // enclosing scf.for's induction variable) used as its second subscript
    // alongside the matched (driver-stream) coordinate.
    Value outputMemref;
    Value outputFreeIndex;

    // An extra loop-invariant scalar factor multiplied into the per-match
    // term alongside the N stream values — present in >2-operand
    // contractions where one operand doesn't vary with this co-iteration
    // (e.g. SpMTTKRP's B[i,k,l], SpMMH's C[k,j]). Null for a plain
    // 2-factor product like SpGEMM's B[i,j]*C[j,k].
    Value extraScalarFactor;

    unsigned numStreams() const { return streams.size(); }

    // True when the kernel accumulates into a loop-carried float iter-arg.
    bool hasAccumulator() const { return accArgIndex != ~0u; }

    // True when an extra loop-invariant scalar factor was detected.
    bool hasExtraScalarFactor() const { return static_cast<bool>(extraScalarFactor); }
};

// Recognizes the N-way co-iteration idiom inside `whileOp` and, on success,
// returns a CoIterDescriptor describing it (nullopt if it doesn't match).
//
// Looks for: an AND-tree of `cmpi ult` range checks in the condition block,
// one per stream (N >= 2), each comparing an index-typed BlockArgument
// against an upper bound; N matching coordinate loads in the do-block, plus
// a global min over them (a chain of arith.minui/arith.select); an scf.if
// gated on all N coords == that min; N float loads inside the scf.if
// feeding either a loop-carried accumulator (the scf.if yields a float that
// shows up in the do-block's scf.yield) or a direct scatter-store into an
// output memref (the scf.if has no results, and its body does a
// load/accumulate/store round-trip instead) - exactly one of
// hasAccumulator() / isScatterPattern ends up set. Either way, the
// per-match term must be a left-associative arith.mulf tree whose leaves
// are exactly the N stream values plus at most one extra loop-invariant
// scalar (desc.extraScalarFactor, e.g. a third contraction operand that
// doesn't vary with this co-iteration). Finally, the do-block's scf.yield
// must advance each stream's iter var via arith.select/arith.addi.
//
// See matchConditionBlock/matchDoBlock for the actual matching logic.
std::optional<CoIterDescriptor> tryMatchCoIter(scf::WhileOp whileOp);

} // namespace splyce
} // namespace mlir

#endif // TRANSFORM_SPLYCE_PATTERN_H
