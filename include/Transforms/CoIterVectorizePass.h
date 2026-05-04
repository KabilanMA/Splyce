#ifndef TRANSFORMS_SPLYCE_VECTORIZE_PASS_H
#define TRANSFORMS_SPLYCE_VECTORIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {

class RewritePatternSet;

// Creates the co-iteration vectorization pass.
// vectorWidth   : SIMD lane count W (default 4).
// targetFunction: only transform loops inside this function name ("" = all).
// phaseSelect   : 3-bit selector "XYZ" (X=Phase3, Y=Phase2, Z=Phase1;
//                 '1'=vector, '0'=scalar). Default "001" = Phase 1 only.
std::unique_ptr<Pass> createCoIterVectorizePass(
    unsigned vectorWidth = 4, llvm::StringRef targetFunction = "",
    llvm::StringRef phaseSelect = "001");

// Adds the co-iteration vectorization rewrite pattern to an existing set.
// Exposed separately so callers can combine it with other rewrites.
void populateCoIterVectorizePatterns(RewritePatternSet &patterns,
                                     unsigned vectorWidth,
                                     llvm::StringRef targetFunction = "",
                                     llvm::StringRef phaseSelect = "001");

// Registers the pass so it is available via the CLI flag --splyce.
// Call once at startup.
void registerCoIterVectorizePass();

// TableGen-generated base class boilerplate.
#define GEN_PASS_DECL_COITERVECTORIZE
#include "Transforms/Passes.h.inc"

} // namespace mlir

#endif // TRANSFORMS_SPLYCE_VECTORIZE_PASS_H
