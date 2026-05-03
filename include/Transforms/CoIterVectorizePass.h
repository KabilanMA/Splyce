#ifndef TRANSFORMS_SPLYCE_VECTORIZE_PASS_H
#define TRANSFORMS_SPLYCE_VECTORIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {

class RewritePatternSet;

// Creates the co-iteration vectorization pass.
// vectorWidth   : SIMD lane count W (default 8).
// targetFunction: only transform loops inside this function name ("" = all).
std::unique_ptr<Pass> createCoIterVectorizePass(
    unsigned vectorWidth = 4, llvm::StringRef targetFunction = "");

// Adds the co-iteration vectorization rewrite pattern to an existing set.
// Exposed separately so callers can combine it with other rewrites.
void populateCoIterVectorizePatterns(RewritePatternSet &patterns,
                                     unsigned vectorWidth,
                                     llvm::StringRef targetFunction = "");

// Registers the pass so it is available via the CLI flag --splyce.
// Call once at startup.
void registerCoIterVectorizePass();

// TableGen-generated base class boilerplate.
#define GEN_PASS_DECL_COITERVECTORIZE
#include "Transforms/Passes.h.inc"

} // namespace mlir

#endif // TRANSFORMS_SPLYCE_VECTORIZE_PASS_H
