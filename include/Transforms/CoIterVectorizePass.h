#ifndef TRANSFORMS_SPLYCE_VECTORIZE_PASS_H
#define TRANSFORMS_SPLYCE_VECTORIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {

// Forward declaration - avoid pulling in full dialect headers here
class RewritePatternSet;
class MLIRContext;

// pass factory
// default arguments match the TableGen option defaults
std::unique_ptr<Pass> createCoIterVectorizePass(unsigned vectorWidth = 8, float minDensity = 0.0f,
                                                bool tailFallback = true,
                                                float runtimeDensityThreshold = 0.5f,
                                                llvm::StringRef targetFunction = "");

// pattern population
// exposed separately so the patterns can be used inside other passes or
// combined with additional rewrites in a larger pipeline.
void populateCoIterVectorizePatterns(RewritePatternSet &patterns, unsigned vectorWidth,
                                     float minDensity, float runtimeDensityThreshold = 0.5f,
                                     llvm::StringRef targetFunction = "");

// pass registration
// call once at startup to make the pass available via CLI flag --splyce
void registerCoIterVectorizePass();


// generated base-class boilerplate from TableGen
#define GEN_PASS_DECL_COITERVECTORIZE
#include "Transforms/Passes.h.inc"
} // namespace mlir

#endif // TRANSFORMS_SPLYCE_VECTORIZE_PASS_H