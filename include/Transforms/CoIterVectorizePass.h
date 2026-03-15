#ifndef TRANSFORMS_SPLYCE_VECTORIZE_PASS_H
#define TRANSFORMS_SPLYCE_VECTORIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

// Forward declaration - avoid pulling in full dialect headers here
class RewritePatternSet;
class MLIRContext;

// pass factory
// default arguments match the TableGen option defaults
std::unique_ptr<Pass> createCoIterVectorizePass(unsigned vectorWidth = 8, float minDensity = 0.0f, bool tailFallback = true);

// pattern population
// exposed separately so the patterns can be used inside other passes or
// combined with additional rewrites in a larger pipeline.
void populateCoIterVectorizePatterns(RewritePatternSet &patterns, unsigned vectorWidth, float minDensity);

// pass registration
// call once at startup to make the pass available via CLI flag --splyce
void registerCoIterVectorizePass();

} // namespace mlir

// generated base-class boilerplate from TableGen
#define GEN_PASS_DECL_COITERVECTORIZE
namespace mlir {
#include "Transforms/Passes.h.inc"
}

#endif // TRANSFORMS_SPLYCE_VECTORIZE_PASS_H