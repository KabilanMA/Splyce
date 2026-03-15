# lib/Transforms/CMakeLists.txt
#
# Builds the CoIterVectorizeTransforms shared/static library.
#
# Separation of concerns vs the root CMakeLists:
#   • Source file list lives here, next to the sources themselves.
#   • Dialect link dependencies live here — no other directory needs to
#     know which dialects this transform depends on.
#   • The only thing the root provided was global include paths and
#     the MLIR/LLVM CMake module setup, which is all we need from it.

add_mlir_library(CoIterVectorizeTransforms

  # ── Sources ──────────────────────────────────────────────────────────────────
  CoIterPattern.cpp
  CoIterVectorBuilder.cpp
  CoIterVectorizePass.cpp

  # ── Generated header dependency ───────────────────────────────────────────────
  # Ensures Passes.h.inc exists before any .cpp that includes it is compiled.
  # The target itself lives in include/Transforms/CMakeLists.txt.
  DEPENDS
    CoIterVectorizePassIncGen

  # ── Public link dependencies ──────────────────────────────────────────────────
  # LINK_LIBS PUBLIC means these are transitively required by any target
  # that links CoIterVectorizeTransforms (e.g. coiter-opt).
  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRVectorDialect
    MLIRTransforms        # applyPatternsAndFoldGreedily, GreedyPatternRewriteDriver
    MLIRPass              # Pass, PassManager infrastructure
    MLIRIR                # OpBuilder, Value, Type, Location, …
)