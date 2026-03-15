# include/Transforms/CMakeLists.txt
#
# Owns the TableGen invocation that generates Passes.h.inc from Passes.td.
#
# Why this lives in include/ rather than lib/:
#   - Passes.h.inc is a *header* consumed by both lib/ and any downstream
#     project that wants to programmatically create the pass.
#   - Placing the tablegen target here keeps the generated file next to
#     its source (.td) and its consumers (.h), making the dependency
#     graph obvious without cross-directory path arithmetic.
#   - lib/Transforms/CMakeLists.txt simply lists CoIterVectorizePassIncGen
#     in DEPENDS — it does not need to know where the .inc was generated.

set(LLVM_TARGET_DEFINITIONS Passes.td)

# -gen-pass-decls  : generates the Pass declarations
# -name            : prefix for the generated symbols (CoIterVectorize*)
mlir_tablegen(
  ${PROJECT_BINARY_DIR}/include/Transforms/Passes.h.inc
  -gen-pass-decls
  -name CoIterVectorize
)

# Public CMake target that any library can depend on via DEPENDS.
# Listing it in DEPENDS ensures the .inc exists before any .cpp that
# includes it is compiled.
add_public_tablegen_target(CoIterVectorizePassIncGen)