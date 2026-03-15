# test/CMakeLists.txt
#
# Registers the FileCheck-based test suite with CTest.
#
# Two test modes are provided:
#
#   1. coiter_filecheck_*  — individual CTest targets, one per .mlir file.
#      Run with:  ctest -R coiter_filecheck
#      These are fast (<1 s each) and show per-file pass/fail in IDEs.
#
#   2. check-coiter  — a lit-driven umbrella target that mirrors the
#      llvm-project convention (make check-mlir, make check-llvm …).
#      Run with:  cmake --build build --target check-coiter
#      Requires `lit` on PATH (pip install lit).
#
# ── Tool locations ─────────────────────────────────────────────────────────────
# Prefer the splyce-opt we just built; fall back to the system mlir-opt so
# the tests still work in environments that only have an MLIR install.
if(TARGET splyce-opt)
  set(COITER_OPT_EXE $<TARGET_FILE:splyce-opt>)
else()
  find_program(COITER_OPT_EXE mlir-opt REQUIRED)
endif()

find_program(FILECHECK_EXE FileCheck
  HINTS ${LLVM_TOOLS_BINARY_DIR}
  REQUIRED
)

# ── Helper: register one .mlir file as a CTest + FileCheck test ───────────────
#
# Usage:
#   add_filecheck_test(NAME <name>
#                      SOURCE <file.mlir>
#                      PASS_PIPELINE <opt-flags…>)
#
# The PASS_PIPELINE argument is forwarded verbatim to splyce-opt, so you can
# pass multiple flags:
#   add_filecheck_test(NAME foo SOURCE foo.mlir
#                      PASS_PIPELINE "--coiter-vectorize=vector-width=8" "--canonicalize")
function(add_filecheck_test)
  cmake_parse_arguments(ARG "" "NAME;SOURCE" "PASS_PIPELINE" ${ARGN})

  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SOURCE}")

  add_test(
    NAME    "coiter_filecheck_${ARG_NAME}"
    COMMAND sh -c
      "${COITER_OPT_EXE} ${ARG_PASS_PIPELINE} ${src} | ${FILECHECK_EXE} ${src}"
  )

  # Surface the test in IDEs (VS Code CMake Tools, CLion, …).
  set_tests_properties("coiter_filecheck_${ARG_NAME}"
    PROPERTIES
      LABELS          "filecheck"
      TIMEOUT         60
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endfunction()

# ── Register test files ────────────────────────────────────────────────────────

# N=2 and N=3 co-iteration + negative (unprofitable) test.
add_filecheck_test(
  NAME          coiter_scf
  SOURCE        coiter_scf.mlir
  PASS_PIPELINE "--splyce=vector-width=8"
)

# To add more tests, just call add_filecheck_test() again:
#
#   add_filecheck_test(
#     NAME          coiter_vectorize_avx512
#     SOURCE        coiter_vectorize.mlir
#     PASS_PIPELINE "--coiter-vectorize=vector-width=16"
#   )
#
#   add_filecheck_test(
#     NAME          coiter_density_gate
#     SOURCE        coiter_density.mlir
#     PASS_PIPELINE "--coiter-vectorize=vector-width=8 --coiter-vectorize=min-density=0.5"
#   )

# ── lit umbrella target (optional) ────────────────────────────────────────────
# Mirrors the llvm-project convention so CI scripts can use the same
# `cmake --build . --target check-coiter` invocation they use for upstream.
find_program(LIT_EXE lit HINTS ${LLVM_TOOLS_BINARY_DIR})

if(LIT_EXE)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py
    @ONLY
  )

  add_custom_target(check-coiter
    COMMAND ${LIT_EXE} ${CMAKE_CURRENT_BINARY_DIR}
              --param coiter_opt=${COITER_OPT_EXE}
              --param filecheck=${FILECHECK_EXE}
              -v
    DEPENDS splyce-opt
    COMMENT "Running co-iteration vectorize lit tests"
    USES_TERMINAL
  )
else()
  message(STATUS
    "lit not found — 'check-coiter' target unavailable. "
    "Install with: pip install lit"
  )
endif()