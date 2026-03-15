# lit.cfg.py.in  —  configured by CMake into the build directory as lit.cfg.py
#
# Teaches lit where our tool and FileCheck live, and which file extension
# identifies runnable tests.

import lit.formats

config.name            = "CoIterVectorize"
config.test_format     = lit.formats.ShTest(not lit_config.useValgrind)
config.suffixes        = [".mlir"]
config.test_source_root = "@CMAKE_CURRENT_SOURCE_DIR@"
config.test_exec_root   = "@CMAKE_CURRENT_BINARY_DIR@"

# Substitutions used in RUN lines: %coiter-opt and %filecheck
config.substitutions.append(("%coiter-opt", "@COITER_OPT_EXE@"))
config.substitutions.append(("%filecheck",  "@FILECHECK_EXE@"))