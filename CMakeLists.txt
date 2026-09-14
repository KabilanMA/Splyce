cmake_minimum_required(VERSION 3.20.0)

# Default to clang/clang++ unless the user explicitly picked a compiler 
# via -DCMAKE_C_COMPILER=/ -DCMAKE_CXX_COMPILER=, or the CC/CXX env vars 
# CMake itself would otherwise honor. 
# Must be set before project(), since that's where compiler detection actually happens.
if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
  set(CMAKE_C_COMPILER clang)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER clang++)
endif()

# Default to a Release build unless the user explicitly picked a build type
# (-DCMAKE_BUILD_TYPE=...)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(splyce-mlir LANGUAGES CXX C)

# Find MLIR
if(NOT DEFINED MLIR_DIR AND DEFINED ENV{LLVM_INSTALL})
  set(MLIR_DIR "$ENV{LLVM_INSTALL}/lib/cmake/mlir" CACHE PATH "Path to MLIRConfig.cmake")
endif()
if(NOT DEFINED LLVM_DIR AND DEFINED ENV{LLVM_INSTALL})
  set(LLVM_DIR "$ENV{LLVM_INSTALL}/lib/cmake/llvm" CACHE PATH "Path to LLVMConfig.cmake")
endif()
if(NOT DEFINED MLIR_DIR)
  message(FATAL_ERROR "MLIR_DIR is not set and $LLVM_INSTALL is not in the environment — "
                       "either export LLVM_INSTALL=/path/to/llvm-install first, or pass "
                       "-DMLIR_DIR=... -DLLVM_DIR=... explicitly (see README).")
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Setup output directories
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Import MLIR/LLVM CMake macros
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_definitions(${LLVM_DEFINITIONS})

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Build subdirectories
add_subdirectory(include/Transforms)
add_subdirectory(lib/Transforms)
add_subdirectory(tools)
# add_subdirectory(test)