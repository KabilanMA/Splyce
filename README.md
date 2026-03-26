# Splyce (Slicing Sparse Data)

To translate Sparse Dialect code into SCF loops use the follows command:

`mlir-opt sparse_mttkrp.mlir --sparsification-and-bufferization -o scf_loops.mlir`

```
mlir-opt scf_loops.mlir \
  --canonicalize \
  --cse \
  --loop-invariant-code-motion \
  --lower-vector-mask \
  --convert-vector-to-scf \
  --canonicalize \ 
  --cse \
  --expand-realloc \
  --sparse-storage-specifier-to-llvm \
  --convert-linalg-to-loops \
  --lower-affine \
  --canonicalize \
  --cse \
  --convert-scf-to-cf \
  --expand-strided-metadata \
  --finalize-memref-to-llvm \
  --convert-vector-to-llvm="enable-x86vector=1" \
  --convert-math-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts   
  -o llvm_dialect.mlir
```

`mlir-opt sparse_mttkrp.mlir --sparsifier | mlir-translate --mlir-to-llvmir -o mttkrp.ll`

`clang -O3 mttkrp.ll   -L"/home/kabilan/llvm-dev/lib"   -lmlir_c_runner_utils   -lmlir_runner_utils   -Wl,-rpath,"/home/kabilan/llvm-dev/lib"   -o mttkrp_benchmark`

### Questions to Answer

1. Why not use a BlockSparse encoding?

    - It's the wrong abstraction for tensors other than 2D. BSR is a 2D matrix format -- `(d0 floordiv B, d1 floordiv B, d0 mod B, d1 mod B)`. There is no block dimension to exploit. The encoding simply doesnn't apply.
    - It moves the intersection problem, not solves it. The scalar `scf.while` co-iteration loop reappears at the block-coordinate level. We still get the same branchy, un-vectorizable pointer-chasing, just a coarser granularity. The fundamental issue is untouched.
    - It introduces phantom nonzeros. Coordinates in our `.tns` file are arbitrary. Forcing them into fixed-size blocks requires storing zeros that don't exist in the original data, inflating memory and bandwidth for no computational gain.

2. What are the conditions under which BlockSparse will be beneficial compared to this vectorization?

For this, the data is natually block-structured. If real nonzeros cluster into BxB tiles with >50% density inside each tile, fill-in is cheap and the dense inner dimension give the auto-vectorizer free SIMD.

3. What are the condiions that direct sparse dialect will be beneficial comparared to this vectorization?


cmake -G Ninja .. -DMLIR_DIR=/home/kabilan/llvm-dev/lib/cmake/mlir
ninja
./bin/splyce-opt --splyce="vector-width=8" --debug-only=splyce-vectorize ../test/coiter_scf.mlir -o /dev/null

# Building and Running the CoIter Vectorize Pass
# ═══════════════════════════════════════════════════════════════════════════════
#
# This guide covers two paths:
#
#   A. Build LLVM+MLIR from source  (required if you don't have an install)
#   B. Use a pre-built LLVM install (faster, skip to "Build the pass")
#
# Tested against LLVM 18 and 19.  The pass uses no deprecated APIs so it
# should work on any LLVM >= 17.
# ═══════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────────
# Prerequisites
# ───────────────────────────────────────────────────────────────────────────────

# Ubuntu / Debian
sudo apt-get install -y \
  cmake ninja-build clang lld \
  python3 python3-pip \
  git zlib1g-dev

# macOS (Homebrew)
brew install cmake ninja llvm python3

# RHEL / Fedora
sudo dnf install -y cmake ninja-build clang lld python3 git zlib-devel


# ───────────────────────────────────────────────────────────────────────────────
# PATH A — Build LLVM + MLIR from source
# (skip to PATH B if you already have an MLIR install with cmake files)
# ───────────────────────────────────────────────────────────────────────────────

git clone --depth=1 https://github.com/llvm/llvm-project.git
cd llvm-project

# Configure.
# -DLLVM_ENABLE_PROJECTS="mlir" — build MLIR alongside LLVM core
# -DLLVM_TARGETS_TO_BUILD="X86" — only native target; add AArch64/RISCV as needed
# -DLLVM_ENABLE_ASSERTIONS=ON   — required for MLIR pattern matching debug output
# -DCMAKE_BUILD_TYPE=Release    — use RelWithDebInfo if you want debuggable IR
cmake -S llvm -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install

# Build and install (takes ~20–40 min on a 16-core machine).
ninja -C build install

# Set shell variables used throughout the rest of this guide.
export LLVM_INSTALL=$HOME/llvm-install
export PATH=$LLVM_INSTALL/bin:$PATH

cd ..   # back to your workspace root


# ───────────────────────────────────────────────────────────────────────────────
# PATH B — Use a pre-built LLVM install
# ───────────────────────────────────────────────────────────────────────────────
# If your distro ships MLIR cmake files (Ubuntu 22.04+ with llvm-XX-dev):
#
#   sudo apt-get install llvm-19-dev mlir-19-tools libmlir-19-dev
#   export LLVM_INSTALL=/usr/lib/llvm-19
#   export PATH=$LLVM_INSTALL/bin:$PATH
#
# Homebrew (macOS):
#   brew install llvm
#   export LLVM_INSTALL=$(brew --prefix llvm)
#   export PATH=$LLVM_INSTALL/bin:$PATH
#
# Verify the install exposes cmake files:
ls $LLVM_INSTALL/lib/cmake/mlir/MLIRConfig.cmake   # must exist
ls $LLVM_INSTALL/lib/cmake/llvm/LLVMConfig.cmake   # must exist


# ───────────────────────────────────────────────────────────────────────────────
# Build the pass
# ───────────────────────────────────────────────────────────────────────────────

# Clone or enter the project directory.
# (If you received the files directly, just cd into them.)
# cd splyce-mlir

# Configure.
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL/lib/cmake/llvm

# Build everything: the library, coiter-opt, and register tests.
ninja -C build

# Verify the tool was built.
./build/bin/splyce-opt --help | grep splyce
# Expected output:
#   --splyce                    - Vectorize scf.while co-iteration loops


# ───────────────────────────────────────────────────────────────────────────────
# Run the pass on the example file
# ───────────────────────────────────────────────────────────────────────────────

# 1. Print the transformed IR to stdout for human inspection.
./build/bin/splyce-opt \
  --splyce="vector-width=8" \
  test/coiter_scf.mlir

# 2. Dump to a file for further lowering.
./build/bin/splyce-opt \
  --splyce="vector-width=8" \
  test/coiter_scf.mlir \
  -o /tmp/coiter_vec_out.mlir

cat /tmp/coiter_vec_out.mlir

# 3. Enable debug output to see why patterns are (or are not) matching.
./build/bin/splyce-opt \
  --splyce="vector-width=8" \
  --mlir-print-ir-after-all \
  --debug-only=splyce-vectorize \
  test/coiter_scf.mlir

# Expected debug lines (stderr):
#   [coiter] Matched 2-way co-iteration
#   [coiter] Vectorizing co-iteration loop at ... (W=8)
#   [coiter] Matched 3-way co-iteration
#   [coiter] Vectorizing co-iteration loop at ... (W=8)
#   [coiter] Only 1 FP ops, need 2 — Not profitable — skipping   (unprofitable test)


# ───────────────────────────────────────────────────────────────────────────────
# Run on the sparse_mul_vectorized.mlir file from the earlier session
# ───────────────────────────────────────────────────────────────────────────────

# The scf loop file is the *input* to the pass (scalar SCF, not the
# hand-vectorized file).  Feed the scalar SCF output from sparse lowering:
#
# Step 1: Lower the linalg.generic to SCF loops (this is what MLIR normally
#         produces before our pass runs).
mlir-opt \
  --sparsification \
  --sparse-tensor-conversion \
  --linalg-bufferize \
  --convert-linalg-to-loops \
  sparse_mul_scalar.mlir \
  -o /tmp/scf_loops.mlir

# Step 2: Run our pass on the resulting SCF IR.
./build/bin/splyce-opt \
  /tmp/scf_loops.mlir \
  --splyce="vector-width=8" \
  -o /tmp/scf_vectorized.mlir

# Step 3: Lower all the way to LLVM IR and execute.
mlir-opt \
  --convert-vector-to-llvm="enable-avx2" \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  /tmp/scf_vectorized.mlir \
  | mlir-cpu-runner \
      --entry-point-result=void \
      --shared-libs=$LLVM_INSTALL/lib/libmlir_runner_utils.so,$LLVM_INSTALL/lib/libmlir_c_runner_utils.so


# ───────────────────────────────────────────────────────────────────────────────
# Run the FileCheck tests
# ───────────────────────────────────────────────────────────────────────────────

# CTest (individual test per .mlir file).
ctest --test-dir build --output-on-failure

# Or target the specific test by name.
ctest --test-dir build -R coiter_filecheck_coiter_scf -V

# Expected output:
#   Test  #1: coiter_filecheck_coiter_vectorize ........... Passed  0.12 sec

# If you installed lit (pip install lit), use the umbrella target instead.
pip install lit
cmake --build build --target check-coiter


# ───────────────────────────────────────────────────────────────────────────────
# Try different vector widths
# ───────────────────────────────────────────────────────────────────────────────

# AVX2  — 8 lanes of f32 (default)
./build/bin/splyce-opt \
  --splyce="vector-width=8" \
  test/coiter_scf.mlir

# AVX-512 — 16 lanes of f32
./build/bin/splyce-opt \
  --splyce="vector-width=16" \
  test/coiter_scf.mlir

# ARM SVE / RISC-V V — try width=4 for a conservative start
./build/bin/splyce-opt \
  --splyce="vector-width=4" \
  test/coiter_scf.mlir

# Skip sparse loops below 10% intersection density.
./build/bin/splyce-opt \
  --splyce="vector-width=8 min-density=0.1" \
  test/coiter_scf.mlir


# ───────────────────────────────────────────────────────────────────────────────
# Typical build times (for reference)
# ───────────────────────────────────────────────────────────────────────────────
#
#   LLVM from source, 16-core Ryzen 9 / Apple M2:  ~25 min (Release)
#   LLVM from source, 8-core laptop:               ~50 min (Release)
#   The pass itself (3 .cpp files):                < 30 sec
#
# Tip: if you only need to iterate on the pass and already have a working
# LLVM install, rebuilding after editing a .cpp file takes ~5 seconds with
# Ninja because it only recompiles changed translation units.