# Splyce (Slicing Sparse Data)

Splyce is an out-of-tree MLIR project featuring a custom pass (`--splyce`) designed to recognize and vectorize N-way co-iteration loops (`scf.while`). By replacing branchy, un-vectorizable scalar epilogues with SIMD fast-lane execution paths, Splyce accelerates sparse tensor computations (like SpGEMM and MTTKRP).

---

## Building and Installation

Tested against LLVM 23.0.0 (but it should work on any LLVM >= 23.0.0git).

### Prerequisites

**Ubuntu / Debian:**
```bash
sudo apt-get install -y cmake ninja-build clang lld python3 python3-pip git zlib1g-dev
```
**macOS (Homebrew):**
```zsh
brew install cmake ninja llvm python3
```
**Fedora / RHEL:**
```bash
sudo dnf install -y cmake ninja-build clang lld python3 git zlib-devel
```

### 1. LLVM & MLIR Setup

You can either build LLVM from source or use a pre-built installation.

**PATH A - Build from Source (Recommended)**
```bash
git clone --depth=1 https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install
ninja -C build install

export LLVM_INSTALL=$HOME/llvm-install
export PATH=$LLVM_INSTALL/bin:$PATH
```

**PATH B - Pre-built Install (Faster)**
If your OS provides MLIR cmake files (e.g., Ubuntu 22.04+ or macOS):
*   **Ubuntu:** `sudo apt-get install llvm-19-dev mlir-19-tools libmlir-19-dev`
*   **macOS:** `brew install llvm`

*(Make sure to export `LLVM_INSTALL` to your installation directory and add it to your `PATH`).*

### 2. Build the Splyce Pass
```bash
# Run inside the Splyce repository root
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL/lib/cmake/llvm
ninja -C build
```
Verify it built correctly:
```bash
./build/bin/splyce-opt --help | grep splyce
```

---

## End-to-End Compilation Pipelines

The following examples demonstrate compiling Sparse Dialect structures all the way to LLVM IR executable benchmarks, inserting the `splyce-opt` vectorization pass in the middle of the pipeline.

### Single-Threaded Compilation (e.g., SpGEMM)

**1. SparseTensor -> SCF**
```bash
mlir-opt ./playground/spgemm.mlir --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
  --linalg-generalize-named-ops --pre-sparsification-rewrite \
  --pre-sparsification-rewrite --empty-tensor-to-alloc-tensor --sparse-reinterpret-map \
  --sparsification --stage-sparse-ops --lower-sparse-ops-to-foreach \
  --sparse-reinterpret-map --lower-sparse-foreach-to-scf --loop-invariant-code-motion \
  --sparse-tensor-conversion -o ./playground/spgemm_scf.mlir
```

**2. Splyce Vectorization (SCF -> Vectorized SCF)**
```bash
./build/bin/splyce-opt ./playground/spgemm_scf.mlir --splyce="target-function=spgemm vector-width=4 phase-select=001" -o ./playground/spgemm_splyce.mlir
```

**3. SCF -> LLVM Dialect**
```bash
mlir-opt ./playground/spgemm_scf.mlir --canonicalize --cse --loop-invariant-code-motion \
  --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-from-loops=true" \
  --convert-bufferization-to-memref --lower-vector-mask --convert-vector-to-scf \
  --canonicalize --cse --expand-realloc --sparse-storage-specifier-to-llvm \
  --convert-linalg-to-loops --lower-affine --canonicalize --cse --convert-scf-to-cf \
  --expand-strided-metadata --finalize-memref-to-llvm \
  --convert-vector-to-llvm="enable-x86vector=1" --convert-math-to-llvm \
  --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
  --reconcile-unrealized-casts -o ./playground/spgemm_llvm_splyce.mlir
```

**4. Binary Generation**
```bash
mlir-translate spgemm_llvm_splyce.mlir --mlir-to-llvmir -o spgemm_splyce.ll

clang -O3 spgemm_splyce.ll -mavx512f -mavx512vl -fno-vectorize -fno-slp-vectorize  -L"$LLVM_INSTALL_DIR/lib"   -lmlir_c_runner_utils   -lmlir_runner_utils   -Wl,-rpath,"$LLVM_INSTALL_DIR/lib"   -o test_benchmark_spgemm_splyce
```

### Multi-Threaded Compilation (e.g., SpGEMM)

**1. SparseTensor -> SCF (Parallel)**
```bash
mlir-opt ./playground/spgemm.mlir --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
  --linalg-generalize-named-ops --pre-sparsification-rewrite \
  --pre-sparsification-rewrite --empty-tensor-to-alloc-tensor --sparse-reinterpret-map \
  --sparsification="parallelization-strategy=dense-outer-loop" --stage-sparse-ops \
  --lower-sparse-ops-to-foreach --sparse-reinterpret-map --lower-sparse-foreach-to-scf \
  --loop-invariant-code-motion --sparse-tensor-conversion -o ./playground/spgemm_scf_parallel.mlir
```

**2. Splyce Vectorization**
```bash
./build/bin/splyce-opt ./playground/spgemm_scf_parallel.mlir --splyce="target-function=spgemm vector-width=4 phase-select=001" -o ./playground/spgemm_splyce_parallel.mlir
```

**3. SCF -> LLVM Dialect**
```bash
mlir-opt ./playground/spgemm_splyce_parallel.mlir --canonicalize --sparsification-and-bufferization \
  --sparse-storage-specifier-to-llvm --canonicalize --convert-linalg-to-loops \
  --convert-vector-to-scf --expand-realloc --convert-scf-to-openmp \
  --convert-openmp-to-llvm --convert-scf-to-cf --expand-strided-metadata \
  --lower-affine --convert-vector-to-llvm --convert-complex-to-standard \
  --arith-expand --convert-math-to-llvm --convert-complex-to-libm \
  --convert-vector-to-llvm --convert-to-llvm --reconcile-unrealized-casts \
  -o ./playground/spgemm_llvm_splyce_parallel.mlir
```

**4. Binary Generation (with OpenMP & AVX512)**
```bash
mlir-translate ./playground/spgemm_llvm_splyce_parallel.mlir --mlir-to-llvmir -o ./playground/spgemm_llvm_splyce_parallel.ll

clang -O3 ./playground/spgemm_llvm_splyce_parallel.ll -mavx512f -mavx512vl -fopenmp -fno-vectorize \
  -fno-slp-vectorize -L"$LLVM_INSTALL/lib" -lmlir_c_runner_utils -lmlir_runner_utils \
  -Wl,-rpath,"$LLVM_INSTALL/lib" -o test_benchmark_spgemm_splyce_parallel
```

---

## FAQ

**1. Why not use a BlockSparse encoding?**
*   It's the wrong abstraction for tensors other than 2D. BSR is a 2D matrix format (`d0 floordiv B`, `d1 floordiv B`, `d0 mod B`, `d1 mod B`). There is no block dimension to exploit for generalized sparse tensors.
*   It moves the intersection problem but does not solve it. The scalar `scf.while` co-iteration loop simply reappears at the block-coordinate level. We still get the same branchy, un-vectorizable pointer-chasing, just at a coarser granularity.
*   It introduces phantom nonzeros. Coordinates in real datasets are arbitrary. Forcing them into fixed-size blocks requires storing zeros that don't exist in the original data, inflating memory and bandwidth for no computational gain.

**2. Under what conditions is BlockSparse beneficial compared to this vectorization?**
If the data is naturally block-structured. If real nonzeros cluster into `B x B` tiles with >50% density inside each tile, fill-in is cheap and the dense inner dimensions grant the auto-vectorizer free SIMD.

**3. When is the direct sparse dialect beneficial compared to Splyce vectorization?**
The Splyce vectorization is an optimization pass applied on top of the sparse dialect. If the density of nonzeros is extremely low, the shuffle and mask-matching overhead in the SIMD fast-lane may outweigh the scalar loop branching. In those rare scenarios, standard scalar sparse dialect lowering might yield better performance.