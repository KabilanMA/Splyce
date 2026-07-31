# Splyce (Slicing Sparse Data)

```bash
mlir-opt spgemm.mlir --linalg-generalize-named-ops --linalg-fuse-elementwise-ops --linalg-generalize-named-ops --pre-sparsification-rewrite --pre-sparsification-rewrite --empty-tensor-to-alloc-tensor --sparse-reinterpret-map --sparsification="parallelization-strategy=dense-outer-loop" --stage-sparse-ops --lower-sparse-ops-to-foreach --sparse-reinterpret-map --lower-sparse-foreach-to-scf --loop-invariant-code-motion --sparse-tensor-conversion -o temp.mlir
```

```bash
mlir-opt spgemm_splyce.mlir --sparsification-and-bufferization --sparse-storage-specifier-to-llvm --canonicalize --convert-linalg-to-loops --convert-vector-to-scf --expand-realloc --convert-scf-to-openmp --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-vector-to-llvm --convert-complex-to-standard --arith-expand --convert-math-to-llvm --convert-complex-to-libm --convert-vector-to-llvm --convert-openmp-to-llvm --convert-to-llvm -o spgemm_llvm.mlir
```
To translate Sparse Dialect code into SCF loops use the follows command:

`mlir-opt sparse_mttkrp.mlir --sparsification-and-bufferization --sparse-vectorization="vl=4" -o scf_loops.mlir`

To translate the generated SCF+Vector loop into LLVM dialect use the following command:

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

To translate a sparse dialect program directly into LLVM IR use the following command:

`mlir-opt sparse_mttkrp.mlir --sparsifier | mlir-translate --mlir-to-llvmir -o mttkrp.ll`

`clang -O3 mttkrp.ll   -L"/home/kabilan/llvm-dev/lib"   -lmlir_c_runner_utils   -lmlir_runner_utils   -Wl,-rpath,"/home/kabilan/llvm-dev/lib"   -o mttkrp_benchmark`

## Questions to Answer

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

## Building and Running the CoIter Vectorize Pass
#### ════════════════════════════════════════════════

This guide covers two paths:
1. Build LLVM+MLIR from source  (required if you don't have an install)
2. Use a pre-built LLVM install (faster, skip to "Build the pass")

Tested against LLVM 23.0.0.  The pass uses no deprecated APIs so it should work on any LLVM >= 23.0.0git.
#### ════════════════════════════════════════════════


### Prerequisites
---

#### Ubuntu / Debian
Note: clang and lld are installed to serve as a fast host compiler/linker to bootstrap the MLIR build and to compile the Splyce pass later.

```bash
sudo apt-get install -y \
  cmake ninja-build clang lld \
  python3 python3-pip \
  git zlib1g-dev
```

#### macOS (Homebrew)
```zsh
brew install cmake ninja llvm python3
```

#### RHEL / Fedora
```bash
sudo dnf install -y cmake ninja-build clang lld python3 git zlib-devel
```

### PATH A — Build LLVM + MLIR from source

__(Skip to PATH B if you already have an MLIR install with cmake files)__

```bash
git clone --depth=1 https://github.com/llvm/llvm-project.git
cd llvm-project
```
#### Configure.

-DLLVM_ENABLE_PROJECTS="mlir" — build MLIR alongside LLVM core
-DLLVM_TARGETS_TO_BUILD="X86" — only native target; add AArch64/RISCV as needed
-DLLVM_ENABLE_ASSERTIONS=ON   — required for MLIR pattern matching debug output
-DCMAKE_BUILD_TYPE=Release    — use RelWithDebInfo if you want debuggable IR

```bash
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
```

#### Build and install (takes ~20–40 min on a 16-core machine).
```bash
ninja -C build install
```

#### Set shell variables used throughout the rest of this guide.
```bash
export LLVM_INSTALL=$HOME/llvm-install
export PATH=$LLVM_INSTALL/bin:$PATH

cd ..   # back to your workspace root
```

### PATH B — Use a pre-built LLVM install

If your distro ships MLIR cmake files (Ubuntu 22.04+ with llvm-XX-dev):

Ubuntu / Debian:
```bash
sudo apt-get install llvm-19-dev mlir-19-tools libmlir-19-dev
export LLVM_INSTALL=/usr/lib/llvm-XX
export PATH=$LLVM_INSTALL/bin:$PATH
```

Homebrew (macOS):
```zsh
brew install llvm
export LLVM_INSTALL=$(brew --prefix llvm)
export PATH=$LLVM_INSTALL/bin:$PATH
```

#### Verify the install exposes cmake files:
```bash
ls $LLVM_INSTALL/lib/cmake/mlir/MLIRConfig.cmake   # must exist
ls $LLVM_INSTALL/lib/cmake/llvm/LLVMConfig.cmake   # must exist
```

---
### Build the pass
---

Clone or enter the project directory. (If you received the files directly, just cd into them.)

```bash
cd splyce
```

#### Configure.
```bash
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL/lib/cmake/llvm
```

#### Build everything: the library, coiter-opt, and register tests.
```bash
ninja -C build

# Verify the tool was built.
./build/bin/splyce-opt --help | grep splyce
# Expected output:
#   --splyce - Vectorize scf.while co-iteration loops using Vector dialect
```


### Run the pass on the example file

#### Print the transformed IR to stdout for human inspection.

```bash
./build/bin/splyce-opt \
  "--splyce=target-function=mttkrp_kernel" \
  --debug-only=splyce-vectorize \
  ./playground/spgemm_scf.mlir
```

Remove the `--debug-only=splyce-vectorize` to avoid debug output.


#### Dump to a file for further lowering.

```bash
./build/bin/splyce-opt \
  "--splyce=target-function=mttkrp_kernel" \
  --debug-only=splyce-vectorize \
  ./playground/spgemm_scf.mlir
  -o /tmp/coiter_vec_out.mlir

cat /tmp/coiter_vec_out.mlir
```

#### All possible cmd arguments for the pass.
```bash
./build/bin/splyce-opt --splyce="vector-width=8 min-density=0.1 target-function=spgemm" input.mlir

```
<!-- # 3. Enable debug output to see why patterns are (or are not) matching.
./build/bin/splyce-opt \
  --splyce="vector-width=8" \
  --mlir-print-ir-after-all \
  --debug-only=splyce-vectorize \
  test/coiter_scf.mlir -->

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir -DLLVM_DIR=$LLVM_INSTALL/lib/cmake/llvm -DCMAKE_C_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13" -DCMAKE_CXX_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"