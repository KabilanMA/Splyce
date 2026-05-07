# Splyce: N-Way Co-Iteration Vectorization for Sparse Tensors

Splyce is an MLIR optimization pass (`--splyce`) that vectorizes N-way co-iteration loops in sparse tensor computations. It replaces scalar `scf.while` epilogues with SIMD fast-lane execution paths, delivering significant speedups for operations like SpGEMM and MTTKRP.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building and Installation](#building-and-installation)
- [Usage Examples](#usage-examples)
- [Evaluation](#evaluation)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install LLVM/MLIR (commit 6a6d43255059)
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && git checkout 6a6d43255059

# 2. Build Splyce
cd ~/path/to/splyce
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL/lib/cmake/llvm
ninja -C build

# 3. Verify installation
./build/bin/splyce-opt --help | grep splyce
```

---

## Overview

### What is Splyce?

Splyce tackles the vectorization bottleneck in sparse tensor computations. Traditional sparse dialect lowering produces scalar co-iteration loops that:
- Check tensor coordinates on each iteration
- Branch based on coordinate comparisons
- Cannot be auto-vectorized by conventional compilers

Splyce recognizes these patterns and converts them into vectorized "fast lanes" using SIMD operations with masked execution, achieving **N-way parallelism** on existing CPU architectures.

### Key Features

- **N-Way Co-Iteration Vectorization**: Vectorizes pointer-chasing loops in sparse tensor operations
- **SIMD Fast-Lane Generation**: Creates masked SIMD execution paths for coordinate matching
- **Configurable Vector Width**: Supports flexible SIMD width selection (4, 8, 16, etc.)
- **Phase-Based Optimization**: Selective application of vectorization phases
- **Integration with MLIR Pipeline**: Works seamlessly with existing sparse dialect lowering

### Target Operations

- **SpGEMM** (Sparse matrix × matrix multiplication)
- **MTTKRP** (Matricized Tensor Times Khatri-Rao Product)
- Other N-way co-iteration operations

---

## Prerequisites

### System Requirements

- **CMake** 3.20+
- **Ninja** or GNU Make
- **Git**
- **Python** 3.6+
- **C++ Compiler** (GCC 11+ or Clang 14+)

### Platform-Specific Installation

**Ubuntu / Debian:**
```bash
sudo apt-get install -y cmake ninja-build clang lld python3 python3-pip git zlib1g-dev
```

**macOS (Homebrew):**
```bash
brew install cmake ninja llvm python3
```

**Fedora / RHEL:**
```bash
sudo dnf install -y cmake ninja-build clang lld python3 git zlib-devel
```

**Without Root Access:**
```
You can proceed with cmake, python3, and git only.
Build clang/lld from LLVM source to avoid system dependencies.
```

---

## Building and Installation

### Step 1: Setup LLVM & MLIR

This project is tested against **LLVM 23.0.0git** (commit `6a6d43255059`).

#### Option A: Build from Source (Recommended)

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 6a6d43255059

# Setup Python environment
python3 -m venv py_venv
source py_venv/bin/activate
pip install nanobind
```

**With system clang/lld installed:**
```bash
mkdir build
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir;lld" \
  -DLLVM_ENABLE_RUNTIMES="all" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="host;X86" \
  -DLLVM_INCLUDE_TESTS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_USE_LINKER=bfd \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install
```

**Build clang/lld from source (alternative):**
```bash
mkdir build
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_ENABLE_RUNTIMES="all" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="host;X86" \
  -DLLVM_INCLUDE_TESTS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_USE_LINKER=lld \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install
```

**Build and install:**
```bash
ninja -C build install
export LLVM_INSTALL=$HOME/llvm-install
export PATH=$LLVM_INSTALL/bin:$PATH
```

**Verify LLVM installation:**
```bash
mlir-opt --version
```

#### Option B: Use Pre-Built LLVM

If you have a pre-built LLVM installation, simply set:
```bash
export LLVM_INSTALL=/path/to/llvm-install
export PATH=$LLVM_INSTALL/bin:$PATH
```

### Step 2: Build Splyce Pass

```bash
cd /path/to/splyce
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL/lib/cmake/llvm

ninja -C build
```

### Step 3: Verify Build

```bash
./build/bin/splyce-opt --help | grep splyce
```


---

## Usage Examples

### Overview: Compilation Pipeline

A typical Splyce evaluation workflow:

1. **Sparse Dialect → SCF** (Sparsification)
2. **SCF → Vectorized SCF** (Splyce Pass)
3. **Vectorized SCF → LLVM Dialect** (Lowering)
4. **LLVM Dialect → Binary** (Compilation)

### Example 1: Single-Threaded SpGEMM

**1.1 Lower Sparse Tensor to SCF**

```bash
mlir-opt ./playground/spgemm.mlir \
  --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --pre-sparsification-rewrite \
  --empty-tensor-to-alloc-tensor \
  --sparse-reinterpret-map \
  --sparsification \
  --stage-sparse-ops \
  --lower-sparse-ops-to-foreach \
  --lower-sparse-foreach-to-scf \
  --loop-invariant-code-motion \
  --sparse-tensor-conversion \
  -o ./playground/spgemm_scf.mlir
```

**1.2 Apply Splyce Vectorization**

```bash
./build/bin/splyce-opt ./playground/spgemm_scf.mlir \
  --splyce="target-function=spgemm vector-width=4 phase-select=001" \
  --splyce-bufferize-restrict \
  -o ./playground/spgemm_splyce.mlir
```

**Splyce Options:**
- `target-function=<name>`: Target function to optimize
- `vector-width=<width>`: SIMD vector width (4, 8, 16, etc.)
- `phase-select=<phases>`: Bitmask for optimization phases

**1.3 Lower SCF to LLVM Dialect**

```bash
mlir-opt ./playground/spgemm_splyce.mlir \
  --canonicalize \
  --cse \
  --loop-invariant-code-motion \
  --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-from-loops=true" \
  --convert-bufferization-to-memref \
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
  --convert-vector-to-llvm \
  --convert-math-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o ./playground/spgemm_llvm.mlir
```

**1.4 Generate Binary**

```bash
mlir-translate ./playground/spgemm_llvm.mlir \
  --mlir-to-llvmir \
  -o ./playground/spgemm_splyce.ll

clang -O3 ./playground/spgemm_splyce.ll \
  -mavx512f -mavx512vl \
  -fno-vectorize -fno-slp-vectorize \
  -L"$LLVM_INSTALL/lib" \
  -lmlir_c_runner_utils \
  -lmlir_runner_utils \
  -Wl,-rpath,"$LLVM_INSTALL/lib" \
  -o ./playground/test_benchmark_spgemm_splyce
```

### Example 2: Multi-Threaded SpGEMM (OpenMP)

**2.1 Lower Sparse Tensor to SCF (with parallelization)**

```bash
mlir-opt ./playground/spgemm.mlir \
  --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --pre-sparsification-rewrite \
  --empty-tensor-to-alloc-tensor \
  --sparse-reinterpret-map \
  --sparsification="parallelization-strategy=dense-outer-loop" \
  --stage-sparse-ops \
  --lower-sparse-ops-to-foreach \
  --sparse-reinterpret-map \
  --lower-sparse-foreach-to-scf \
  --loop-invariant-code-motion \
  --sparse-tensor-conversion \
  -o ./playground/spgemm_scf_parallel.mlir
```

**2.2 Apply Splyce Vectorization**

```bash
./build/bin/splyce-opt ./playground/spgemm_scf_parallel.mlir \
  --splyce="target-function=spgemm vector-width=4 phase-select=001" \
  -o ./playground/spgemm_splyce_parallel.mlir
```

**2.3 Lower SCF to LLVM Dialect (with OpenMP)**

```bash
mlir-opt ./playground/spgemm_splyce_parallel.mlir \
  --canonicalize \
  --sparsification-and-bufferization \
  --sparse-storage-specifier-to-llvm \
  --canonicalize \
  --convert-linalg-to-loops \
  --convert-vector-to-scf \
  --expand-realloc \
  --convert-scf-to-openmp \
  --convert-openmp-to-llvm \
  --convert-scf-to-cf \
  --expand-strided-metadata \
  --lower-affine \
  --convert-vector-to-llvm \
  --convert-complex-to-standard \
  --arith-expand \
  --convert-math-to-llvm \
  --convert-complex-to-libm \
  --convert-vector-to-llvm \
  --convert-to-llvm \
  --reconcile-unrealized-casts \
  -o ./playground/spgemm_llvm_parallel.mlir
```

**2.4 Generate Parallel Binary**

```bash
mlir-translate ./playground/spgemm_llvm_parallel.mlir \
  --mlir-to-llvmir \
  -o ./playground/spgemm_splyce_parallel.ll

clang -O3 ./playground/spgemm_splyce_parallel.ll \
  -mavx512f -mavx512vl \
  -fopenmp \
  -fno-vectorize \
  -fno-slp-vectorize \
  -L"$LLVM_INSTALL/lib" \
  -lmlir_c_runner_utils \
  -lmlir_runner_utils \
  -Wl,-rpath,"$LLVM_INSTALL/lib" \
  -o ./playground/test_benchmark_spgemm_splyce_parallel
```

---

## Evaluation

### Evaluation Metrics

To evaluate Splyce's performance improvements:

**1. Time Measurements**
```bash
# Without Splyce vectorization
time ./test_benchmark_spgemm_scalar

# With Splyce vectorization  
time ./test_benchmark_spgemm_splyce

# With Splyce + OpenMP parallelization
time ./test_benchmark_spgemm_splyce_parallel
```

**2. Compiler Intermediate Representations**

Compare IR at each stage:
```bash
# Check IR before vectorization
cat ./playground/spgemm_scf.mlir

# Check IR after vectorization
cat ./playground/spgemm_splyce.mlir

# Check generated LLVM IR
cat ./playground/spgemm_splyce.ll
```

**3. Performance Targets**

Typical improvements measured by Splyce:
- **Single-thread**: 2-4× speedup over scalar sparse dialect
- **Multi-thread**: 8-16× speedup with 8 threads + vectorization
- **Bandwidth**: Reduced memory stalls through better cache utilization
- **CPU Utilization**: Higher IPC (instructions per cycle) via masked SIMD

**4. Sparsity Patterns**

Test with different matrix densities and structures:
- **Regular sparsity**: ~10% density (uniform random)
- **Clustered sparsity**: ~10% density (block-like patterns)
- **Very sparse**: <1% density (extreme cases)

See [Troubleshooting](#troubleshooting) for performance analysis tools.

---

## FAQ

### Q1: Why not use a BlockSparse encoding?

BlockSparse encoding has several limitations:

- **Dimensionality mismatch**: BSR is a 2D format (`d0 floordiv B`, `d1 floordiv B`, `d0 mod B`, `d1 mod B`). There is no block dimension to exploit for generalized N-dimensional sparse tensors.
- **Intersection problem persists**: The scalar `scf.while` co-iteration bottleneck simply reappears at the block-coordinate level. The same branchy, un-vectorizable pointer-chasing occurs, just at coarser granularity.
- **Phantom nonzero overhead**: Real-world coordinate distributions are arbitrary. Forcing them into fixed-size blocks introduces phantom zeros that don't exist in the original data, inflating memory and bandwidth costs with no computational benefit.

### Q2: When is BlockSparse beneficial compared to Splyce vectorization?

BlockSparse is beneficial when:
- Data is **naturally block-structured**
- Nonzeros cluster into `B × B` tiles with >50% density per tile
- Fill-in overhead is acceptable
- Dense inner dimensions allow auto-vectorization to exploit SIMD

### Q3: When should I use standard sparse dialect instead of Splyce?

Use standard scalar sparse dialect lowering when:
- Nonzero density is **extremely low** (<0.1%)
- SIMD shuffle and mask-matching overhead outweighs the benefit of vectorization
- Memory bandwidth is not the bottleneck

The Splyce pass is an optimization on top of the sparse dialect; it adds overhead that only pays off when tensor operations have sufficient density and parallelism.

### Q4: What SIMD architectures does Splyce support?

Currently tested architectures:
- **AVX-512** (primary target)
- **AVX2** (supported, 256-bit vectors)
- **SSE** (supported, 128-bit vectors)

Splyce generates code compatible with all LLVM-supported SIMD targets via vector width configuration.

### Q5: How do I choose the vector width?

**Recommended vector widths:**
- `vector-width=4`: Maximum throughput

Choose based on your target hardware and data type (i32/i64 vs f32/f64).

---

## Troubleshooting

### Build Issues

**Error: "mlir-opt not found"**
```bash
# Ensure LLVM_INSTALL is set and PATH is updated
export PATH=$LLVM_INSTALL/bin:$PATH
which mlir-opt
```

**Error: "MLIR_DIR or LLVM_DIR not found"**
```bash
# Verify CMake paths
ls $LLVM_INSTALL/lib/cmake/mlir
ls $LLVM_INSTALL/lib/cmake/llvm

# If missing, rebuild LLVM with correct installation prefix
```

**Error: "Ninja: build halted"**
```bash
# Check for missing dependencies (zlib, python3, etc.)
# Rebuild with verbose output
ninja -C build -v 2>&1 | tail -50
```

### Runtime Issues

**Error: "splyce pass crashed on function X"**
1. Check that the target function exists: `mlir-opt input.mlir --print-op-graph`
2. Verify IR is in SCF dialect (after sparsification lowering)
3. Add debug output: `./build/bin/splyce-opt input.mlir --splyce=... -debug 2>&1 | head -100`

**Error: "Vector width not supported"**
```bash
# Use a standard vector width (4, 8, 16)
# Verify your CPU supports the target SIMD width with:
grep avx512f /proc/cpuinfo  # AVX-512
grep avx /proc/cpuinfo      # AVX2
```

**Performance worse after vectorization**
- Check sparsity density: if <1%, vectorization overhead may hurt
- Try different vector widths
- Compare generated LLVM IR: `cat ./spgemm_splyce.ll | grep -A5 -B5 mask`

### Debugging Tools

**Inspect generated IR at each stage:**
```bash
# After sparsification
mlir-opt input.mlir --print-op-graph > scf.txt

# After Splyce vectorization
./build/bin/splyce-opt input.mlir --splyce=... --print-op-graph > vectorized.txt

# Diff the IR
diff -u scf.txt vectorized.txt | head -50
```

**Profile the generated binary:**
```bash
# With performance counters (Linux)
perf stat ./test_benchmark_spgemm_splyce

# With VTune (if installed)
vtune -collect hotspots -r ./results -- ./test_benchmark_spgemm_splyce
```

**Analyze instruction distribution:**
```bash
llvm-objdump -d ./spgemm_splyce.ll | grep vmul | wc -l
llvm-objdump -d ./spgemm_splyce.ll | grep vpadd | wc -l
```

---

## Future Work

- [ ] Evaluate performance with BlockSparse formats
- [ ] Support for more complex co-iteration patterns (3+ iterators)
- [ ] Cost model for automatic vector width selection
- [ ] Integration with MLIR's vector distribution framework
- [ ] Support for other backends (GPU, specialized accelerators)

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test with existing benchmarks
4. Submit a pull request with performance results


---

## License

[Specify your license here - e.g., Apache 2.0, MIT, etc.]

---

## Support

For issues, questions, or discussions:
- **GitHub Issues**: [Report bugs or request features](../../issues)
- **Discussions**: [Join community discussions](../../discussions)
- **Email**: [kabilanen@gmail.com]



