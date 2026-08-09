# Splyce: SIMD Vectorization of Sparse Coiteration

Splyce is an MLIR optimization pass that vectorizes 2-way co-iteration loops in sparse tensor computations. It replaces scalar `scf.while` epilogues with SIMD fast-lane execution paths, delivering significant speedups for sparse tensor kernels.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building and Installation](#building-and-installation)
- [Usage Examples](#usage-examples)
- [Evaluation](#evaluation)
  - [Automated Benchmark Script](#automated-benchmark-script)
  - [Evaluation Metrics](#evaluation-metrics)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)


---

## Overview

### What is Splyce?

Splyce tackles the vectorization bottleneck in sparse tensor computations. Traditional sparse dialect lowering produces scalar coiteration loops that:
- Check tensor coordinates on each iteration
- Branch based on coordinate comparisons
- Cannot be auto-vectorized by conventional compilers

Splyce recognizes these patterns and converts them into vectorized "fast lanes" using SIMD operations with masked execution, achieving **2-way parallelism** on existing CPU architectures.

### Key Features

- **2-Way Co-Iteration Vectorization**: Vectorizes pointer-chasing loops in sparse tensor operations
- **SIMD Fast-Lane Generation**: Creates masked SIMD execution paths for coordinate matching
- **Scalar Epilogue**: Keep the previous scalar coiteration loop as an epilogue.
- **Configurable Vector Width**: Supports flexible SIMD width selection (4, 8, 16, etc.)
- **Phase-Based Optimization**: Selective application of vectorization phases
- **Integration with MLIR Pipeline**: Works seamlessly with existing sparse dialect lowering

### Target Operations

- **SpGEMM** 
- **SpMTTKRP** 
- **SpMSpV** 
- **SpMMH** 
- **SpTTSpM**
- Other sparse coiteration operations.

---

## Prerequisites

### System Requirements

- **CMake** 3.28+
- **Ninja**
- **Git**
- **Python** 3.10+
- **C++ Compiler** (GCC 13+)

### Platform-Specific Installation

**Ubuntu / Debian:**
```bash
sudo apt-get install -y cmake ninja-build python3 python3-pip git zlib1g-dev
```

**Fedora / RHEL:**
```bash
sudo dnf install -y cmake ninja-build python3 git zlib-devel
```

**Without Root Access:**
```
You can proceed with cmake, python3, and git only.
```

---

## Building and Installation

### Step 1: Setup LLVM & MLIR

This project is tested against **LLVM 23.0.0git** (commit `6a6d432550598a59605ee062bd0e35c9d452c0c5`).

#### Option A: Build from Source (Recommended)

**Fast (recommended):** `llvm-project`'s full history is large, but GitHub lets you fetch a single commit directly, skipping all of it:
```bash
git init llvm-project && cd llvm-project
git remote add origin https://github.com/llvm/llvm-project.git
git fetch --depth 1 origin 6a6d432550598a59605ee062bd0e35c9d452c0c5
git checkout FETCH_HEAD
```

**Simple (slower):** clones the full history, then checks out the commit:
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 6a6d432550598a59605ee062bd0e35c9d452c0c5
```

**Build clang/lld from source (no system clang/lld required):**

Because clang itself is being built from source here, it will auto-detect a GCC installation on your system to borrow its C++ standard library/headers for building the runtimes. To pin this to a specific GCC instead of whatever the auto-detection picks, pass its install directory explicitly via `-DRUNTIMES_CMAKE_ARGS` below --- update the path (`/usr/lib/gcc/x86_64-linux-gnu/13`) to match your system's GCC location and version.

```bash
mkdir build
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir;lld;openmp" \
  -DLLVM_ENABLE_RUNTIMES="all" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="host;X86" \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_USE_LINKER=bfd \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install
  -DRUNTIMES_CMAKE_ARGS="-DCMAKE_C_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13;-DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
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

`$LLVM_INSTALL` here is the same one exported in Step 1 (Option A or B) - it's what points `MLIR_DIR`/`LLVM_DIR` at your built/installed LLVM. Also set `--gcc-install-dir` below to the same GCC install you built LLVM's runtimes against earlier, so Splyce links against matching C++ standard library headers/libs.

```bash
cd /path/to/splyce
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL/lib/cmake/llvm \
  -DCMAKE_C_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13" \
  -DCMAKE_CXX_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"

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

1. **Sparse Dialect -> SCF** (Sparsification)
2. **SCF -> Vectorized SCF** (Splyce Pass)
3. **Vectorized SCF -> LLVM Dialect** (Lowering)
4. **LLVM Dialect -> Binary** (Compilation)

Steps 1–3 can be run either as a single `splyce-opt` command, or as the
individual `mlir-opt` / `splyce-opt` stages it's built from. Both examples
below show both ways; pick whichever suits your workflow - a single
`splyce-opt` invocation for everyday use, or the broken-out stages when you
need to inspect or modify the IR between passes.

**`splyce-opt`'s pipeline-bundling flags:**
- `--sparsify-to-scf`: Runs the full linalg/sparse_tensor -> scf lowering pipeline (the `mlir-opt` stage below) in one flag.
- `--lower-to-llvm`: Runs the full scf -> LLVM dialect lowering pipeline (the other `mlir-opt` stage below) in one flag.
- `--parallelization`: Add to either of the above to switch to the OpenMP/parallel pipeline variant instead of the default single-threaded one.
- `--splyce="..."`: The vectorization pass itself (unchanged either way).


### Example 1: Single-Threaded SpGEMM

#### Option A: `splyce-opt` only (single command)

```bash
./build/bin/splyce-opt ./playground/spgemm.mlir \
  --sparsify-to-scf \
  --splyce="target-function=spgemm vector-width=4 phase-select=001" \
  --splyce-bufferize-restrict \
  --lower-to-llvm \
  -o ./playground/spgemm_llvm.mlir
```

**Splyce Options:**
- `target-function=<name>`: Target function to optimize
- `vector-width=<width>`: SIMD vector width (4, 8, 16, etc.)
- `phase-select=<phases>`: Bitmask for optimization phases

Continue with **1.4 Generate Binary** below.

#### Option B: `mlir-opt` + `splyce-opt` (step-by-step)

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

Both options produce the same `./playground/spgemm_llvm.mlir`.

**1.4 Generate Binary**

```bash
./build/bin/splyce-translate ./playground/spgemm_llvm.mlir \
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
**1.5 Run Binary**

The binary reads its input tensors from `tensor_B.tns` and `tensor_C.tns` in its working directory. Generate them first:
```bash
python3 ./playground/gen_data.py
```
This writes `tensor_B.tns` and `tensor_C.tns` into `playground/`, regardless of where you run it from.

Then run the binary from `playground/`, since it looks for those files relative to its own working directory:
```bash
cd playground && ./test_benchmark_spgemm_splyce
```

**1.6 Compare with Baseline**

Baseline binary can be generated from either with `mlir-opt` or using `splyce-opt`.

Using `mlir-opt`:
```bash
mlir-opt ./playground/spgemm.mlir \
  --sparsifier -o ./playground/spgemm_baseline.mlir
mlir-translate ./playground/spgemm_baseline.mlir --mlir-to-llvmir -o ./playground/spgemm_baseline.ll
```

Using `splyce-opt` with the `Splyce` pass disabled:
```bash
./build/bin/splyce-opt ./playground/spgemm.mlir \
  --sparsify-to-scf \
  --splyce-bufferize-restrict \
  --lower-to-llvm \
  -o ./playground/spgemm_baseline.mlir
./build/bin/splyce-translate ./playground/spgemm_baseline.mlir \
  --mlir-to-llvmir -o ./playground/spgemm_baseline.ll
```

Generate the binary from Clang:
```bash
clang -O3 ./playground/spgemm_baseline.ll \
  -mavx512f -mavx512vl \
  -fno-vectorize -fno-slp-vectorize \
  -L"$LLVM_INSTALL/lib" \
  -lmlir_c_runner_utils \
  -lmlir_runner_utils \
  -Wl,-rpath,"$LLVM_INSTALL/lib" \
  -o ./playground/test_benchmark_spgemm_baseline
```

and now run the baseline:
```bash
cd playground && ./test_benchmark_spgemm_baseline
```

### Example 2: Multi-Threaded SpGEMM (OpenMP)

#### Option A: `splyce-opt` only (single command)

```bash
./build/bin/splyce-opt ./playground/spgemm.mlir \
  --sparsify-to-scf --parallelization \
  --splyce="target-function=spgemm vector-width=4 phase-select=001" \
  --splyce-bufferize-restrict \
  --lower-to-llvm --parallelization \
  -o ./playground/spgemm_llvm_parallel.mlir
```

Continue with **2.4 Generate Parallel Binary** below.

#### Option B: `mlir-opt` + `splyce-opt` (step-by-step)

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
  --splyce-bufferize-restrict \
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

Both options produce the same `./playground/spgemm_llvm_parallel.mlir`.

**2.4 Generate Parallel Binary**

```bash
./build/bin/splyce-translate ./playground/spgemm_llvm_parallel.mlir \
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

**2.5 Run Parallel Binary**

This binary reads the same `tensor_B.tns`/`tensor_C.tns` inputs as the single-threaded example. If you already generated them in **1.5 Run Binary**, reuse them as-is; otherwise generate them first:
```bash
python3 ./playground/gen_data.py
```

Set `OMP_NUM_THREADS` to however many threads you want the OpenMP-parallelized loop to use, then run from `playground/`:
```bash
cd playground && OMP_NUM_THREADS=4 ./test_benchmark_spgemm_splyce_parallel
```

> For reliable timing (rather than a quick check), pin the run to real cores on a single NUMA node instead of leaving thread placement to the OS scheduler — e.g. `numactl --physcpubind=0-3 --membind=0 env OMP_NUM_THREADS=4 ./test_benchmark_spgemm_splyce_parallel` (or `taskset -c 0-3` if `numactl` isn't available). See [experiments/multicore/run.sh](experiments/multicore/run.sh) for the full pinning approach used in the benchmark suite.


> Note: `mlir-translate --mlir-to-llvmir` is a drop-in equivalent for `splyce-translate` in steps 1.4/2.4, if you're already relying on a separate LLVM install being on `$PATH`.


---

## Evaluation

### Automated Benchmark Script

Every experiment lives under the `experiments` directory, and each one has its own `compile.sh`/`run.sh` (or, for the real-world-data kernels, `compile.sh`/`run_suitesparse_benchmark.py`) — but all of them are driven from a single entry point, `experiments/run.sh`.

**Prerequisites** (in addition to [Building and Installation](#building-and-installation)):
- `LLVM_INSTALL` set, with `mlir-opt`, `mlir-translate`, and `clang` on `$PATH`.
- `build/bin/splyce-opt` and `build/bin/splyce-translate` already built.
- Every plotting experiment (anything other than `table2`/`realdata`) needs `matplotlib`. Set up a virtual environment once before running anything:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install matplotlib
  ```

Run every command below from inside the `experiments` directory.

`experiments` has four standalone experiment directories (`phase_ablation`, `vector_width`, `sparsity_scaling`, `multicore`) plus a `speedups/{synthetic_data,real_world_data}/{spgemm,spmmh,spmspv,spmttkrp,spttspm}` tree — 14 named experiments in total, each runnable individually by name. All of them can also run in one shot:
```bash
./run.sh all
```
> `all` runs every experiment above sequentially (including the SuiteSparse downloads for the `_realworld` kernels), so it needs network access and can take a while. `table2` and `realdata` also print/write their combined summary tables once their five kernels finish; `all` prints both.

#### Experiment 1 - Phase Ablation

Phase ablation is the experiment to find which phase configuration gives the better performance on SpGEMM kernel. Also the same experiment helps to find the TMA numbers.

You can simply run this experiment as follows:
```bash
./run.sh phase_ablation
```

This will generate two files:
1. `phase_ablation/tma_results.csv` - Table 1 - use `phase_ablation/reference.csv` for reference.
2. `phase_ablation/tma_breakdown_plot.png` - Figure 11 - use `phase_ablation/reference.png` for reference.


#### Experiment 2 - Experiment on Synthetic Data (Table 2)

This experiment automatically generates the required data and will produce the experimental results shown in Table 2 of the submitted paper for all 5 sparse tensor kernels.

You can simply run this experiment as follows:
```bash
./run.sh table2
```
This generates a `results.csv` file inside each kernel's directory under `speedups/synthetic_data`, plus one combined `speedups/synthetic_data/speedup_summary.csv` - use `speedups/synthetic_data/speedup_summary_reference.csv` for reference.

If you want to run just one of the five kernels, you can run it individually instead:
```bash
./run.sh spgemm_speedup
./run.sh spmspv_speedup
./run.sh spmttkrp_speedup
./run.sh spmmh_speedup
./run.sh spttspm_speedup
```

Once every kernel's `results.csv` exists, `python3 speedups/synthetic_data/print_speedup_summary.py` will print the combined table to the terminal and (re)write `speedup_summary.csv`.

#### Experiment 3 - Vector Width Ablation (Figure 12)

In different hardware depending on their capabilities the vector width to get the optimal performance can vary. For the hardware environment we test, we found that vector width 4 seems to show the significant speedup compared to other vector width in different sparsity factors.

You can simply run this experiment as follows:
```bash
./run.sh vector_width
```

This will generate two results files:
1. `vector_width/results.csv` contains all the execution numbers - use `vector_width/reference.csv` for reference.
2. `vector_width/vector_width_speedup_plot.png` - Figure 12 - use `vector_width/reference.png` for reference.

#### Experiment 4 - Sparsity Scaling (Figure 13)

This experiment is to show that Splyce's performance increase with increasing non zero density and even in extreme sparse data, Splyce does not show any significant degradation in performance.

You can simply run this experiment as follows:
```bash
./run.sh sparsity_scaling
```

This will generate two results files:
1. `sparsity_scaling/results.csv` contains all execution numbers - use `sparsity_scaling/reference.csv` for reference.
2. `sparsity_scaling/sparsity_scaling_plot.png` - Figure 13 - use `sparsity_scaling/reference.png` for reference.

#### Experiment 5 - Parallel Scalability (Figure 14)

This experiment is to show the performance speedup Splyce produce for single core is linearly propotional to its parallel core execution. This ensure all the performance improvement are pinned to a single core and use multiple cores gives the same amount of performance as number of cores being used.

You can simply run this experiment as follows:
```bash
./run.sh multicore
```

This will generate two results files:
1. `multicore/results.csv` contains all execution numbers - use `multicore/reference.csv` for reference.
2. `multicore/speedup_plot.png` - Figure 14 - use `multicore/reference.png` for reference.

> The parallel variant needs LLVM built with `-DLLVM_ENABLE_RUNTIMES=openmp` so that `libomp.so` is present under `$LLVM_INSTALL/lib`; `multicore/run.sh` detects its absence and skips the parallel variant automatically.

#### Experiment 6 - Performance on SuiteSparse Matrices (Table 3)

This is to show that Splyce not only gives performance on synthetic data but also on real-world matrices from the SuiteSparse matrix library. Matrices are not included in the repository - the script downloads each one it needs on demand (and, the first time any `_realworld` kernel runs, one-time scrapes `speedups/real_world_data/suitesparse/matrix_metadata.json` for download URLs).

You can simply run this experiment as follows:
```bash
./run.sh realdata
```

This downloads the required matrices, runs them on all five kernels, and produces `speedups/real_world_data/realworld_summary.csv` - compare against `speedups/real_world_data/realworld_summary_reference.csv` for reference.

If you want to run just one of the five kernels, you can run it individually instead:
```bash
./run.sh spgemm_realworld
./run.sh spmspv_realworld
./run.sh spmttkrp_realworld
./run.sh spmmh_realworld
./run.sh spttspm_realworld
```

Once every kernel's `<kernel>_realworld_results.csv` exists, `python3 speedups/real_world_data/print_realworld_summary.py` will print the combined table to the terminal and (re)write `realworld_summary.csv`.

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

## License

Splyce is licensed under the Apache License 2.0 with LLVM Exceptions, matching the license of the MLIR/LLVM infrastructure it builds on. See [LICENSE](LICENSE) for the full text.

