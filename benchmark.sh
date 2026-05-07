#!/usr/bin/env bash
# benchmark.sh — Build and run SpGEMM benchmarks: SCF baseline, Splyce, Splyce+OpenMP.
#
# Benchmark results are written to:
#   ./benchmark_scf              (baseline scalar SCF)
#   ./benchmark_splyce           (Splyce vectorized, single-thread)
#   ./benchmark_splyce_parallel  (Splyce vectorized + OpenMP)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMMON_LOWER_FLAGS=(
  --canonicalize
  --cse
  --loop-invariant-code-motion
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true"
  --convert-bufferization-to-memref
  --lower-vector-mask
  --convert-vector-to-scf
  --canonicalize
  --cse
  --expand-realloc
  --sparse-storage-specifier-to-llvm
  --convert-linalg-to-loops
  --lower-affine
  --canonicalize
  --cse
  --convert-scf-to-cf
  --expand-strided-metadata
  --finalize-memref-to-llvm
  --convert-vector-to-llvm
  --convert-math-to-llvm
  --convert-arith-to-llvm
  --convert-func-to-llvm
  --convert-cf-to-llvm
  --reconcile-unrealized-casts
)

# ---------------------------------------------------------------------------
# 1. Sparsify spgemm.mlir → single-thread SCF
# ---------------------------------------------------------------------------
echo "[1/9] Sparsify to SCF (single-thread) ..."
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

# ---------------------------------------------------------------------------
# Check for LLVM's libomp (required for the parallel variant)
# ---------------------------------------------------------------------------
HAVE_OMP=0
if find "${LLVM_INSTALL}/lib" /usr/lib /usr/lib/x86_64-linux-gnu \
     -maxdepth 2 -name "libomp.so*" 2>/dev/null | grep -q .; then
  HAVE_OMP=1
fi

if [[ $HAVE_OMP -eq 0 ]]; then
  echo "NOTE: libomp not found — skipping parallel variant."
  echo "      Rebuild LLVM with -DLLVM_ENABLE_RUNTIMES=openmp to enable it."
fi

# ---------------------------------------------------------------------------
# 2. Sparsify spgemm.mlir → parallel SCF (dense-outer-loop)
# ---------------------------------------------------------------------------
if [[ $HAVE_OMP -eq 1 ]]; then
  echo "[2/9] Sparsify to SCF (parallel) ..."
  mlir-opt ./playground/spgemm.mlir \
    --linalg-generalize-named-ops \
    --linalg-fuse-elementwise-ops \
    --pre-sparsification-rewrite \
    --empty-tensor-to-alloc-tensor \
    --sparse-reinterpret-map \
    "--sparsification=parallelization-strategy=dense-outer-loop" \
    --stage-sparse-ops \
    --lower-sparse-ops-to-foreach \
    --sparse-reinterpret-map \
    --lower-sparse-foreach-to-scf \
    --loop-invariant-code-motion \
    --sparse-tensor-conversion \
    -o ./playground/spgemm_scf_parallel.mlir
else
  echo "[2/9] Skipping parallel sparsification (no libomp)."
fi

# ---------------------------------------------------------------------------
# 3. Apply Splyce (single-thread)
# ---------------------------------------------------------------------------
echo "[3/9] Applying Splyce transformation (single-thread) ..."
./build/bin/splyce-opt ./playground/spgemm_scf.mlir \
  "--splyce=target-function=spgemm vector-width=4 phase-select=001" \
  --splyce-bufferize-restrict \
  -o ./playground/spgemm_splyce.mlir

# ---------------------------------------------------------------------------
# 4. Apply Splyce (parallel)
# ---------------------------------------------------------------------------
if [[ $HAVE_OMP -eq 1 ]]; then
  echo "[4/9] Applying Splyce transformation (parallel) ..."
  ./build/bin/splyce-opt ./playground/spgemm_scf_parallel.mlir \
    "--splyce=target-function=spgemm vector-width=4 phase-select=001" \
    --splyce-bufferize-restrict \
    -o ./playground/spgemm_splyce_parallel.mlir
else
  echo "[4/9] Skipping parallel Splyce transform (no libomp)."
fi

# ---------------------------------------------------------------------------
# 5. Lower SCF baseline → LLVM dialect
# ---------------------------------------------------------------------------
echo "[5/9] Lowering SCF baseline to LLVM dialect ..."
./build/bin/splyce-opt ./playground/spgemm_scf.mlir \
  --splyce-bufferize-restrict \
  -o ./playground/spgemm_scf_restrict.mlir

mlir-opt ./playground/spgemm_scf_restrict.mlir \
  "${COMMON_LOWER_FLAGS[@]}" \
  -o ./playground/spgemm_llvm_scf.mlir

# ---------------------------------------------------------------------------
# 6. Lower Splyce single-thread → LLVM dialect
# ---------------------------------------------------------------------------
echo "[6/9] Lowering Splyce (single-thread) to LLVM dialect ..."
mlir-opt ./playground/spgemm_splyce.mlir \
  "${COMMON_LOWER_FLAGS[@]}" \
  -o ./playground/spgemm_llvm_splyce.mlir

# ---------------------------------------------------------------------------
# 7. Lower Splyce parallel → LLVM dialect (OpenMP path)
# ---------------------------------------------------------------------------
if [[ $HAVE_OMP -eq 1 ]]; then
  echo "[7/9] Lowering Splyce (parallel) to LLVM dialect ..."
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
    -o ./playground/spgemm_llvm_splyce_parallel.mlir
else
  echo "[7/9] Skipping parallel lowering (no libomp)."
fi

# ---------------------------------------------------------------------------
# 8. Translate to LLVM IR and compile binaries
# ---------------------------------------------------------------------------
echo "[8/9] Translating to LLVM IR and compiling ..."

mlir-translate ./playground/spgemm_llvm_scf.mlir \
  --mlir-to-llvmir -o ./playground/spgemm_scf.ll
mlir-translate ./playground/spgemm_llvm_splyce.mlir \
  --mlir-to-llvmir -o ./playground/spgemm_splyce.ll
if [[ $HAVE_OMP -eq 1 ]]; then
  mlir-translate ./playground/spgemm_llvm_splyce_parallel.mlir \
    --mlir-to-llvmir -o ./playground/spgemm_splyce_parallel.ll
fi

clang -O3 ./playground/spgemm_scf.ll \
  -mavx512f -mavx512vl -fno-vectorize -fno-slp-vectorize \
  -L"${LLVM_INSTALL}/lib" -lmlir_c_runner_utils -lmlir_runner_utils \
  -Wl,-rpath,"${LLVM_INSTALL}/lib" \
  -o ./playground/test_benchmark_spgemm_scf

clang -O3 ./playground/spgemm_splyce.ll \
  -mavx512f -mavx512vl -fno-vectorize -fno-slp-vectorize \
  -L"${LLVM_INSTALL}/lib" -lmlir_c_runner_utils -lmlir_runner_utils \
  -Wl,-rpath,"${LLVM_INSTALL}/lib" \
  -o ./playground/test_benchmark_spgemm_splyce

if [[ $HAVE_OMP -eq 1 ]]; then
  clang -O3 ./playground/spgemm_splyce_parallel.ll \
    -mavx512f -mavx512vl -fno-vectorize -fno-slp-vectorize \
    -L"${LLVM_INSTALL}/lib" -lmlir_c_runner_utils -lmlir_runner_utils -lomp \
    -Wl,-rpath,"${LLVM_INSTALL}/lib" \
    -o ./playground/test_benchmark_spgemm_splyce_parallel
fi

# ---------------------------------------------------------------------------
# 9. Run benchmarks — binary writes "benchmark"; rename after each run
# ---------------------------------------------------------------------------
echo "[9/9] Running benchmarks ..."

echo "  -> SCF baseline ..."
./playground/test_benchmark_spgemm_scf || true
mv benchmark benchmark_scf

echo "  -> Splyce (single-thread) ..."
./playground/test_benchmark_spgemm_splyce || true
mv benchmark benchmark_splyce

if [[ $HAVE_OMP -eq 1 ]]; then
  echo "  -> Splyce (parallel / OpenMP) ..."
  ./playground/test_benchmark_spgemm_splyce_parallel || true
  mv benchmark benchmark_splyce_parallel
fi

# ---------------------------------------------------------------------------
# Print summary: total time + speedup vs SCF baseline
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Performance Summary"
echo "========================================"

summarize() {
  local label="$1"
  local file="$2"
  awk -v label="$label" '
    NF==1 { sum += $1; n++ }
    END {
      if (n > 0) {
        avg = sum / n
        printf "  %-30s  total=%8.4f s  avg=%8.4f s  (n=%d)\n", label, sum, avg, n
        print sum   # last line used by caller
      }
    }
  ' "$file"
}

# Capture totals for speedup calculation
scf_total=$(awk 'NF==1{s+=$1}END{print s}' benchmark_scf)
splyce_total=$(awk 'NF==1{s+=$1}END{print s}' benchmark_splyce)
n_iters=$(awk 'NF==1{n++}END{print n}' benchmark_scf)

awk -v label="SCF baseline" -v total="$scf_total" -v n="$n_iters" \
  'BEGIN{ printf "  %-30s  total=%8.4f s  avg=%8.4f s  (n=%d)\n", label, total, total/n, n }'
awk -v label="Splyce (single-thread)" -v total="$splyce_total" -v n="$n_iters" \
  'BEGIN{ printf "  %-30s  total=%8.4f s  avg=%8.4f s  (n=%d)\n", label, total, total/n, n }'

if [[ $HAVE_OMP -eq 1 ]]; then
  par_total=$(awk 'NF==1{s+=$1}END{print s}' benchmark_splyce_parallel)
  awk -v label="Splyce (parallel/OpenMP)" -v total="$par_total" -v n="$n_iters" \
    'BEGIN{ printf "  %-30s  total=%8.4f s  avg=%8.4f s  (n=%d)\n", label, total, total/n, n }'
fi

echo ""
awk -v scf="$scf_total" -v sp="$splyce_total" 'BEGIN{
  printf "  Splyce vs SCF baseline:  %.2fx speedup\n", scf/sp
}'
if [[ $HAVE_OMP -eq 1 ]]; then
  awk -v scf="$scf_total" -v sp="$splyce_total" -v pp="$par_total" 'BEGIN{
    printf "  Splyce parallel vs SCF:  %.2fx speedup\n", scf/pp
    printf "  Splyce parallel vs Splyce: %.2fx speedup\n", sp/pp
  }'
fi
echo "========================================"
echo ""
echo "Result files:"
echo "  $(pwd)/benchmark_scf"
echo "  $(pwd)/benchmark_splyce"
[[ $HAVE_OMP -eq 1 ]] && echo "  $(pwd)/benchmark_splyce_parallel"

# ---------------------------------------------------------------------------
# Cleanup: remove all generated intermediate files
# ---------------------------------------------------------------------------
echo ""
echo "Cleaning up intermediate files ..."
rm -f \
  ./playground/spgemm_scf_parallel.mlir \
  ./playground/spgemm_scf_restrict.mlir \
  ./playground/spgemm_splyce_parallel.mlir \
  ./playground/spgemm_llvm_scf.mlir \
  ./playground/spgemm_llvm_splyce.mlir \
  ./playground/spgemm_llvm_splyce_parallel.mlir \
  ./playground/spgemm_scf.ll \
  ./playground/spgemm_splyce.ll \
  ./playground/spgemm_splyce_parallel.ll \
  ./playground/test_benchmark_spgemm_scf \
  ./playground/test_benchmark_spgemm_splyce \
  ./playground/test_benchmark_spgemm_splyce_parallel 2>/dev/null || true
echo "Done."
