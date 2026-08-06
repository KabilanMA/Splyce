export LLVM_INSTALL=/home/kabilan/llvm-dev

export PATH="$LLVM_INSTALL/bin:$PATH"

./compile.sh

rm -f test_benchmark_spgemm_scf test_benchmark_spgemm_splyce_phase_*