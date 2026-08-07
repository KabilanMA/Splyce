import random
import math
import os
import sys

def generate_sparse_3d_tns(filename, dim1, dim2, dim3, sparsity):
    total_elements = dim1 * dim2 * dim3
    density = 1.0 - sparsity
    tmp_path = filename + ".tmp"

    # Geometric skip sampling: O(1) memory, visits only ~nnz elements.
    # Each element is included with probability `density`. The gap to the
    # next included element follows Geom(density), computed as
    # floor(log(U) / log(1 - density)) for uniform U in (0, 1).
    actual_nnz = 0
    with open(tmp_path, 'w') as tmp:
        idx = 0
        while idx < total_elements:
            i = idx // (dim2 * dim3)
            j = (idx % (dim2 * dim3)) // dim3
            k = idx % dim3
            val = random.uniform(0.5, 2.5)
            tmp.write(f"{i + 1} {j + 1} {k + 1} {val:.4f}\n")
            actual_nnz += 1
            r = random.random()
            if r == 0.0:
                break
            idx += math.floor(math.log(r) / math.log(1.0 - density)) + 1

    # Write final file: header first, then data from temp file.
    with open(filename, 'w') as f:
        f.write("# extended FROSTT format\n")
        f.write(f"3 {actual_nnz}\n")
        f.write(f"{dim1} {dim2} {dim3}\n")
        with open(tmp_path) as tmp:
            for line in tmp:
                f.write(line)
    os.remove(tmp_path)
                    
    print(f"Generated sparse 3D tensor: {filename} | Shape: ({dim1}, {dim2}, {dim3}) | NNZ: {actual_nnz} | Sparsity: {sparsity}")

def generate_sparse_1d_tns(filename, dim1, sparsity):
    total_elements = dim1
    density = 1.0 - sparsity
    tmp_path = filename + ".tmp"

    actual_nnz = 0
    with open(tmp_path, 'w') as tmp:
        idx = 0
        while idx < total_elements:
            val = random.uniform(0.5, 2.5)
            tmp.write(f"{idx + 1} {val:.4f}\n")
            actual_nnz += 1
            r = random.random()
            if r == 0.0:
                break
            idx += math.floor(math.log(r) / math.log(1.0 - density)) + 1

    with open(filename, 'w') as f:
        f.write("# extended FROSTT format\n")
        f.write(f"1 {actual_nnz}\n")
        f.write(f"{dim1}\n")
        with open(tmp_path) as tmp:
            for line in tmp:
                f.write(line)
    os.remove(tmp_path)
                    
    print(f"Generated sparse 1D tensor: {filename} | Shape: ({dim1},) | NNZ: {actual_nnz} | Sparsity: {sparsity}")

def generate_dense_2d_tns(filename, rows, cols):
    nnz = rows * cols
    
    with open(filename, 'w') as f:
        f.write("# extended FROSTT format\n")
        
        f.write(f"2 {nnz}\n")
        
        f.write(f"{rows} {cols}\n")

        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                val = random.uniform(0.5, 2.5)
                f.write(f"{r} {c} {val:.4f}\n")
                
    print(f"Generated dense tensor: {filename} | Shape: ({rows}, {cols}) | NNZ: {nnz}")

def generate_sparse_2d_tns(filename, rows, cols, sparsity):
    total_elements = rows * cols
    density = 1.0 - sparsity
    tmp_path = filename + ".tmp"

    actual_nnz = 0
    with open(tmp_path, 'w') as tmp:
        idx = 0
        while idx < total_elements:
            i = idx // cols
            j = idx % cols
            val = random.uniform(0.5, 2.5)
            tmp.write(f"{i + 1} {j + 1} {val:.4f}\n")
            actual_nnz += 1
            r = random.random()
            if r == 0.0:
                break
            idx += math.floor(math.log(r) / math.log(1.0 - density)) + 1

    with open(filename, 'w') as f:
        f.write("# extended FROSTT format\n")
        f.write(f"2 {actual_nnz}\n")
        f.write(f"{rows} {cols}\n")
        with open(tmp_path) as tmp:
            for line in tmp:
                f.write(line)
    os.remove(tmp_path)
                    
    print(f"Generated sparse 2D tensor: {filename} | Shape: ({rows}, {cols}) | NNZ: {actual_nnz} | Sparsity: {sparsity}")

def generate_sparse_2d_tns_col_sparsity(filename, rows, cols, sparsity):
    # Same geometric skip sampling as generate_sparse_2d_tns, but idx walks
    # column-major (j outer, i inner) instead of row-major, so the sparsity
    # is applied down each column instead of across each row.
    total_elements = rows * cols
    density = 1.0 - sparsity
    tmp_path = filename + ".tmp"

    actual_nnz = 0
    with open(tmp_path, 'w') as tmp:
        idx = 0
        while idx < total_elements:
            j = idx // rows
            i = idx % rows
            val = random.uniform(0.5, 2.5)
            tmp.write(f"{i + 1} {j + 1} {val:.4f}\n")
            actual_nnz += 1
            r = random.random()
            if r == 0.0:
                break
            idx += math.floor(math.log(r) / math.log(1.0 - density)) + 1

    with open(filename, 'w') as f:
        f.write("# extended FROSTT format\n")
        f.write(f"2 {actual_nnz}\n")
        f.write(f"{rows} {cols}\n")
        with open(tmp_path) as tmp:
            for line in tmp:
                f.write(line)
    os.remove(tmp_path)

    print(f"Generated sparse 2D tensor (column sparsity): {filename} | Shape: ({rows}, {cols}) | NNZ: {actual_nnz} | Sparsity: {sparsity}")


def generate_phase_ablation_data():
    generate_sparse_2d_tns("./phase_ablation/tensor_B.tns", 5000, 5000, 0.95)
    generate_sparse_2d_tns("./phase_ablation/tensor_C.tns", 5000, 5000, 0.95)

def generate_speedup_data(kernel_name: str):
    if (kernel_name == "spgemm"):
        generate_sparse_2d_tns("./speedups/synthetic_data/spgemm/tensor_B.tns", 5000, 5000, 0.95)
        generate_sparse_2d_tns("./speedups/synthetic_data/spgemm/tensor_C.tns", 5000, 5000, 0.95)

    if (kernel_name == "spmmh"):
        generate_sparse_2d_tns("./speedups/synthetic_data/spmmh/tensor_B.tns", 5000, 5000, 0.95)
        generate_dense_2d_tns("./speedups/synthetic_data/spmmh/tensor_C.tns", 5000, 5000)
        generate_sparse_2d_tns("./speedups/synthetic_data/spmmh/tensor_D.tns", 5000, 5000, 0.95)

    if (kernel_name == "spmspv"):
        generate_sparse_2d_tns("./speedups/synthetic_data/spmspv/tensor_B.tns", 100, 100000000, 0.95)
        generate_sparse_1d_tns("./speedups/synthetic_data/spmspv/tensor_x.tns", 100000000, 0.95)

    if (kernel_name == "spmttkrp"):
        generate_sparse_3d_tns("./speedups/synthetic_data/spmttkrp/tensor_B.tns", 1000, 1000, 1000, 0.95)
        generate_sparse_2d_tns("./speedups/synthetic_data/spmttkrp/tensor_C.tns", 1000, 1000, 0.95)
        generate_sparse_2d_tns("./speedups/synthetic_data/spmttkrp/tensor_D.tns", 1000, 1000, 0.95)

    if (kernel_name == "spttspm"):
        generate_sparse_3d_tns("./speedups/synthetic_data/spttspm/tensor_B.tns", 500, 500, 500, 0.95)
        generate_sparse_2d_tns("./speedups/synthetic_data/spttspm/tensor_C.tns", 500, 500, 0.95)

def generate_spgemm_speedup_data():
    generate_speedup_data("spgemm")

def generate_spmmh_speedup_data():
    generate_speedup_data("spmmh")

def generate_spmspv_speedup_data():
    generate_speedup_data("spmspv")

def generate_spmttkrp_speedup_data():
    generate_speedup_data("spmttkrp")

def generate_spttspm_speedup_data():
    generate_speedup_data("spttspm")

def generate_multicore_data():
    # Same shape/sparsity as speedups/synthetic_data/spmttkrp — this
    # experiment reuses that kernel's compiled-for-parallel-execution
    # binaries (see multicore/compile.sh) for a core-count scaling
    # study, not a different dataset.
    generate_sparse_3d_tns("./multicore/tensor_B.tns", 1000, 1000, 1000, 0.95)
    generate_sparse_2d_tns("./multicore/tensor_C.tns", 1000, 1000, 0.95)
    generate_sparse_2d_tns("./multicore/tensor_D.tns", 1000, 1000, 0.95)

def generate_sparsity_scaling_data(sparsity_pct="1"):
    # Same 5000x5000 shape as speedups/synthetic_data/spgemm — this
    # experiment reuses that dimension but sweeps sparsity instead of
    # holding it fixed at 0.95. sparsity_pct is the *nonzero density*, in
    # percent (e.g. "0.01", "1", "10" — see sparsity_scaling/run.sh, which
    # regenerates data at each sweep point via this function before running
    # both of sparsity_scaling/compile.sh's binaries against it).
    sparsity = 1.0 - (float(sparsity_pct) / 100.0)
    generate_sparse_2d_tns("./sparsity_scaling/tensor_B.tns", 5000, 5000, sparsity)
    generate_sparse_2d_tns("./sparsity_scaling/tensor_C.tns", 5000, 5000, sparsity)

def generate_vector_width_data(sparsity_pct="1"):
    # Same 5000x5000 shape and sparsity_pct convention as
    # generate_sparsity_scaling_data — see vector_width/run.sh, which
    # regenerates data at each sweep point (1%, 5%, 10%) via this function
    # before running all 5 of vector_width/compile.sh's binaries against it.
    sparsity = 1.0 - (float(sparsity_pct) / 100.0)
    generate_sparse_2d_tns("./vector_width/tensor_B.tns", 5000, 5000, sparsity)
    generate_sparse_2d_tns("./vector_width/tensor_C.tns", 5000, 5000, sparsity)

# Maps an experiment directory name (as passed on the command line by
# run.sh) to the function that generates its tensor data. Any CLI arguments
# after the experiment name are forwarded to the generator (only
# generate_sparsity_scaling_data and generate_vector_width_data currently
# take any).
EXPERIMENTS = {
    "phase_ablation": generate_phase_ablation_data,
    "multicore": generate_multicore_data,
    "sparsity_scaling": generate_sparsity_scaling_data,
    "vector_width": generate_vector_width_data,
    "spgemm_speedup": generate_spgemm_speedup_data,
    "spmmh_speedup": generate_spmmh_speedup_data,
    "spmspv_speedup": generate_spmspv_speedup_data,
    "spmttkrp_speedup": generate_spmttkrp_speedup_data,
    "spttspm_speedup": generate_spttspm_speedup_data,
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 gen_data.py <experiment_name> [generator_args...]")
        print(f"Available experiments: {', '.join(EXPERIMENTS)}")
        sys.exit(1)

    experiment = sys.argv[1]
    generator = EXPERIMENTS.get(experiment)
    if generator is None:
        print(f"Unknown experiment: {experiment}")
        print(f"Available experiments: {', '.join(EXPERIMENTS)}")
        sys.exit(1)

    generator(*sys.argv[2:])

if __name__ == "__main__":
    main()