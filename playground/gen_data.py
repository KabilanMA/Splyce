import random
import math
import os

def generate_dense_tns(filename, rows, cols):
    nnz = rows * cols
    
    with open(filename, 'w') as f:
        # 1. MLIR's magic string for arbitrary-rank tensors
        f.write("# extended FROSTT format\n")
        
        # 2. Rank (2 for a matrix) and Total Non-Zeros (NNZ)
        f.write(f"2 {nnz}\n")
        
        # 3. Dimension sizes (Rows Cols)
        f.write(f"{rows} {cols}\n")
        
        # 4. Write data (FROSTT is 1-indexed)
        # For a completely dense tensor, we iterate through every single coordinate
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                # Generate a random float, formatted to 4 decimal places
                val = random.uniform(0.5, 2.5)
                f.write(f"{r} {c} {val:.4f}\n")
                
    print(f"Generated dense tensor: {filename} | Shape: ({rows}, {cols}) | NNZ: {nnz}")

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

def generate_sparse_4d_tns(filename, dim1, dim2, dim3, dim4, sparsity):
    total_elements = dim1 * dim2 * dim3 * dim4
    density = 1.0 - sparsity
    tmp_path = filename + ".tmp"

    actual_nnz = 0
    with open(tmp_path, 'w') as tmp:
        idx = 0
        while idx < total_elements:
            i = idx // (dim2 * dim3 * dim4)
            rem = idx % (dim2 * dim3 * dim4)
            j = rem // (dim3 * dim4)
            rem = rem % (dim3 * dim4)
            k = rem // dim4
            l = rem % dim4
            
            val = random.uniform(0.5, 2.5)
            tmp.write(f"{i + 1} {j + 1} {k + 1} {l + 1} {val:.4f}\n")
            actual_nnz += 1
            r = random.random()
            if r == 0.0:
                break
            idx += math.floor(math.log(r) / math.log(1.0 - density)) + 1

    with open(filename, 'w') as f:
        f.write("# extended FROSTT format\n")
        f.write(f"4 {actual_nnz}\n")
        f.write(f"{dim1} {dim2} {dim3} {dim4}\n")
        with open(tmp_path) as tmp:
            for line in tmp:
                f.write(line)
    os.remove(tmp_path)
                    
    print(f"Generated sparse 4D tensor: {filename} | Shape: ({dim1}, {dim2}, {dim3}, {dim4}) | NNZ: {actual_nnz} | Sparsity: {sparsity}")


if __name__ == "__main__":
    # A[i,j] = B[i,k,l] * D[l,j] * C[k,j]
    # Generate Tensor D
    # generate_dense_tns("./sparse_dialect/tensor_D.tns", 28818, 42)
    
    # Generate Tensor C
    # generate_dense_tns("./sparse_dialect/tensor_C.tns", 9184, 42)
    # generate_sparse_3d_tns("./sparse_dialect/tensor_B.tns", 1000, 100, 100, sparsity=0.9)
    # generate_dense_tns("./sparse_dialect/tensor_D.tns", 100, 42)
    # generate_dense_tns("./sparse_dialect/tensor_C.tns", 100, 42)

    # E = L(a,d,x) * Bra(a,b,c) * W(d,e,c,f) * Ket(x,y,f) * R(b,e,f)
    generate_sparse_4d_tns("./sparse_dialect/tensor_W.tns", 10, 10, 10, 10, sparsity=0.5)
    generate_sparse_3d_tns("./sparse_dialect/tensor_L.tns", 10, 10, 10, sparsity=0.5)
    generate_sparse_3d_tns("./sparse_dialect/tensor_Bra.tns", 10, 10, 10, sparsity=0.5)
    generate_sparse_3d_tns("./sparse_dialect/tensor_Ket.tns", 10, 10, 10, sparsity=0.5)
    generate_sparse_3d_tns("./sparse_dialect/tensor_R.tns", 10, 10, 10, sparsity=0.5)