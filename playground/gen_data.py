import random

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
    nnz = int(total_elements * (1.0 - sparsity))
    
    with open(filename, 'w') as f:
        # 1. MLIR's magic string for arbitrary-rank tensors
        f.write("# extended FROSTT format\n")
        
        # 2. Rank (3 for a 3D tensor) and Total Non-Zeros (NNZ)
        f.write(f"3 {nnz}\n")
        
        # 3. Dimension sizes
        f.write(f"{dim1} {dim2} {dim3}\n")
        
        # 4. Write data (FROSTT is 1-indexed)
        # Randomly sample exactly `nnz` unique 1D coordinates and sort them
        # This avoids iterating over the entire multi-trillion element dense space
        coords = random.sample(range(total_elements), nnz)
        coords.sort()
        
        for idx in coords:
            i = idx // (dim2 * dim3)
            j = (idx % (dim2 * dim3)) // dim3
            k = idx % dim3
            val = random.uniform(0.5, 2.5)
            f.write(f"{i + 1} {j + 1} {k + 1} {val:.4f}\n")
                    
    print(f"Generated sparse 3D tensor: {filename} | Shape: ({dim1}, {dim2}, {dim3}) | NNZ: {nnz} | Sparsity: {sparsity}")

def generate_sparse_tns(filename, rows, cols):
    filename = "test_data.tns"
    num_elements = 10485760
    block_size = 32
    active_blocks = 262144
    nnz = active_blocks * block_size

    with open(filename, 'w') as f:
        # 1. MLIR's magic string for arbitrary-rank tensors
        f.write("# extended FROSTT format\n")
        
        # 2. Rank (1) and Total Non-Zeros (NNZ)
        f.write(f"1 {nnz}\n")
        
        # 3. Dimension sizes
        f.write(f"{num_elements}\n")
        
        blocks = random.sample(range(num_elements // block_size), active_blocks)
        blocks.sort()
        
        # 4. Write data (FROSTT is also 1-indexed)
        for block in blocks:
            for i in range(block_size):
                idx = (block * block_size) + i + 1 
                val = random.uniform(0.5, 2.5)
                f.write(f"{idx} {val:.4f}\n")

    print(f"Generated {filename}")

if __name__ == "__main__":
    # Generate Tensor D
    # generate_dense_tns("./sparse_dialect/tensor_D.tns", 28818, 42)
    
    # Generate Tensor C
    # generate_dense_tns("./sparse_dialect/tensor_C.tns", 9184, 42)
    generate_sparse_3d_tns("./sparse_dialect/tensor_B.tns", 12092, 9184, 28818, sparsity=0.999)