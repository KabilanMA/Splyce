import random

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