# Splyce (Slicing Sparse Data)

To translate Sparse Dialect code into SCF loops use the follows command:

`mlir-opt sparse.mlir --sparse-reinterpret-map --sparsification --lower-sparse-iteration-to-scf -o sparse.llvm.mlir`

### Questions to Answer

1. Why not use a BlockSparse encoding?

    - It's the wrong abstraction for tensors other than 2D. BSR is a 2D matrix format -- `(d0 floordiv B, d1 floordiv B, d0 mod B, d1 mod B)`. There is no block dimension to exploit. The encoding simply doesnn't apply.
    - It moves the intersection problem, not solves it. The scalar `scf.while` co-iteration loop reappears at the block-coordinate level. We still get the same branchy, un-vectorizable pointer-chasing, just a coarser granularity. The fundamental issue is untouched.
    - It introduces phantom nonzeros. Coordinates in our `.tns` file are arbitrary. Forcing them into fixed-size blocks requires storing zeros that don't exist in the original data, inflating memory and bandwidth for no computational gain.

2. What are the conditions under which BlockSparse will be beneficial compared to this vectorization?

For this, the data is natually block-structured. If real nonzeros cluster into BxB tiles with >50% density inside each tile, fill-in is cheap and the dense inner dimension give the auto-vectorizer free SIMD.

3. What are the condiions that direct sparse dialect will be beneficial comparared to this vectorization?

4. 
