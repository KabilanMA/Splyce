# Splyce (Slicing Sparse Data)

To translate Sparse Dialect code into SCF loops use the follows command:

`mlir-opt sparse.mlir --sparse-reinterpret-map --sparsification --lower-sparse-iteration-to-scf -o sparse.llvm.mlir`

