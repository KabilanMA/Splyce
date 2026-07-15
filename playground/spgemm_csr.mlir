// SpGEMM: C(i,k) = Σ_j A(i,j) · B(j,k)
//
// Same kernel as spgemm.mlir, but the output C is produced directly in
// CSR format instead of as a dense tensor.
//
// Iteration space: (i, k, j)
//   Parallel:  i, k
//   Reduction: j  (innermost)
//
// Encoding strategy for scf.while innermost loop (inner-product approach):
//   A in CSR — (i: dense, j: compressed):  for a fixed i, j is sparse-iterated
//   B in CSC — (k: dense, j: compressed):  for a fixed k, j is sparse-iterated
//   C in CSR — (i: dense, k: compressed):  nonzeros inserted in (i,k) order
//
//   Both A and B expose j as a compressed level in the reduction loop.
//   The sparsifier must co-iterate (intersect) the two j-lists, which
//   lowers to an scf.while merge loop at the innermost position. Since C's
//   compressed level (k) is inserted immediately after the j-reduction
//   completes for each (i,k) pair, the (i,k,j) iteration order is a valid
//   insertion order for CSR output.

// ---------------------------------------------------------------------------
// 1. Sparse Encodings
// ---------------------------------------------------------------------------

// A(i,j) in CSR: outer dimension i is dense, inner dimension j is compressed.
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// B(j,k) in CSC: stored column-major — outer level is k (dense), inner level
// is j (compressed).  Logically B is still indexed as (j, k); the permuted
// storage means "for each column k, store the nonzero row indices j."
// d0 = j (first logical dim), d1 = k (second logical dim).
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

// ---------------------------------------------------------------------------
// 2. External Utilities
// ---------------------------------------------------------------------------
// NOTE: sparse_tensor.print is intentionally not used here. Its lowering
// (PreSparsificationRewritePass's PrintRewriter) builds a vector.print op
// without first loading the Vector dialect into the context, which crashes
// when --pre-sparsification-rewrite is run standalone (i.e. outside the
// composite --sparsifier pipeline that happens to load Vector earlier).
// Correctness is instead checked by converting the CSR result back to a
// dense tensor and printing it.
func.func private @printMemrefF64(%ptr : tensor<*xf64>)

// ---------------------------------------------------------------------------
// File-name Globals (null-terminated). sparse_tensor.new reads A and B from
// these small hand-written FROSTT-format files rather than via
// sparse_tensor.convert from a dense constant: the runtime-library-based
// --sparse-tensor-conversion pass only supports identity-mapped conversion
// (asserts "Run reinterpret-map before conversion" otherwise), so it cannot
// build the permuted CSC encoding for B that way. sparse_tensor.new goes
// through the COO-based runtime constructor instead, which does handle it —
// this matches how spgemm.mlir itself loads tensor_A.tns / tensor_B.tns.
// ---------------------------------------------------------------------------
llvm.mlir.global internal constant @fileA("playground/tensor_A_test.tns\00")
llvm.mlir.global internal constant @fileB("playground/tensor_B_test.tns\00")

// ---------------------------------------------------------------------------
// 3. SpGEMM Kernel (CSR output)
//
//   C(i,k) = Σ_j A(i,j) · B(j,k)
//
//   Loop order after sparsification:
//     for i  (parallel, dense — from A's outer level, and C's outer level)
//       for k  (parallel, compressed — from B's outer CSC level, and C's
//               compressed level; nonzero inserted after the j-reduction)
//         scf.while  (co-iterate A's compressed j  ∩  B's compressed j)
//           C[i,k] += A[i,j] * B[j,k]
// ---------------------------------------------------------------------------
func.func @spgemm(
    %argA: tensor<?x?xf64, #CSR>,   // A(i,j)
    %argB: tensor<?x?xf64, #CSC>    // B(j,k)
) -> tensor<?x?xf64, #CSR> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Output dimensions: i from A's dim-0, k from B's dim-1
  %dim_i = tensor.dim %argA, %c0 : tensor<?x?xf64, #CSR>
  %dim_k = tensor.dim %argB, %c1 : tensor<?x?xf64, #CSC>

  // Sparse outputs start empty — no linalg.fill: nonzeros are inserted by
  // the sparsifier as the (i, k) iteration space is walked.
  %initC = tensor.empty(%dim_i, %dim_k) : tensor<?x?xf64, #CSR>

  // ---------------------------------------------------------------------------
  // 3D Linalg Generic
  //   Iteration order: (i, k, j)
  //   — i and k are the two parallel loops (outer)
  //   — j is the innermost reduction; compressed in both A and B
  //     → the sparse compiler emits an scf.while co-iteration here
  // ---------------------------------------------------------------------------
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, k, j) -> (i, j)>,   // A(i,j)
      affine_map<(i, k, j) -> (j, k)>,   // B(j,k)
      affine_map<(i, k, j) -> (i, k)>    // C(i,k)  out
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%argA, %argB : tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>)
    outs(%initC : tensor<?x?xf64, #CSR>) {
  ^bb0(%a: f64, %b: f64, %c_in: f64):
    // C[i,k] += A[i,j] * B[j,k]
    %prod  = arith.mulf %a, %b   : f64
    %c_out = arith.addf %c_in, %prod : f64
    linalg.yield %c_out : f64
  } -> tensor<?x?xf64, #CSR>

  return %result : tensor<?x?xf64, #CSR>
}

// ===========================================================================
// Main: correctness check against simple, hand-computable data
//
//   A (2x3) = [[1, 0, 2],
//              [0, 3, 0]]
//   B (3x2) = [[1, 0],
//              [0, 2],
//              [3, 0]]
//
//   C = A x B = [[1*1+0*0+2*3, 1*0+0*2+2*0],   = [[7, 0],
//                [0*1+3*0+0*3, 0*0+3*2+0*0]]      [0, 6]]
// ===========================================================================
func.func @main() {
  %file_a = llvm.mlir.addressof @fileA : !llvm.ptr
  %file_b = llvm.mlir.addressof @fileB : !llvm.ptr

  %A = sparse_tensor.new %file_a : !llvm.ptr to tensor<?x?xf64, #CSR>
  %B = sparse_tensor.new %file_b : !llvm.ptr to tensor<?x?xf64, #CSC>

  %C = func.call @spgemm(%A, %B)
    : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64, #CSR>

  // Round-trip the CSR result back to dense purely so it can be printed
  // and hand-checked; the computation itself already happened in CSR.
  // Expect: [[7, 0], [0, 6]]
  %Cdense = sparse_tensor.convert %C : tensor<?x?xf64, #CSR> to tensor<?x?xf64>
  %Cprint = tensor.cast %Cdense : tensor<?x?xf64> to tensor<*xf64>
  func.call @printMemrefF64(%Cprint) : (tensor<*xf64>) -> ()

  bufferization.dealloc_tensor %A : tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %B : tensor<?x?xf64, #CSC>
  bufferization.dealloc_tensor %C : tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %Cdense : tensor<?x?xf64>

  return
}
