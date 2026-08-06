// SpGEMM: A(i,k) = Σ_j B(i,j) · C(j,k)
//
// This file compares two data layouts for C against the same B (CSR):
//
//   1. @spgemm_csr_csc — B in CSR, C in CSC.  j is compressed in both, so
//      the sparsifier emits an scf.while merge/co-iteration loop.
//   2. @spgemm_csr_csr — B in CSR, C in CSR.  j is compressed in B but
//      dense in C, so the sparsifier emits a nested-indexed (Gustavson)
//      loop instead — no merge required.
//
// The data on disk is only available as B in CSR and C in CSC, so using
// the CSR x CSR kernel requires first translating C from CSC to CSR
// (@csc_to_csr).  main() times all three costs separately:
//   - running SpGEMM directly on CSR x CSC (no translation needed)
//   - translating C from CSC to CSR
//   - running SpGEMM on CSR x CSR (after translation)

// ---------------------------------------------------------------------------
// 1. Sparse Encodings
// ---------------------------------------------------------------------------

// B(i,j) in CSR: outer dimension i is dense, inner dimension j is compressed.
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// C(j,k) in CSC as stored on disk: outer level is k (dense), inner level is
// j (compressed).  Logically C is still indexed as (j, k); the permuted
// storage means "for each column k, store the nonzero row indices j."
// d0 = j (first logical dim), d1 = k (second logical dim).
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

// ---------------------------------------------------------------------------
// 2. External Utilities
// ---------------------------------------------------------------------------
func.func private @rtclock() -> f64
func.func private @printNewline()
func.func private @printF64(f64)

// ---------------------------------------------------------------------------
// 3. File-name Globals (null-terminated)
// ---------------------------------------------------------------------------
llvm.mlir.global internal constant @fileB("tensor_B.tns\00")
llvm.mlir.global internal constant @fileC("tensor_C.tns\00")

// ---------------------------------------------------------------------------
// 4. SpGEMM Kernel: B(CSR) x C(CSC)  — merge co-iteration
//
//   Loop order after sparsification:
//     for i  (dense — B's outer level)
//       for k  (dense — C's outer CSC level)
//         scf.while  (co-iterate B's compressed j  ∩  C's compressed j)
//           A[i,k] += B[i,j] * C[j,k]
// ---------------------------------------------------------------------------
func.func @spgemm_csr_csc(
    %argB: tensor<?x?xf64, #CSR>,   // B(i,j)
    %argC: tensor<?x?xf64, #CSC>    // C(j,k)
) -> tensor<?x?xf64> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_i = tensor.dim %argB, %c0 : tensor<?x?xf64, #CSR>
  %dim_k = tensor.dim %argC, %c1 : tensor<?x?xf64, #CSC>

  %initA  = tensor.empty(%dim_i, %dim_k) : tensor<?x?xf64>
  %c0_f64 = arith.constant 0.0 : f64
  %zeroA  = linalg.fill ins(%c0_f64 : f64) outs(%initA : tensor<?x?xf64>) -> tensor<?x?xf64>

  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, k, j) -> (i, j)>,   // B(i,j)
      affine_map<(i, k, j) -> (j, k)>,   // C(j,k)
      affine_map<(i, k, j) -> (i, k)>    // A(i,k)  out
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%argB, %argC : tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>)
    outs(%zeroA : tensor<?x?xf64>) {
  ^bb0(%b: f64, %c: f64, %a_in: f64):
    %prod  = arith.mulf %b, %c   : f64
    %a_out = arith.addf %a_in, %prod : f64
    linalg.yield %a_out : f64
  } -> tensor<?x?xf64>

  return %result : tensor<?x?xf64>
}

// ---------------------------------------------------------------------------
// 5. CSC -> CSR Translation
//
//   Converts C from its on-disk CSC layout to the CSR layout the
//   spgemm_csr_csr kernel below expects.  sparse_tensor.convert re-sorts/
//   re-packs the pointer/index/value arrays; its cost is exactly the
//   "translation cost" we want to measure separately.
// ---------------------------------------------------------------------------
func.func @csc_to_csr(%in: tensor<?x?xf64, #CSC>) -> tensor<?x?xf64, #CSR> {
  %out = sparse_tensor.convert %in : tensor<?x?xf64, #CSC> to tensor<?x?xf64, #CSR>
  return %out : tensor<?x?xf64, #CSR>
}

// ---------------------------------------------------------------------------
// 6. SpGEMM Kernel: B(CSR) x C(CSR)  — nested-indexed (Gustavson)
//
//   Loop order after sparsification:
//     for i        (dense — B's outer level)
//       for j       (compressed — B's row i nonzeros)
//         for k      (compressed — C's row j nonzeros; no merge needed)
//           A[i,k] += B[i,j] * C[j,k]
// ---------------------------------------------------------------------------
func.func @spgemm_csr_csr(
    %argB: tensor<?x?xf64, #CSR>,   // B(i,j)
    %argC: tensor<?x?xf64, #CSR>    // C(j,k)
) -> tensor<?x?xf64> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_i = tensor.dim %argB, %c0 : tensor<?x?xf64, #CSR>
  %dim_k = tensor.dim %argC, %c1 : tensor<?x?xf64, #CSR>

  %initA  = tensor.empty(%dim_i, %dim_k) : tensor<?x?xf64>
  %c0_f64 = arith.constant 0.0 : f64
  %zeroA  = linalg.fill ins(%c0_f64 : f64) outs(%initA : tensor<?x?xf64>) -> tensor<?x?xf64>

  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, k, j) -> (i, j)>,   // B(i,j)
      affine_map<(i, k, j) -> (j, k)>,   // C(j,k)
      affine_map<(i, k, j) -> (i, k)>    // A(i,k)  out
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%argB, %argC : tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
    outs(%zeroA : tensor<?x?xf64>) {
  ^bb0(%b: f64, %c: f64, %a_in: f64):
    %prod  = arith.mulf %b, %c   : f64
    %a_out = arith.addf %a_in, %prod : f64
    linalg.yield %a_out : f64
  } -> tensor<?x?xf64>

  return %result : tensor<?x?xf64>
}

// ===========================================================================
// Main: single-shot cost comparison (no iteration loop, no file output)
// ===========================================================================
func.func @main() {
  %file_b = llvm.mlir.addressof @fileB : !llvm.ptr
  %file_c = llvm.mlir.addressof @fileC : !llvm.ptr

  // Data on disk: B in CSR, C in CSC.
  %B     = sparse_tensor.new %file_b : !llvm.ptr to tensor<?x?xf64, #CSR>
  %C_csc = sparse_tensor.new %file_c : !llvm.ptr to tensor<?x?xf64, #CSC>

  // -- Cost 1: translate C from CSC to CSR --
  %start_convert = func.call @rtclock() : () -> f64
  %C_csr = func.call @csc_to_csr(%C_csc) : (tensor<?x?xf64, #CSC>) -> tensor<?x?xf64, #CSR>
  %end_convert = func.call @rtclock() : () -> f64
  %elapsed_convert = arith.subf %end_convert, %start_convert : f64
  func.call @printF64(%elapsed_convert) : (f64) -> ()
  func.call @printNewline() : () -> ()

  // -- Cost 2: SpGEMM on CSR x CSR, after translation (nested/Gustavson) --
  %start_csr = func.call @rtclock() : () -> f64
  %res_csr = func.call @spgemm_csr_csr(%B, %C_csr)
    : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>) -> tensor<?x?xf64>
  %end_csr = func.call @rtclock() : () -> f64
  %elapsed_csr = arith.subf %end_csr, %start_csr : f64
  func.call @printF64(%elapsed_csr) : (f64) -> ()
  func.call @printNewline() : () -> ()
  bufferization.dealloc_tensor %res_csr : tensor<?x?xf64>

  // -- Cost 3: SpGEMM directly on CSR x CSC (merge co-iteration) --
  %start_csc = func.call @rtclock() : () -> f64
  %res_csc = func.call @spgemm_csr_csc(%B, %C_csc)
    : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64>
  %end_csc = func.call @rtclock() : () -> f64
  %elapsed_csc = arith.subf %end_csc, %start_csc : f64
  func.call @printF64(%elapsed_csc) : (f64) -> ()
  func.call @printNewline() : () -> ()
  bufferization.dealloc_tensor %res_csc : tensor<?x?xf64>

  bufferization.dealloc_tensor %C_csr : tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %B : tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %C_csc : tensor<?x?xf64, #CSC>

  return
}
