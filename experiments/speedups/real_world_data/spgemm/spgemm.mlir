// SpGEMM: A(i,k) = Σ_j B(i,j) · C(j,k)
//
// Same kernel as spgemm_dn.mlir, but the output A is produced directly in
// CSR format instead of as a dense tensor.
//
// Iteration space: (i, k, j)
//   Parallel:  i, k
//   Reduction: j  (innermost)
//
// Encoding strategy for scf.while innermost loop (inner-product approach):
//   B in CSR — (i: dense, j: compressed):  for a fixed i, j is sparse-iterated
//   C in CSC — (k: dense, j: compressed):  for a fixed k, j is sparse-iterated
//   A in CSR — (i: dense, k: compressed):  nonzeros inserted in (i,k) order
//
//   Both B and C expose j as a compressed level in the reduction loop.
//   The sparsifier must co-iterate (intersect) the two j-lists, which
//   lowers to an scf.while merge loop at the innermost position. Since A's
//   compressed level (k) is inserted immediately after the j-reduction
//   completes for each (i,k) pair, the (i,k,j) iteration order is a valid
//   insertion order for CSR output.

// ---------------------------------------------------------------------------
// 1. Sparse Encodings
// ---------------------------------------------------------------------------

// B(i,j) in CSR: outer dimension i is dense, inner dimension j is compressed.
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// C(j,k) in CSC: stored column-major — outer level is k (dense), inner level
// is j (compressed).  Logically C is still indexed as (j, k); the permuted
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
llvm.func external @fopen(!llvm.ptr, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
llvm.func external @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32 attributes {llvm.emit_c_interface}
llvm.func external @fclose(!llvm.ptr) -> i32 attributes {llvm.emit_c_interface}

// ---------------------------------------------------------------------------
// 3. File-name Globals (null-terminated)
// ---------------------------------------------------------------------------
llvm.mlir.global internal constant @fileB("tensor_B.tns\00")
llvm.mlir.global internal constant @fileC("tensor_C.tns\00")
llvm.mlir.global internal constant @benchmarkFile("benchmark\00")
llvm.mlir.global internal constant @mode_w("w\00")
llvm.mlir.global internal constant @fmt_time("%f\n\00")

// ---------------------------------------------------------------------------
// 4. SpGEMM Kernel (CSR output)
//
//   A(i,k) = Σ_j B(i,j) · C(j,k)
//
//   Loop order after sparsification:
//     for i  (parallel, dense — from B's outer level, and A's outer level)
//       for k  (parallel, compressed — from C's outer CSC level, and A's
//               compressed level; nonzero inserted after the j-reduction)
//         scf.while  (co-iterate B's compressed j  ∩  C's compressed j)
//           A[i,k] += B[i,j] * C[j,k]
// ---------------------------------------------------------------------------
func.func @spgemm(
    %argB: tensor<?x?xf64, #CSR>,   // B(i,j)
    %argC: tensor<?x?xf64, #CSC>    // C(j,k)
) -> tensor<?x?xf64, #CSR> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Output dimensions: i from B's dim-0, k from C's dim-1
  %dim_i = tensor.dim %argB, %c0 : tensor<?x?xf64, #CSR>
  %dim_k = tensor.dim %argC, %c1 : tensor<?x?xf64, #CSC>

  // Sparse outputs start empty — no linalg.fill: nonzeros are inserted by
  // the sparsifier as the (i, k) iteration space is walked.
  %initA = tensor.empty(%dim_i, %dim_k) : tensor<?x?xf64, #CSR>

  // ---------------------------------------------------------------------------
  // 3D Linalg Generic
  //   Iteration order: (i, k, j)
  //   — i and k are the two parallel loops (outer)
  //   — j is the innermost reduction; compressed in both B and C
  //     → the sparse compiler emits an scf.while co-iteration here
  // ---------------------------------------------------------------------------
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, k, j) -> (i, j)>,   // B(i,j)
      affine_map<(i, k, j) -> (j, k)>,   // C(j,k)
      affine_map<(i, k, j) -> (i, k)>    // A(i,k)  out
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%argB, %argC : tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>)
    outs(%initA : tensor<?x?xf64, #CSR>) {
  ^bb0(%b: f64, %c: f64, %a_in: f64):
    // A[i,k] += B[i,j] * C[j,k]
    %prod  = arith.mulf %b, %c   : f64
    %a_out = arith.addf %a_in, %prod : f64
    linalg.yield %a_out : f64
  } -> tensor<?x?xf64, #CSR>

  return %result : tensor<?x?xf64, #CSR>
}

// ===========================================================================
// Main Benchmark Routine
// ===========================================================================
func.func @main() {
  %file_b = llvm.mlir.addressof @fileB : !llvm.ptr
  %file_c = llvm.mlir.addressof @fileC : !llvm.ptr

  %B = sparse_tensor.new %file_b : !llvm.ptr to tensor<?x?xf64, #CSR>
  %C = sparse_tensor.new %file_c : !llvm.ptr to tensor<?x?xf64, #CSC>

  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %iters = arith.constant 4 : index

  // ==========================================
  // Correctness check: compute one result (unused beyond dealloc — kept to
  // mirror spgemm_dn.mlir's warm-up-then-benchmark structure). Point
  // extraction is skipped here: unlike a dense tensor, cheaply reading a
  // single element out of a CSR result isn't a plain memory load, so
  // there's no equivalent cheap %val to (optionally) print.
  // ==========================================
  %ref_result = func.call @spgemm(%B, %C)
    : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %ref_result : tensor<?x?xf64, #CSR>

  // ==========================================
  // Benchmark SpGEMM: run 4 iterations, collect times, write to file
  // ==========================================
  %times = memref.alloc() : memref<4xf64>

  scf.for %iter = %c0 to %iters step %c1 {
    %start_iter = func.call @rtclock() : () -> f64
    %res = func.call @spgemm(%B, %C)
      : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64, #CSR>
    %end_iter = func.call @rtclock() : () -> f64
    %elapsed_iter = arith.subf %end_iter, %start_iter : f64
    func.call @printF64(%elapsed_iter) : (f64) -> ()
    func.call @printNewline() : () -> ()
    memref.store %elapsed_iter, %times[%iter] : memref<4xf64>
    bufferization.dealloc_tensor %res : tensor<?x?xf64, #CSR>
  }

  // Write times to file
  %file_ptr = llvm.mlir.addressof @benchmarkFile : !llvm.ptr
  %mode = llvm.mlir.addressof @mode_w : !llvm.ptr
  %fp = llvm.call @fopen(%file_ptr, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %format = llvm.mlir.addressof @fmt_time : !llvm.ptr

  scf.for %iter = %c0 to %iters step %c1 {
    %time_val = memref.load %times[%iter] : memref<4xf64>
    %dummy = llvm.call @fprintf(%fp, %format, %time_val) {var_callee_type = !llvm.func<i32 (ptr, ptr, ...)>} : (!llvm.ptr, !llvm.ptr, f64) -> i32
  }

  %dummy_close = llvm.call @fclose(%fp) : (!llvm.ptr) -> i32

  memref.dealloc %times : memref<4xf64>

  bufferization.dealloc_tensor %B : tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %C : tensor<?x?xf64, #CSC>

  return
}
