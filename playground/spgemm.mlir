// SpGEMM: C(i,k) = Σ_j A(i,j) · B(j,k)
//
// Iteration space: (i, k, j)
//   Parallel:  i, k
//   Reduction: j  (innermost)
//
// Encoding strategy for scf.while innermost loop (inner-product approach):
//   A in CSR — (i: dense, j: compressed):  for a fixed i, j is sparse-iterated
//   B in CSC — (k: dense, j: compressed):  for a fixed k, j is sparse-iterated
//
//   Both A and B expose j as a compressed level in the reduction loop.
//   The sparsifier must co-iterate (intersect) the two j-lists, which
//   lowers to an scf.while merge loop at the innermost position.

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
func.func private @rtclock() -> f64
func.func private @printNewline()
func.func private @printF64(f64)
llvm.func external @fopen(!llvm.ptr, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
llvm.func external @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32 attributes {llvm.emit_c_interface}
llvm.func external @fclose(!llvm.ptr) -> i32 attributes {llvm.emit_c_interface}

// ---------------------------------------------------------------------------
// 3. File-name Globals (null-terminated)
// ---------------------------------------------------------------------------
llvm.mlir.global internal constant @fileA("tensor_A.tns\00")
llvm.mlir.global internal constant @fileB("tensor_B.tns\00")
llvm.mlir.global internal constant @benchmarkFile("benchmark\00")
llvm.mlir.global internal constant @mode_w("w\00")
llvm.mlir.global internal constant @fmt_time("%f\n\00")

// ---------------------------------------------------------------------------
// 4. SpGEMM Kernel
//
//   C(i,k) = Σ_j A(i,j) · B(j,k)
//
//   Loop order after sparsification:
//     for i  (parallel, dense — from A's outer level)
//       for k  (parallel, dense — from B's outer CSC level)
//         scf.while  (co-iterate A's compressed j  ∩  B's compressed j)
//           C[i,k] += A[i,j] * B[j,k]
// ---------------------------------------------------------------------------
func.func @spgemm(
    %argA: tensor<?x?xf64, #CSR>,   // A(i,j)
    %argB: tensor<?x?xf64, #CSC>    // B(j,k)
) -> tensor<?x?xf64> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Output dimensions: i from A's dim-0, k from B's dim-1
  %dim_i = tensor.dim %argA, %c0 : tensor<?x?xf64, #CSR>
  %dim_k = tensor.dim %argB, %c1 : tensor<?x?xf64, #CSC>

  // Initialize dense output C(i,k) to zero
  %initC  = tensor.empty(%dim_i, %dim_k) : tensor<?x?xf64>
  %c0_f64 = arith.constant 0.0 : f64
  %zeroC  = linalg.fill ins(%c0_f64 : f64) outs(%initC : tensor<?x?xf64>) -> tensor<?x?xf64>

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
    outs(%zeroC : tensor<?x?xf64>) {
  ^bb0(%a: f64, %b: f64, %c_in: f64):
    // C[i,k] += A[i,j] * B[j,k]
    %prod  = arith.mulf %a, %b   : f64
    %c_out = arith.addf %c_in, %prod : f64
    linalg.yield %c_out : f64
  } -> tensor<?x?xf64>

  return %result : tensor<?x?xf64>
}

// ===========================================================================
// Main Benchmark Routine
// ===========================================================================
func.func @main() {
  %file_a = llvm.mlir.addressof @fileA : !llvm.ptr
  %file_b = llvm.mlir.addressof @fileB : !llvm.ptr

  %A = sparse_tensor.new %file_a : !llvm.ptr to tensor<?x?xf64, #CSR>
  %B = sparse_tensor.new %file_b : !llvm.ptr to tensor<?x?xf64, #CSC>

  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %iters = arith.constant 25 : index

  // ==========================================
  // Correctness check: compute one result and print it
  // ==========================================
  %ref_result = func.call @spgemm(%A, %B)
    : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64>

  %c0_idx = arith.constant 0 : index
  %val    = tensor.extract %ref_result[%c0_idx, %c0_idx] : tensor<?x?xf64>
  // func.call @printF64(%val) : (f64) -> ()
  bufferization.dealloc_tensor %ref_result : tensor<?x?xf64>

  // ==========================================
  // Benchmark SpGEMM: run 25 iterations, collect times, write to file
  // ==========================================
  %times = memref.alloc() : memref<25xf64>

  scf.for %iter = %c0 to %iters step %c1 {
    %start_iter = func.call @rtclock() : () -> f64
    %res = func.call @spgemm(%A, %B)
      : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64>
    %end_iter = func.call @rtclock() : () -> f64
    %elapsed_iter = arith.subf %end_iter, %start_iter : f64
    func.call @printF64(%elapsed_iter) : (f64) -> ()
    func.call @printNewline() : () -> ()
    memref.store %elapsed_iter, %times[%iter] : memref<25xf64>
    bufferization.dealloc_tensor %res : tensor<?x?xf64>
  }

  // Write times to file
  %file_ptr = llvm.mlir.addressof @benchmarkFile : !llvm.ptr
  %mode = llvm.mlir.addressof @mode_w : !llvm.ptr
  %fp = llvm.call @fopen(%file_ptr, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %format = llvm.mlir.addressof @fmt_time : !llvm.ptr

  scf.for %iter = %c0 to %iters step %c1 {
    %time_val = memref.load %times[%iter] : memref<25xf64>
    %dummy = llvm.call @fprintf(%fp, %format, %time_val) {var_callee_type = !llvm.func<i32 (ptr, ptr, ...)>} : (!llvm.ptr, !llvm.ptr, f64) -> i32
  }

  %dummy_close = llvm.call @fclose(%fp) : (!llvm.ptr) -> i32

  memref.dealloc %times : memref<25xf64>

  bufferization.dealloc_tensor %A : tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %B : tensor<?x?xf64, #CSC>

  return
}
