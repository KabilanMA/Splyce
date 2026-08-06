// SpTTSpM: A(i,j,r) = Σ_k B(i,j,k) · C(k,r)
//
// Iteration space: (i, j, r, k)
//   Parallel:  i, j, r
//   Reduction: k  (innermost)
//
// Encoding strategy for scf.while innermost loop (inner-product approach):
//   B in dense/compressed/compressed — (i: dense, j: compressed, k: compressed):
//     for a fixed i, j is sparse-iterated; for a fixed (i,j), k is sparse-iterated
//   C in CSC — (r: dense, k: compressed):  for a fixed r, k is sparse-iterated
//
//   Both B and C expose k as a compressed level in the reduction loop.
//   The sparsifier must co-iterate (intersect) B's row-(i,j) k-list with
//   C's column-r k-list, which lowers to an scf.while merge loop at the
//   innermost position — the same intersection pattern as SpGEMM, now
//   nested under the outer dense i and compressed j levels from B, and
//   the outer dense r level from C.

// ---------------------------------------------------------------------------
// 1. Sparse Encodings
// ---------------------------------------------------------------------------

// B(i,j,k): outer dimension i is dense, j and k are compressed.
#B3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : compressed)
}>

// C(k,r) in CSC: stored column-major — outer level is r (dense), inner level
// is k (compressed).  Logically C is still indexed as (k, r); the permuted
// storage means "for each column r, store the nonzero row indices k."
// d0 = k (first logical dim), d1 = r (second logical dim).
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
// 4. SpTTSpM Kernel
//
//   A(i,j,r) = Σ_k B(i,j,k) · C(k,r)
//
//   Loop order after sparsification:
//     for i  (parallel, dense — from B's outer level)
//       for j  (parallel — from B's compressed level)
//         for r  (parallel, dense — from C's outer CSC level)
//           scf.while  (co-iterate B's compressed k  ∩  C's compressed k)
//             A[i,j,r] += B[i,j,k] * C[k,r]
// ---------------------------------------------------------------------------
func.func @spttspm(
    %argB: tensor<?x?x?xf64, #B3D>,  // B(i,j,k)
    %argC: tensor<?x?xf64, #CSC>     // C(k,r)
) -> tensor<?x?x?xf64> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Output dimensions: i,j from B's dim-0/dim-1, r from C's dim-1
  %dim_i = tensor.dim %argB, %c0 : tensor<?x?x?xf64, #B3D>
  %dim_j = tensor.dim %argB, %c1 : tensor<?x?x?xf64, #B3D>
  %dim_r = tensor.dim %argC, %c1 : tensor<?x?xf64, #CSC>

  // Initialize dense output A(i,j,r) to zero
  %initA  = tensor.empty(%dim_i, %dim_j, %dim_r) : tensor<?x?x?xf64>
  %c0_f64 = arith.constant 0.0 : f64
  %zeroA  = linalg.fill ins(%c0_f64 : f64) outs(%initA : tensor<?x?x?xf64>) -> tensor<?x?x?xf64>

  // ---------------------------------------------------------------------------
  // 4D Linalg Generic
  //   Iteration order: (i, j, r, k)
  //   — i, j and r are the three parallel loops (outer)
  //   — k is the innermost reduction; compressed in both B and C
  //     → the sparse compiler emits an scf.while co-iteration here
  // ---------------------------------------------------------------------------
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j, r, k) -> (i, j, k)>,  // B(i,j,k)
      affine_map<(i, j, r, k) -> (k, r)>,     // C(k,r)
      affine_map<(i, j, r, k) -> (i, j, r)>   // A(i,j,r)  out
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%argB, %argC : tensor<?x?x?xf64, #B3D>, tensor<?x?xf64, #CSC>)
    outs(%zeroA : tensor<?x?x?xf64>) {
  ^bb0(%b: f64, %c: f64, %a_in: f64):
    // A[i,j,r] += B[i,j,k] * C[k,r]
    %prod  = arith.mulf %b, %c   : f64
    %a_out = arith.addf %a_in, %prod : f64
    linalg.yield %a_out : f64
  } -> tensor<?x?x?xf64>

  return %result : tensor<?x?x?xf64>
}

// ===========================================================================
// Main Benchmark Routine
// ===========================================================================
func.func @main() {
  %file_b = llvm.mlir.addressof @fileB : !llvm.ptr
  %file_c = llvm.mlir.addressof @fileC : !llvm.ptr

  %B = sparse_tensor.new %file_b : !llvm.ptr to tensor<?x?x?xf64, #B3D>
  %C = sparse_tensor.new %file_c : !llvm.ptr to tensor<?x?xf64, #CSC>

  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %iters = arith.constant 6 : index

  // ==========================================
  // Correctness check: compute one result and print it
  // ==========================================
  %ref_result = func.call @spttspm(%B, %C)
    : (tensor<?x?x?xf64, #B3D>, tensor<?x?xf64, #CSC>) -> tensor<?x?x?xf64>

  %c0_idx = arith.constant 0 : index
  %val    = tensor.extract %ref_result[%c0_idx, %c0_idx, %c0_idx] : tensor<?x?x?xf64>
  // func.call @printF64(%val) : (f64) -> ()
  bufferization.dealloc_tensor %ref_result : tensor<?x?x?xf64>

  // ==========================================
  // Benchmark SpTTSpM: run 25 iterations, collect times, write to file
  // ==========================================
  %times = memref.alloc() : memref<25xf64>

  scf.for %iter = %c0 to %iters step %c1 {
    %start_iter = func.call @rtclock() : () -> f64
    %res = func.call @spttspm(%B, %C)
      : (tensor<?x?x?xf64, #B3D>, tensor<?x?xf64, #CSC>) -> tensor<?x?x?xf64>
    %end_iter = func.call @rtclock() : () -> f64
    %elapsed_iter = arith.subf %end_iter, %start_iter : f64
    func.call @printF64(%elapsed_iter) : (f64) -> ()
    func.call @printNewline() : () -> ()
    memref.store %elapsed_iter, %times[%iter] : memref<25xf64>
    bufferization.dealloc_tensor %res : tensor<?x?x?xf64>
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

  bufferization.dealloc_tensor %B : tensor<?x?x?xf64, #B3D>
  bufferization.dealloc_tensor %C : tensor<?x?xf64, #CSC>

  return
}
