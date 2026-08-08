// SpMTTKRP: A(i,k) = Σ_l Σ_j B(i,k,l) · C(l,j) · D(k,j)
//
// Iteration space: (i, k, l, j)
//   Parallel:  i, k
//   Reduction: l, j  (l outer reduction, j innermost reduction)
//
// Encoding strategy for scf.while innermost loop (inner-product approach):
//   B in dense/compressed/compressed — (i: dense, k: compressed, l: compressed):
//     for a fixed i, k is sparse-iterated; for a fixed (i,k), l is sparse-iterated
//   C in CSR — (l: dense, j: compressed):  for a fixed l, j is sparse-iterated
//   D in CSR — (k: dense, j: compressed):  for a fixed k, j is sparse-iterated
//
//   B's compressed l level is walked directly (no intersection needed — C is
//   densely indexable by l), giving random access into C's row l.  The
//   innermost j loop, however, is shared by both C's row-l and D's row-k,
//   each of which expose j as a compressed level.  The sparsifier must
//   co-iterate (intersect) these two j-lists, which lowers to an scf.while
//   merge loop at the innermost position — the same intersection pattern
//   as SpGEMM, now nested one level deeper under the l reduction.

// ---------------------------------------------------------------------------
// 1. Sparse Encodings
// ---------------------------------------------------------------------------

// B(i,k,l): outer dimension i is dense, k and l are compressed.
#B3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : compressed)
}>

// C(l,j) and D(k,j) in CSR: outer dimension is dense, inner dimension compressed.
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
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
llvm.mlir.global internal constant @fileD("tensor_D.tns\00")
llvm.mlir.global internal constant @benchmarkFile("benchmark\00")
llvm.mlir.global internal constant @mode_w("w\00")
llvm.mlir.global internal constant @fmt_time("%f\n\00")

// ---------------------------------------------------------------------------
// 4. SpMTTKRP Kernel
//
//   A(i,k) = Σ_l Σ_j B(i,k,l) · C(l,j) · D(k,j)
//
//   Loop order after sparsification:
//     for i  (parallel, dense — from B's outer level)
//       for k  (parallel — from B's compressed level / D's outer CSR level)
//         for l  (reduction — B's compressed level, directly selects C's row)
//           scf.while  (co-iterate C[l,:]'s compressed j  ∩  D[k,:]'s compressed j)
//             A[i,k] += B[i,k,l] * C[l,j] * D[k,j]
// ---------------------------------------------------------------------------
func.func @spmttkrp(
    %argB: tensor<?x?x?xf64, #B3D>,  // B(i,k,l)
    %argC: tensor<?x?xf64, #CSR>,    // C(l,j)
    %argD: tensor<?x?xf64, #CSR>     // D(k,j)
) -> tensor<?x?xf64> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Output dimensions: i from B's dim-0, k from B's dim-1
  %dim_i = tensor.dim %argB, %c0 : tensor<?x?x?xf64, #B3D>
  %dim_k = tensor.dim %argB, %c1 : tensor<?x?x?xf64, #B3D>

  // Initialize dense output A(i,k) to zero
  %initA  = tensor.empty(%dim_i, %dim_k) : tensor<?x?xf64>
  %c0_f64 = arith.constant 0.0 : f64
  %zeroA  = linalg.fill ins(%c0_f64 : f64) outs(%initA : tensor<?x?xf64>) -> tensor<?x?xf64>

  // ---------------------------------------------------------------------------
  // 4D Linalg Generic
  //   Iteration order: (i, k, l, j)
  //   — i and k are the two parallel loops (outer)
  //   — l and j are the two reduction loops (innermost);
  //     j is compressed in both C and D → the sparse compiler emits an
  //     scf.while co-iteration at the innermost position
  // ---------------------------------------------------------------------------
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, k, l, j) -> (i, k, l)>,  // B(i,k,l)
      affine_map<(i, k, l, j) -> (l, j)>,     // C(l,j)
      affine_map<(i, k, l, j) -> (k, j)>,     // D(k,j)
      affine_map<(i, k, l, j) -> (i, k)>      // A(i,k)  out
    ],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%argB, %argC, %argD : tensor<?x?x?xf64, #B3D>, tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
    outs(%zeroA : tensor<?x?xf64>) {
  ^bb0(%b: f64, %c: f64, %d: f64, %a_in: f64):
    // A[i,k] += B[i,k,l] * C[l,j] * D[k,j]
    %prod1 = arith.mulf %b, %c     : f64
    %prod2 = arith.mulf %prod1, %d : f64
    %a_out = arith.addf %a_in, %prod2 : f64
    linalg.yield %a_out : f64
  } -> tensor<?x?xf64>

  return %result : tensor<?x?xf64>
}

// ===========================================================================
// Main Benchmark Routine
// ===========================================================================
func.func @main() {
  %file_b = llvm.mlir.addressof @fileB : !llvm.ptr
  %file_c = llvm.mlir.addressof @fileC : !llvm.ptr
  %file_d = llvm.mlir.addressof @fileD : !llvm.ptr

  %B = sparse_tensor.new %file_b : !llvm.ptr to tensor<?x?x?xf64, #B3D>
  %C = sparse_tensor.new %file_c : !llvm.ptr to tensor<?x?xf64, #CSR>
  %D = sparse_tensor.new %file_d : !llvm.ptr to tensor<?x?xf64, #CSR>

  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %iters = arith.constant 6 : index

  // ==========================================
  // Correctness check: compute one result and print it
  // ==========================================
  %ref_result = func.call @spmttkrp(%B, %C, %D)
    : (tensor<?x?x?xf64, #B3D>, tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>) -> tensor<?x?xf64>

  %c0_idx = arith.constant 0 : index
  %val    = tensor.extract %ref_result[%c0_idx, %c0_idx] : tensor<?x?xf64>
  // func.call @printF64(%val) : (f64) -> ()
  bufferization.dealloc_tensor %ref_result : tensor<?x?xf64>

  // ==========================================
  // Benchmark SpMTTKRP: run 6 iterations, collect times, write to file
  // ==========================================
  %times = memref.alloc() : memref<6xf64>

  scf.for %iter = %c0 to %iters step %c1 {
    %start_iter = func.call @rtclock() : () -> f64
    %res = func.call @spmttkrp(%B, %C, %D)
      : (tensor<?x?x?xf64, #B3D>, tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>) -> tensor<?x?xf64>
    %end_iter = func.call @rtclock() : () -> f64
    %elapsed_iter = arith.subf %end_iter, %start_iter : f64
    func.call @printF64(%elapsed_iter) : (f64) -> ()
    func.call @printNewline() : () -> ()
    memref.store %elapsed_iter, %times[%iter] : memref<6xf64>
    bufferization.dealloc_tensor %res : tensor<?x?xf64>
  }

  // Write times to file
  %file_ptr = llvm.mlir.addressof @benchmarkFile : !llvm.ptr
  %mode = llvm.mlir.addressof @mode_w : !llvm.ptr
  %fp = llvm.call @fopen(%file_ptr, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %format = llvm.mlir.addressof @fmt_time : !llvm.ptr

  scf.for %iter = %c0 to %iters step %c1 {
    %time_val = memref.load %times[%iter] : memref<6xf64>
    %dummy = llvm.call @fprintf(%fp, %format, %time_val) {var_callee_type = !llvm.func<i32 (ptr, ptr, ...)>} : (!llvm.ptr, !llvm.ptr, f64) -> i32
  }

  %dummy_close = llvm.call @fclose(%fp) : (!llvm.ptr) -> i32

  memref.dealloc %times : memref<6xf64>

  bufferization.dealloc_tensor %B : tensor<?x?x?xf64, #B3D>
  bufferization.dealloc_tensor %C : tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %D : tensor<?x?xf64, #CSR>

  return
}
