// SpMSpV: y(i) = Σ_j B(i,j) · x(j)
//
// Iteration space: (i, j)
//   Parallel:  i
//   Reduction: j  (innermost)
//
// Encoding strategy for scf.while innermost loop (inner-product approach):
//   B in CSR — (i: dense, j: compressed):  for a fixed i, j is sparse-iterated
//   x — (j: compressed):  the sparse vector exposes j as a compressed level
//
//   Both B and x expose j as a compressed level in the reduction loop.
//   The sparsifier must co-iterate (intersect) B's row j-list with x's
//   j-list, which lowers to an scf.while merge loop at the innermost
//   position — the same intersection pattern as SpGEMM, but with one
//   fewer parallel dimension (no k).

// ---------------------------------------------------------------------------
// 1. Sparse Encodings
// ---------------------------------------------------------------------------

// B(i,j) in CSR: outer dimension i is dense, inner dimension j is compressed.
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// x(j) as a sparse vector: the single dimension j is compressed.
#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

// ---------------------------------------------------------------------------
// 2. External Utilities
// ---------------------------------------------------------------------------
func.func private @rtclock() -> f64
llvm.func external @fopen(!llvm.ptr, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
llvm.func external @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32 attributes {llvm.emit_c_interface}
llvm.func external @fclose(!llvm.ptr) -> i32 attributes {llvm.emit_c_interface}
llvm.func external @printf(!llvm.ptr, ...) -> i32 attributes {llvm.emit_c_interface}

// ---------------------------------------------------------------------------
// 3. File-name Globals (null-terminated)
// ---------------------------------------------------------------------------
llvm.mlir.global internal constant @fileB("tensor_B.tns\00")
llvm.mlir.global internal constant @fileX("tensor_x.tns\00")
llvm.mlir.global internal constant @benchmarkFile("benchmark\00")
llvm.mlir.global internal constant @mode_w("w\00")
llvm.mlir.global internal constant @fmt_time("%f\n\00")
llvm.mlir.global internal constant @fmt_exec_time("Execution time: %f\n\00")

// ---------------------------------------------------------------------------
// 4. SpMSpV Kernel
//
//   y(i) = Σ_j B(i,j) · x(j)
//
//   Loop order after sparsification:
//     for i  (parallel, dense — from B's outer level)
//       scf.while  (co-iterate B's compressed row j  ∩  x's compressed j)
//         y[i] += B[i,j] * x[j]
// ---------------------------------------------------------------------------
func.func @spmspv(
    %argB: tensor<?x?xf64, #CSR>,        // B(i,j)
    %argX: tensor<?xf64, #SparseVector>  // x(j)
) -> tensor<?xf64> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Output dimension: i from B's dim-0
  %dim_i = tensor.dim %argB, %c0 : tensor<?x?xf64, #CSR>

  // Initialize dense output y(i) to zero
  %initY  = tensor.empty(%dim_i) : tensor<?xf64>
  %c0_f64 = arith.constant 0.0 : f64
  %zeroY  = linalg.fill ins(%c0_f64 : f64) outs(%initY : tensor<?xf64>) -> tensor<?xf64>

  // ---------------------------------------------------------------------------
  // 2D Linalg Generic
  //   Iteration order: (i, j)
  //   — i is the single parallel loop (outer)
  //   — j is the innermost reduction; compressed in both B and x
  //     → the sparse compiler emits an scf.while co-iteration here
  // ---------------------------------------------------------------------------
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,   // B(i,j)
      affine_map<(i, j) -> (j)>,      // x(j)
      affine_map<(i, j) -> (i)>       // y(i)  out
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%argB, %argX : tensor<?x?xf64, #CSR>, tensor<?xf64, #SparseVector>)
    outs(%zeroY : tensor<?xf64>) {
  ^bb0(%b: f64, %x: f64, %y_in: f64):
    // y[i] += B[i,j] * x[j]
    %prod  = arith.mulf %b, %x   : f64
    %y_out = arith.addf %y_in, %prod : f64
    linalg.yield %y_out : f64
  } -> tensor<?xf64>

  return %result : tensor<?xf64>
}

// ===========================================================================
// Main Benchmark Routine
// ===========================================================================
func.func @main() {
  %file_b = llvm.mlir.addressof @fileB : !llvm.ptr
  %file_x = llvm.mlir.addressof @fileX : !llvm.ptr

  %B = sparse_tensor.new %file_b : !llvm.ptr to tensor<?x?xf64, #CSR>
  %X = sparse_tensor.new %file_x : !llvm.ptr to tensor<?xf64, #SparseVector>

  // ==========================================
  // Run SpMSpV once, timed
  // ==========================================
  %start = func.call @rtclock() : () -> f64
  %res = func.call @spmspv(%B, %X)
    : (tensor<?x?xf64, #CSR>, tensor<?xf64, #SparseVector>) -> tensor<?xf64>
  %end = func.call @rtclock() : () -> f64
  %elapsed = arith.subf %end, %start : f64
  bufferization.dealloc_tensor %res : tensor<?xf64>

  // Print "Execution time: <t>" to the terminal
  %exec_fmt = llvm.mlir.addressof @fmt_exec_time : !llvm.ptr
  %dummy_print = llvm.call @printf(%exec_fmt, %elapsed) {var_callee_type = !llvm.func<i32 (ptr, ...)>} : (!llvm.ptr, f64) -> i32

  // Save the execution time to file
  %file_ptr = llvm.mlir.addressof @benchmarkFile : !llvm.ptr
  %mode = llvm.mlir.addressof @mode_w : !llvm.ptr
  %fp = llvm.call @fopen(%file_ptr, %mode) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  %format = llvm.mlir.addressof @fmt_time : !llvm.ptr
  %dummy_write = llvm.call @fprintf(%fp, %format, %elapsed) {var_callee_type = !llvm.func<i32 (ptr, ptr, ...)>} : (!llvm.ptr, !llvm.ptr, f64) -> i32
  %dummy_close = llvm.call @fclose(%fp) : (!llvm.ptr) -> i32

  bufferization.dealloc_tensor %B : tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %X : tensor<?xf64, #SparseVector>

  return
}
