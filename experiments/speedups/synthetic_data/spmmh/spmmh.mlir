// SpMMH: A(i,j) = B(i,k) · C(k,j) · D(i,j)
//                = D(i,j) · Σ_k B(i,k) · C(k,j)
//
// A sparse matrix-multiply (SpMM) whose result is Hadamard-multiplied by a
// second sparse matrix D — hence "SpMMH".  D does not depend on the
// reduction index k, so it can be folded directly into the reduction body:
// Σ_k B(i,k)·C(k,j)·D(i,j) = D(i,j) · Σ_k B(i,k)·C(k,j).
//
// Iteration space: (i, j, k)
//   Parallel:  i, j
//   Reduction: k  (innermost)
//
// Encoding strategy:
//   B in CSC — (i,k): outer level k is dense, inner level i is compressed.
//     For a fixed k, the nonzero rows i are sparse-iterated.
//   C is dense (k,j): every (k,j) entry is present — no compression.
//   D in CSC — (i,j): outer level j is dense, inner level i is compressed.
//     For a fixed j, the nonzero rows i are sparse-iterated.
//
//   Unlike SpGEMM (where the two sparse operands share their compressed
//   dimension as the reduction index), here B and D both compress the same
//   *parallel* index i, but each drives it from a different dense outer
//   index (k for B, j for D).  Because D multiplies into every reduction
//   step, A(i,j) is only nonzero where B's row-i (for some k) and D's
//   row-i (for column j) coincide — the sparsifier must co-iterate
//   (intersect) B's and D's compressed i-lists, which lowers to an
//   scf.while merge loop, nested inside the outer j and k dense loops.

// ---------------------------------------------------------------------------
// 1. Sparse Encodings
// ---------------------------------------------------------------------------

// B(i,k) and D(i,j) in CSC: stored column-major — outer level is the second
// logical dim (dense), inner level is the first logical dim i (compressed).
// d0 = i (first logical dim), d1 = second logical dim (k for B, j for D).
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

// C(k,j) is fully dense.  sparse_tensor.new requires a sparse-annotated
// result type, so C is read as an all-dense encoded tensor, which the
// sparsifier treats identically to a plain dense tensor.
#Dense = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : dense)
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
// 4. SpMMH Kernel
//
//   A(i,j) = B(i,k) · C(k,j) · D(i,j)
//
//   Loop order after sparsification:
//     for j  (parallel, dense — from D's outer CSC level)
//       for k  (reduction, dense — from B's outer CSC level)
//         scf.while  (co-iterate B's compressed i  ∩  D's compressed i)
//           A[i,j] += B[i,k] * C[k,j] * D[i,j]
// ---------------------------------------------------------------------------
func.func @spmmh(
    %argB: tensor<?x?xf64, #CSC>,    // B(i,k)
    %argC: tensor<?x?xf64, #Dense>,  // C(k,j)
    %argD: tensor<?x?xf64, #CSC>     // D(i,j)
) -> tensor<?x?xf64> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Output dimensions: i from B's dim-0, j from D's dim-1
  %dim_i = tensor.dim %argB, %c0 : tensor<?x?xf64, #CSC>
  %dim_j = tensor.dim %argD, %c1 : tensor<?x?xf64, #CSC>

  // Initialize dense output A(i,j) to zero
  %initA  = tensor.empty(%dim_i, %dim_j) : tensor<?x?xf64>
  %c0_f64 = arith.constant 0.0 : f64
  %zeroA  = linalg.fill ins(%c0_f64 : f64) outs(%initA : tensor<?x?xf64>) -> tensor<?x?xf64>

  // ---------------------------------------------------------------------------
  // 3D Linalg Generic
  //   Iteration order: (i, j, k)
  //   — i and j are the two parallel loops (outer)
  //   — k is the innermost reduction; compressed in B and D indirectly
  //     via i, and D(i,j) is folded into the reduction body since it is
  //     invariant with respect to k
  // ---------------------------------------------------------------------------
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j, k) -> (i, k)>,  // B(i,k)
      affine_map<(i, j, k) -> (k, j)>,  // C(k,j)
      affine_map<(i, j, k) -> (i, j)>,  // D(i,j)
      affine_map<(i, j, k) -> (i, j)>   // A(i,j)  out
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%argB, %argC, %argD : tensor<?x?xf64, #CSC>, tensor<?x?xf64, #Dense>, tensor<?x?xf64, #CSC>)
    outs(%zeroA : tensor<?x?xf64>) {
  ^bb0(%b: f64, %c: f64, %d: f64, %a_in: f64):
    // A[i,j] += B[i,k] * C[k,j] * D[i,j]
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

  %B = sparse_tensor.new %file_b : !llvm.ptr to tensor<?x?xf64, #CSC>
  %C = sparse_tensor.new %file_c : !llvm.ptr to tensor<?x?xf64, #Dense>
  %D = sparse_tensor.new %file_d : !llvm.ptr to tensor<?x?xf64, #CSC>

  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %iters = arith.constant 6 : index

  // ==========================================
  // Correctness check: compute one result and print it
  // ==========================================
  %ref_result = func.call @spmmh(%B, %C, %D)
    : (tensor<?x?xf64, #CSC>, tensor<?x?xf64, #Dense>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64>

  %c0_idx = arith.constant 0 : index
  %val    = tensor.extract %ref_result[%c0_idx, %c0_idx] : tensor<?x?xf64>
  // func.call @printF64(%val) : (f64) -> ()
  bufferization.dealloc_tensor %ref_result : tensor<?x?xf64>

  // ==========================================
  // Benchmark SpMMH: run 25 iterations, collect times, write to file
  // ==========================================
  %times = memref.alloc() : memref<25xf64>

  scf.for %iter = %c0 to %iters step %c1 {
    %start_iter = func.call @rtclock() : () -> f64
    %res = func.call @spmmh(%B, %C, %D)
      : (tensor<?x?xf64, #CSC>, tensor<?x?xf64, #Dense>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64>
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

  bufferization.dealloc_tensor %B : tensor<?x?xf64, #CSC>
  bufferization.dealloc_tensor %C : tensor<?x?xf64, #Dense>
  bufferization.dealloc_tensor %D : tensor<?x?xf64, #CSC>

  return
}
