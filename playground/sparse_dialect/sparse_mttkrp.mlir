// ---------------------------------------------------------------------------
// 1. Define the two formats from the text.
// ---------------------------------------------------------------------------

// Format 1: DSS-ijk (Default lexicographical index order)
// Outermost dense, two innermost compressed.
#DSS_ijk = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : compressed)
}>

// Format 2: DSS-ikj (Alternative format)
// Outermost dense, two innermost compressed, but dimensions 1 and 2 are swapped.
#DSS_ikj = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d2 : dense, d1 : compressed)
}>

#DenseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// Declare external utilities from the MLIR C Runner Library
func.func private @rtclock() -> f64
func.func private @printF64(f64)

// Define the filenames as LLVM globals (must be null-terminated strings)
llvm.mlir.global internal constant @tensorB("tensor_B.tns\00")
llvm.mlir.global internal constant @tensorD("tensor_D.tns\00")
llvm.mlir.global internal constant @tensorC("tensor_C.tns\00")

// ---------------------------------------------------------------------------
// 2. The MTTKRP Kernel (using the DSS_ijk format for this example)
// ---------------------------------------------------------------------------
// A[i,j] = B[i,k,l] * D[l,j] * C[k,j]
//
// Dimensions mapping based on the text:
// i: dim 0 of B (e.g., 12092)
// k: dim 1 of B (e.g., 9184)
// l: dim 2 of B (e.g., 28818)
// j: The factor rank, fixed to 42.
// ---------------------------------------------------------------------------

func.func @mttkrp_kernel(%argB: tensor<12092x9184x28818xf64, #DSS_ijk>, 
                         %argD: tensor<28818x42xf64, #DenseMatrix>, 
                         %argC: tensor<9184x42xf64, #DenseMatrix>) -> tensor<12092x42xf64> {

  // Retrieve dynamic dimension 'i' from the sparse tensor B to allocate output A
  %c0 = arith.constant 0 : index
//   %dim_i = tensor.dim %argB, %c0 : tensor<12092x9184x28818xf64, #DSS_ijk>
  
  // Initialize the dense output matrix A[i, j]
  // Note: j is statically known as 42
  %initA = tensor.empty() : tensor<12092x42xf64>
  %c0_f64 = arith.constant 0.0 : f64
  %zeroA = linalg.fill ins(%c0_f64 : f64) outs(%initA : tensor<12092x42xf64>) -> tensor<12092x42xf64>

  // The Linalg Generic Op defining the math and iteration space
  // Loop order: (i, j, k, l)
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j, k, l) -> (i, k, l)>,  // Read B[i,k,l]
      affine_map<(i, j, k, l) -> (l, j)>,     // Read D[l,j]
      affine_map<(i, j, k, l) -> (k, j)>,     // Read C[k,j]
      affine_map<(i, j, k, l) -> (i, j)>      // Read/Write A[i,j]
    ],
    // i, j are parallel loops. k, l are the "two reduction loops"
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%argB, %argD, %argC : tensor<12092x9184x28818xf64, #DSS_ijk>, tensor<28818x42xf64, #DenseMatrix>, tensor<9184x42xf64, #DenseMatrix>)
    outs(%zeroA : tensor<12092x42xf64>) {
  ^bb0(%b: f64, %d: f64, %c: f64, %a: f64):
    // Perform the multiplication: B[i,k,l] * D[l,j] * C[k,j]
    %m1 = arith.mulf %b, %d : f64
    %m2 = arith.mulf %m1, %c : f64
    
    // Accumulate into A[i,j]
    %sum = arith.addf %a, %m2 : f64
    linalg.yield %sum : f64
  } -> tensor<12092x42xf64>

  return %result : tensor<12092x42xf64>
}


// ---------------------------------------------------------------------------
// Main Benchmark Routine
// ---------------------------------------------------------------------------
func.func @main() {
  %file_b = llvm.mlir.addressof @tensorB : !llvm.ptr
  %file_d = llvm.mlir.addressof @tensorD : !llvm.ptr
  %file_c = llvm.mlir.addressof @tensorC : !llvm.ptr

  %bf = sparse_tensor.new %file_b : !llvm.ptr to tensor<12092x9184x28818xf64, #DSS_ijk>
  %df = sparse_tensor.new %file_d : !llvm.ptr to tensor<28818x42xf64, #DenseMatrix>
  %cf = sparse_tensor.new %file_c : !llvm.ptr to tensor<9184x42xf64, #DenseMatrix>

  // --- Setup Benchmark Bounds ---
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %iters = arith.constant 100 : index

  // ==========================================
  // Benchmark MTTKRP
  // ==========================================
  %start_mttkrp = func.call @rtclock() : () -> f64

  scf.for %i = %c0 to %iters step %iters {
    %res_mttkrp = func.call @mttkrp_kernel(%bf, %df, %cf)
      : (tensor<12092x9184x28818xf64, #DSS_ijk>,
         tensor<28818x42xf64, #DenseMatrix>,
         tensor<9184x42xf64, #DenseMatrix>) -> tensor<12092x42xf64>

    // Free memory to prevent OOM errors during the 100 iterations
    bufferization.dealloc_tensor %res_mttkrp : tensor<12092x42xf64>
  }

  %end_mttkrp = func.call @rtclock() : () -> f64
  %elapsed_mttkrp = arith.subf %end_mttkrp, %start_mttkrp : f64

  // --- Output Results ---
  // (Prints the total time for the 100 iterations. Divide by 100 later for average)
  func.call @printF64(%elapsed_mttkrp) : (f64) -> ()

  return
}