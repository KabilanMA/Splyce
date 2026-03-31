// ---------------------------------------------------------------------------
// 1. Define the Sparse Encodings (Block-Sparse is standard in DMRG)
// ---------------------------------------------------------------------------

// 3D Tensors (L, R, Bra, Ket)
#Sp3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : compressed)
}>

// 4D Tensor (W / MPO)
#Sp4D = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : dense, d1 : compressed, d2 : compressed, d3 : compressed)
}>

// ---------------------------------------------------------------------------
// External utilities and filenames
// ---------------------------------------------------------------------------
func.func private @rtclock() -> f64
func.func private @printF64(f64)

llvm.mlir.global internal constant @tensorL("tensor_L.tns\00")
llvm.mlir.global internal constant @tensorBra("tensor_Bra.tns\00")
llvm.mlir.global internal constant @tensorW("tensor_W.tns\00")
llvm.mlir.global internal constant @tensorKet("tensor_Ket.tns\00")
llvm.mlir.global internal constant @tensorR("tensor_R.tns\00")

// ---------------------------------------------------------------------------
// 2. The 5-Tensor Fused Kernel
// ---------------------------------------------------------------------------

func.func @dmrg_expectation_value(%argL: tensor<?x?x?xf64, #Sp3D>, 
                                  %argBra: tensor<?x?x?xf64, #Sp3D>, 
                                  %argW: tensor<?x?x?x?xf64, #Sp4D>, 
                                  %argKet: tensor<?x?x?xf64, #Sp3D>, 
                                  %argR: tensor<?x?x?xf64, #Sp3D>) -> tensor<f64> {

  // Initialize the 0D output tensor (a single scalar) to 0.0
  %initE = tensor.empty() : tensor<f64>
  %c0_f64 = arith.constant 0.0 : f64
  %zeroE = linalg.fill ins(%c0_f64 : f64) outs(%initE : tensor<f64>) -> tensor<f64>

  // ---------------------------------------------------------------------------
  // The Linalg Generic Op: 8D Iteration Space
  // Loop order: (a, b, c, d, e, f, x, y)
  // All 8 loops are "reduction" loops because the output is a 0D scalar.
  // ---------------------------------------------------------------------------
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(a, b, c, d, e, f, x, y) -> (a, d, x)>,       // Read L
      affine_map<(a, b, c, d, e, f, x, y) -> (a, b, c)>,       // Read Bra
      affine_map<(a, b, c, d, e, f, x, y) -> (d, e, c, f)>,    // Read W (MPO)
      affine_map<(a, b, c, d, e, f, x, y) -> (x, y, f)>,       // Read Ket
      affine_map<(a, b, c, d, e, f, x, y) -> (b, e, y)>,       // Read R
      affine_map<(a, b, c, d, e, f, x, y) -> ()>               // Read/Write E (Scalar)
    ],
    iterator_types = ["reduction", "reduction", "reduction", "reduction", 
                      "reduction", "reduction", "reduction", "reduction"]
  } ins(%argL, %argBra, %argW, %argKet, %argR : 
        tensor<?x?x?xf64, #Sp3D>, tensor<?x?x?xf64, #Sp3D>, tensor<?x?x?x?xf64, #Sp4D>, 
        tensor<?x?x?xf64, #Sp3D>, tensor<?x?x?xf64, #Sp3D>)
    outs(%zeroE : tensor<f64>) {
  ^bb0(%l: f64, %bra: f64, %w: f64, %ket: f64, %r: f64, %e_in: f64):
    
    // 5-way Fused Chain Multiplication
    // The hardware will execute this entirely inside SIMD registers
    %m1 = arith.mulf %l, %bra : f64
    %m2 = arith.mulf %m1, %w : f64
    %m3 = arith.mulf %m2, %ket : f64
    %m4 = arith.mulf %m3, %r : f64
    
    // Accumulate the continuous product into the scalar expectation value
    %e_out = arith.addf %e_in, %m4 : f64
    linalg.yield %e_out : f64
  } -> tensor<f64>

  return %result : tensor<f64>
}

// ---------------------------------------------------------------------------
// Main Benchmark Routine
// ---------------------------------------------------------------------------
func.func @main() {
  %file_l = llvm.mlir.addressof @tensorL : !llvm.ptr
  %file_bra = llvm.mlir.addressof @tensorBra : !llvm.ptr
  %file_w = llvm.mlir.addressof @tensorW : !llvm.ptr
  %file_ket = llvm.mlir.addressof @tensorKet : !llvm.ptr
  %file_r = llvm.mlir.addressof @tensorR : !llvm.ptr

  %l = sparse_tensor.new %file_l : !llvm.ptr to tensor<?x?x?xf64, #Sp3D>
  %bra = sparse_tensor.new %file_bra : !llvm.ptr to tensor<?x?x?xf64, #Sp3D>
  %w = sparse_tensor.new %file_w : !llvm.ptr to tensor<?x?x?x?xf64, #Sp4D>
  %ket = sparse_tensor.new %file_ket : !llvm.ptr to tensor<?x?x?xf64, #Sp3D>
  %r = sparse_tensor.new %file_r : !llvm.ptr to tensor<?x?x?xf64, #Sp3D>

  // --- Setup Benchmark Bounds ---
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %iters = arith.constant 100 : index

  // ==========================================
  // Save one result for comparison
  // ==========================================
  %ref_result = func.call @dmrg_expectation_value(%l, %bra, %w, %ket, %r)
    : (tensor<?x?x?xf64, #Sp3D>,
       tensor<?x?x?xf64, #Sp3D>,
       tensor<?x?x?x?xf64, #Sp4D>,
       tensor<?x?x?xf64, #Sp3D>,
       tensor<?x?x?xf64, #Sp3D>) -> tensor<f64>

  // Extract the single scalar value from the 0D output tensor and print it
  %val = tensor.extract %ref_result[] : tensor<f64>
  func.call @printF64(%val) : (f64) -> ()
  bufferization.dealloc_tensor %ref_result : tensor<f64>
  

  // ==========================================
  // Benchmark DMRG
  // ==========================================
  %start_dmrg = func.call @rtclock() : () -> f64

  scf.for %i = %c0 to %iters step %c1 {
    %res_dmrg = func.call @dmrg_expectation_value(%l, %bra, %w, %ket, %r)
      : (tensor<?x?x?xf64, #Sp3D>,
         tensor<?x?x?xf64, #Sp3D>,
         tensor<?x?x?x?xf64, #Sp4D>,
         tensor<?x?x?xf64, #Sp3D>,
         tensor<?x?x?xf64, #Sp3D>) -> tensor<f64>

    // Free memory to prevent OOM errors during the 100 iterations
    bufferization.dealloc_tensor %res_dmrg : tensor<f64>
  }

  %end_dmrg = func.call @rtclock() : () -> f64
  %elapsed_dmrg = arith.subf %end_dmrg, %start_dmrg : f64

  // --- Output Results ---
  // (Prints the total time for the 100 iterations. Divide by 100 later for average)
  func.call @printF64(%elapsed_dmrg) : (f64) -> ()

  return
}