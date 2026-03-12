func.func private @rtclock() -> f64
func.func private @printF64(f64)

llvm.mlir.global internal constant @filename("test_data.tns\00")

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

func.func @sparse_mul(%arg0: tensor<10485760xf32, #SparseVector>, 
                      %arg1: tensor<10485760xf32, #SparseVector>) -> tensor<10485760xf32> {
  
  %0 = tensor.empty() : tensor<10485760xf32>
  
  %c0_f32 = arith.constant 0.0 : f32
  %filled = linalg.fill ins(%c0_f32 : f32) outs(%0 : tensor<10485760xf32>) -> tensor<10485760xf32>
  
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(i) -> (i)>, // Read A (Sparse)
      affine_map<(i) -> (i)>, // Read B (Sparse)
      affine_map<(i) -> (i)>  // Write Result (Dense)
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<10485760xf32, #SparseVector>, tensor<10485760xf32, #SparseVector>)
    outs(%filled : tensor<10485760xf32>) {
  ^bb0(%inA: f32, %inB: f32, %outC: f32):
    %m1 = arith.mulf %inA, %inB : f32
    %m2 = arith.addf %m1, %inA : f32
    %m3 = arith.mulf %m2, %inB : f32
    %m4 = arith.addf %m3, %inA : f32
    %m5 = arith.mulf %m4, %inB : f32
    %m6 = arith.addf %m5, %inA : f32
    %m7 = arith.mulf %m6, %inB : f32
    %m8 = arith.addf %m7, %inA : f32
    %m9 = arith.mulf %m8, %inB : f32
    
    linalg.yield %m9 : f32
    // %mul = arith.mulf %inA, %inB : f32
    // linalg.yield %mul : f32
  } -> tensor<10485760xf32>
  
  return %1 : tensor<10485760xf32>
}

func.func @main() {
    // 1. Get the pointer to the global filename string
    %file_ptr = llvm.mlir.addressof @filename : !llvm.ptr
    
    // 2. Load the data from the file into our SparseVector format
    %sparse_input = sparse_tensor.new %file_ptr : !llvm.ptr to tensor<10485760xf32, #SparseVector>

    // 3. Setup benchmark loop bounds (100,000 iterations)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %iters = arith.constant 1000 : index

    // 4. Start Timer
    %start = func.call @rtclock() : () -> f64

    // 5. Execute the loop
    scf.for %i = %c0 to %iters step %c1 {
      %result = func.call @sparse_mul(%sparse_input, %sparse_input) 
        : (tensor<10485760xf32, #SparseVector>, tensor<10485760xf32, #SparseVector>) -> tensor<10485760xf32>
      
      bufferization.dealloc_tensor %result : tensor<10485760xf32>
    }

    // 6. Stop Timer and Calculate Elapsed Time
    %end = func.call @rtclock() : () -> f64
    %elapsed = arith.subf %end, %start : f64

    // 7. Print the time (in seconds)
    func.call @printF64(%elapsed) : (f64) -> ()

    return
  }