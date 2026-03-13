func.func private @rtclock() -> f64
func.func private @printF64(f64)

llvm.mlir.global internal constant @tensorX("tensor.tns\00")
llvm.mlir.global internal constant @tensorB("tensor.tns\00")
llvm.mlir.global internal constant @tensorC("tensor.tns\00")

// Sparse 3D tensor encoding
#Sparse3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : compressed)
}>

#Sparse2D = #sparse_tensor.encoding<{
    map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>


func.func @mttkrp(
  %X: tensor<256x256x256xf32, #Sparse3D>,
  %B: tensor<256x64xf32, #Sparse2D>,
  %C: tensor<256x64xf32, #Sparse2D>
) -> tensor<256x64xf32> {

  // Output tensor M(i,r)
  %empty = tensor.empty() : tensor<256x64xf32>

  %c0 = arith.constant 0.0 : f32
  %init = linalg.fill
      ins(%c0 : f32)
      outs(%empty : tensor<256x64xf32>)
      -> tensor<256x64xf32>

  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,j,k,r) -> (i,j,k)>, // X
      affine_map<(i,j,k,r) -> (j,r)>,   // B
      affine_map<(i,j,k,r) -> (k,r)>,   // C
      affine_map<(i,j,k,r) -> (i,r)>    // M
    ],
    iterator_types = ["parallel", "reduction", "reduction", "parallel"]
  }
  ins(%X, %B, %C :
      tensor<256x256x256xf32, #Sparse3D>,
      tensor<256x64xf32, #Sparse2D>,
      tensor<256x64xf32, #Sparse2D>)
  outs(%init : tensor<256x64xf32>) {

  ^bb0(%x: f32, %b: f32, %c: f32, %m: f32):

    %t1 = arith.mulf %x, %b : f32
    %t2 = arith.mulf %t1, %c : f32
    %t3 = arith.addf %m, %t2 : f32

    linalg.yield %t3 : f32

  } -> tensor<256x64xf32>

  return %result : tensor<256x64xf32>
}

func.func @main() {

  %file_x = llvm.mlir.addressof @tensorX : !llvm.ptr
  %file_b = llvm.mlir.addressof @tensorB : !llvm.ptr
  %file_c = llvm.mlir.addressof @tensorC : !llvm.ptr

  // Load sparse tensor
  %X = sparse_tensor.new %file_x : !llvm.ptr to tensor<256x256x256xf32, #Sparse3D>
  %Bf = sparse_tensor.new %file_b : !llvm.ptr to tensor<256x64xf32, #Sparse2D>
  %Cf = sparse_tensor.new %file_c : !llvm.ptr to tensor<256x64xf32, #Sparse2D>

  %c0i = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %iters = arith.constant 100 : index

  %start = func.call @rtclock() : () -> f64

  scf.for %i = %c0i to %iters step %c1 {

    %res = func.call @mttkrp(%X, %Bf, %Cf)
      : (tensor<256x256x256xf32, #Sparse3D>,
         tensor<256x64xf32, #Sparse2D>,
         tensor<256x64xf32, #Sparse2D>)
        -> tensor<256x64xf32>

    bufferization.dealloc_tensor %res : tensor<256x64xf32>
  }

  %end = func.call @rtclock() : () -> f64
  %elapsed = arith.subf %end, %start : f64

  func.call @printF64(%elapsed) : (f64) -> ()

  return
}