#Sparse3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

func.func private @rtclock() -> f64
func.func private @printNewline()
func.func private @printF64(f64)
func.func private @printMemrefF64(%ptr : tensor<*xf64>)

llvm.mlir.global internal constant @fileA("playground/tensor_A_test.tns\00")
llvm.mlir.global internal constant @fileB("playground/tensor_B_test.tns\00")
llvm.mlir.global internal constant @fileB("playground/tensor_C_test.tns\00")


// Computes:
//
//   F(i,k,j) = sum_l B(i,k,l) * C(l,j)
//   E(i,k,j) = F(i,k,j) * D(k,j)
//   A(i,j)   = sum_k E(i,k,j)
//
// Therefore:
//
//   A(i,j) = sum_{k,l} B(i,k,l) * C(l,j) * D(k,j)
//
func.func @spttspm(
    %B: tensor<?x?x?xf64, #Sparse3D>, // B(i,k,l)
    %C: tensor<?x?xf64, #CSC>,        // C(l,j)
    %D: tensor<?x?xf64, #CSC>         // D(k,j)
) -> tensor<?x?xf64> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64

  %dim_i = tensor.dim %B, %c0 : tensor<?x?x?xf64, #Sparse3D>
  %dim_k = tensor.dim %B, %c1 : tensor<?x?x?xf64, #Sparse3D>
  %dim_j = tensor.dim %C, %c1 : tensor<?x?xf64, #CSC>

  %F_empty = tensor.empty(%dim_i, %dim_k, %dim_j) : tensor<?x?x?xf64>
  %F_init = linalg.fill ins(%zero : f64) outs(%F_empty : tensor<?x?x?xf64>) -> tensor<?x?x?xf64>

  %F_result = linalg.generic {
    indexing_maps = [
      affine_map<(i, k, j, l) -> (i, k, l)>,
      affine_map<(i, k, j, l) -> (l, j)>,
      affine_map<(i, k, j, l) -> (i, k, j)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  }
  ins(%B, %C : tensor<?x?x?xf64, #Sparse3D>, tensor<?x?xf64, #CSC>)
  outs(%F_init : tensor<?x?x?xf64>) {
  ^bb0(%b: f64, %c: f64, %f_in: f64):
    %product = arith.mulf %b, %c : f64
    %f_out = arith.addf %f_in, %product : f64
    linalg.yield %f_out : f64
  } -> tensor<?x?x?xf64>

  
  %E_empty = tensor.empty(%dim_i, %dim_k, %dim_j) : tensor<?x?x?xf64>

  %E_init = linalg.fill ins(%zero : f64) outs(%E_empty : tensor<?x?x?xf64>) -> tensor<?x?x?xf64>

  %E_result = linalg.generic {
    indexing_maps = [
      affine_map<(i, k, j) -> (i, k, j)>,
      affine_map<(i, k, j) -> (k, j)>,
      affine_map<(i, k, j) -> (i, k, j)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  }
  ins(%F_result, %D : tensor<?x?x?xf64>, tensor<?x?xf64, #CSC>)
  outs(%E_init : tensor<?x?x?xf64>) {
  ^bb0(%f: f64, %d: f64, %e_in: f64):
    %e_out = arith.mulf %f, %d : f64
    linalg.yield %e_out : f64
  } -> tensor<?x?x?xf64>

  %A_empty = tensor.empty(%dim_i, %dim_j) : tensor<?x?xf64>

  %A_init = linalg.fill ins(%zero : f64) outs(%A_empty : tensor<?x?xf64>) -> tensor<?x?xf64>

  %A_result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j, k) -> (i, k, j)>,
      affine_map<(i, j, k) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  }
  ins(%E_result : tensor<?x?x?xf64>)
  outs(%A_init : tensor<?x?xf64>) {
  ^bb0(%e: f64, %a_in: f64):
    %a_out = arith.addf %a_in, %e : f64
    linalg.yield %a_out : f64
  } -> tensor<?x?xf64>

  return %A_result : tensor<?x?xf64>
}

func.func @main(%tensor_ptr: !llvm.ptr, %matrix_c_ptr: !llvm.ptr, %matrix_d_ptr: !llvm.ptr) {
  %B = sparse_tensor.new %tensor_ptr : !llvm.ptr to tensor<?x?x?xf64, #Sparse3D>
  %C = sparse_tensor.new %matrix_c_ptr : !llvm.ptr to tensor<?x?xf64, #CSC>
  %D = sparse_tensor.new %matrix_d_ptr : !llvm.ptr to tensor<?x?xf64, #CSC>

  %warmup = func.call @spttspm(%B, %C, %D) : (tensor<?x?x?xf64, #Sparse3D>, tensor<?x?xf64, #CSC>, tensor<?x?xf64, #CSC>) -> tensor<?x?xf64>
  bufferization.dealloc_tensor %warmup : tensor<?x?xf64>

  %start_comp = func.call @rtclock() : () -> f64
  %A = func.call @spttspm(%B, %C, %D) :(tensor<?x?x?xf64, #Sparse3D>,tensor<?x?xf64, #CSC>,tensor<?x?xf64, #CSC>) -> tensor<?x?xf64>
  %end_comp = func.call @rtclock() : () -> f64

  %elapsed = arith.subf %end_comp, %start_comp : f64
  func.call @printF64(%elapsed) : (f64) -> ()
  func.call @printNewline() : () -> ()

  bufferization.dealloc_tensor %B : tensor<?x?x?xf64, #Sparse3D>
  bufferization.dealloc_tensor %C : tensor<?x?xf64, #CSC>
  bufferization.dealloc_tensor %D : tensor<?x?xf64, #CSC>
  bufferization.dealloc_tensor %A : tensor<?x?xf64>

  return
}