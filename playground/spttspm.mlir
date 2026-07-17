#Sparse3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : dense, d2 : compressed)
}>

// 2D Sparse Encoding (CSC): stored column-major
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

func.func private @printMemrefF64(%ptr : tensor<*xf64>)

llvm.mlir.global internal constant @fileA("playground/tensor_A_test.tns\00")
llvm.mlir.global internal constant @fileB("playground/tensor_B_test.tns\00")

func.func @spttspm(
    %argA: tensor<?x?x?xf64, #Sparse3D>,   // A(i,j)
    %argB: tensor<?x?xf64, #CSC>    // B(j,k)
) -> tensor<?x?x?xf64, #Sparse3D> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %dim_i = tensor.dim %argA, %c0 : tensor<?x?x?xf64, #Sparse3D>
  %dim_j = tensor.dim %argA, %c1 : tensor<?x?x?xf64, #Sparse3D>
  %dim_l = tensor.dim %argB, %c1 : tensor<?x?xf64, #CSC>

  %initC = tensor.empty(%dim_i, %dim_j, %dim_l) : tensor<?x?x?xf64, #Sparse3D>

  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j, l, k) -> (i, j, k)>,   // A(i,j)
      affine_map<(i, j, l, k) -> (k, l)>,   // B(j,k)
      affine_map<(i, j, l, k) -> (i, j, l)>    // C(i,k)  out
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%argA, %argB : tensor<?x?x?xf64, #Sparse3D>, tensor<?x?xf64, #CSC>)
    outs(%initC : tensor<?x?x?xf64, #Sparse3D>) {
  ^bb0(%a: f64, %b: f64, %c_in: f64):
    %prod  = arith.mulf %a, %b   : f64
    %c_out = arith.addf %c_in, %prod : f64
    linalg.yield %c_out : f64
  } -> tensor<?x?x?xf64, #Sparse3D>

  return %result : tensor<?x?x?xf64, #Sparse3D>
}

func.func @main() {
  %file_a = llvm.mlir.addressof @fileA : !llvm.ptr
  %file_b = llvm.mlir.addressof @fileB : !llvm.ptr

  %A = sparse_tensor.new %file_a : !llvm.ptr to tensor<?x?x?xf64, #Sparse3D>
  %B = sparse_tensor.new %file_b : !llvm.ptr to tensor<?x?xf64, #CSC>

  %C = func.call @spttspm(%A, %B)
    : (tensor<?x?x?xf64, #Sparse3D>, tensor<?x?xf64, #CSC>) -> tensor<?x?x?xf64, #Sparse3D>

  %Cdense = sparse_tensor.convert %C : tensor<?x?x?xf64, #Sparse3D> to tensor<?x?x?xf64>
  %Cprint = tensor.cast %Cdense : tensor<?x?x?xf64> to tensor<*xf64>
  func.call @printMemrefF64(%Cprint) : (tensor<*xf64>) -> ()

  bufferization.dealloc_tensor %A : tensor<?x?x?xf64, #Sparse3D>
  bufferization.dealloc_tensor %B : tensor<?x?xf64, #CSC>
  bufferization.dealloc_tensor %C : tensor<?x?x?xf64, #Sparse3D>
  bufferization.dealloc_tensor %Cdense : tensor<?x?x?xf64>

  return
}
