module {
  func.func private @delSparseTensor(!llvm.ptr)
  func.func private @delSparseTensorReader(!llvm.ptr)
  func.func private @getSparseTensorReaderDimSizes(!llvm.ptr) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @createCheckedSparseTensorReader(!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseLvlSize(!llvm.ptr, index) -> index
  func.func private @printMemrefF64(tensor<*xf64>)
  llvm.mlir.global internal constant @fileA("playground/tensor_A_test.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fileB("playground/tensor_B_test.tns\00") {addr_space = 0 : i32}
  func.func @spttspm(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.ptr {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %0 = call @sparseLvlSize(%arg0, %c0_0) : (!llvm.ptr, index) -> index
    %c1_1 = arith.constant 1 : index
    %1 = call @sparseLvlSize(%arg0, %c1_1) : (!llvm.ptr, index) -> index
    %c0_2 = arith.constant 0 : index
    %2 = call @sparseLvlSize(%arg1, %c0_2) : (!llvm.ptr, index) -> index
    %c65536_i64 = arith.constant 65536 : i64
    %c65536_i64_3 = arith.constant 65536 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c3 = arith.constant 3 : index
    %alloca = memref.alloca(%c3) : memref<?xi64>
    %c0_4 = arith.constant 0 : index
    memref.store %c65536_i64, %alloca[%c0_4] : memref<?xi64>
    %c1_5 = arith.constant 1 : index
    memref.store %c65536_i64_3, %alloca[%c1_5] : memref<?xi64>
    %c2 = arith.constant 2 : index
    memref.store %c262144_i64, %alloca[%c2] : memref<?xi64>
    %c3_6 = arith.constant 3 : index
    %alloca_7 = memref.alloca(%c3_6) : memref<?xindex>
    %c0_8 = arith.constant 0 : index
    memref.store %0, %alloca_7[%c0_8] : memref<?xindex>
    %c1_9 = arith.constant 1 : index
    memref.store %1, %alloca_7[%c1_9] : memref<?xindex>
    %c2_10 = arith.constant 2 : index
    memref.store %2, %alloca_7[%c2_10] : memref<?xindex>
    %c0_11 = arith.constant 0 : index
    %c1_12 = arith.constant 1 : index
    %c2_13 = arith.constant 2 : index
    %c3_14 = arith.constant 3 : index
    %alloca_15 = memref.alloca(%c3_14) : memref<?xindex>
    %c0_16 = arith.constant 0 : index
    memref.store %c0_11, %alloca_15[%c0_16] : memref<?xindex>
    %c1_17 = arith.constant 1 : index
    memref.store %c1_12, %alloca_15[%c1_17] : memref<?xindex>
    %c2_18 = arith.constant 2 : index
    memref.store %c2_13, %alloca_15[%c2_18] : memref<?xindex>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_19 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32_20 = arith.constant 0 : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = call @newSparseTensor(%alloca_7, %alloca_7, %alloca, %alloca_15, %alloca_15, %c0_i32, %c0_i32_19, %c1_i32, %c0_i32_20, %3) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %5 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %6 = call @sparseValuesF64(%arg1) : (!llvm.ptr) -> memref<?xf64>
    %c0_21 = arith.constant 0 : index
    %7 = call @sparseLvlSize(%arg0, %c0_21) : (!llvm.ptr, index) -> index
    %c1_22 = arith.constant 1 : index
    %8 = call @sparseLvlSize(%arg0, %c1_22) : (!llvm.ptr, index) -> index
    %c2_23 = arith.constant 2 : index
    %9 = call @sparsePositions0(%arg0, %c2_23) : (!llvm.ptr, index) -> memref<?xindex>
    %c2_24 = arith.constant 2 : index
    %10 = call @sparseCoordinates0(%arg0, %c2_24) : (!llvm.ptr, index) -> memref<?xindex>
    %c0_25 = arith.constant 0 : index
    %11 = call @sparseLvlSize(%arg1, %c0_25) : (!llvm.ptr, index) -> index
    %c1_26 = arith.constant 1 : index
    %12 = call @sparsePositions0(%arg1, %c1_26) : (!llvm.ptr, index) -> memref<?xindex>
    %c1_27 = arith.constant 1 : index
    %13 = call @sparseCoordinates0(%arg1, %c1_27) : (!llvm.ptr, index) -> memref<?xindex>
    %c3_28 = arith.constant 3 : index
    %alloca_29 = memref.alloca(%c3_28) : memref<?xindex>
    %alloca_30 = memref.alloca() : memref<f64>
    %14 = scf.for %arg2 = %c0 to %7 step %c1 iter_args(%arg3 = %4) -> (!llvm.ptr) {
      %15 = arith.muli %arg2, %8 : index
      %16 = scf.for %arg4 = %c0 to %8 step %c1 iter_args(%arg5 = %arg3) -> (!llvm.ptr) {
        %17 = arith.addi %arg4, %15 : index
        %18 = scf.for %arg6 = %c0 to %11 step %c1 iter_args(%arg7 = %arg5) -> (!llvm.ptr) {
          %19 = memref.load %9[%17] : memref<?xindex>
          %20 = arith.addi %17, %c1 : index
          %21 = memref.load %9[%20] : memref<?xindex>
          %22 = memref.load %12[%arg6] : memref<?xindex>
          %23 = arith.addi %arg6, %c1 : index
          %24 = memref.load %12[%23] : memref<?xindex>
          %25:5 = scf.while (%arg8 = %19, %arg9 = %22, %arg10 = %cst, %arg11 = %false, %arg12 = %arg7) : (index, index, f64, i1, !llvm.ptr) -> (index, index, f64, i1, !llvm.ptr) {
            %27 = arith.cmpi ult, %arg8, %21 : index
            %28 = arith.cmpi ult, %arg9, %24 : index
            %29 = arith.andi %27, %28 : i1
            scf.condition(%29) %arg8, %arg9, %arg10, %arg11, %arg12 : index, index, f64, i1, !llvm.ptr
          } do {
          ^bb0(%arg8: index, %arg9: index, %arg10: f64, %arg11: i1, %arg12: !llvm.ptr):
            %27 = memref.load %10[%arg8] : memref<?xindex>
            %28 = memref.load %13[%arg9] : memref<?xindex>
            %29 = arith.cmpi ult, %28, %27 : index
            %30 = arith.select %29, %28, %27 : index
            %31 = arith.cmpi eq, %27, %30 : index
            %32 = arith.cmpi eq, %28, %30 : index
            %33 = arith.andi %31, %32 : i1
            %34:3 = scf.if %33 -> (f64, i1, !llvm.ptr) {
              %41 = memref.load %5[%arg8] : memref<?xf64>
              %42 = memref.load %6[%arg9] : memref<?xf64>
              %43 = arith.mulf %41, %42 : f64
              %44 = arith.addf %arg10, %43 : f64
              scf.yield %44, %true, %arg12 : f64, i1, !llvm.ptr
            } else {
              scf.yield %arg10, %arg11, %arg12 : f64, i1, !llvm.ptr
            }
            %35 = arith.cmpi eq, %27, %30 : index
            %36 = arith.addi %arg8, %c1 : index
            %37 = arith.select %35, %36, %arg8 : index
            %38 = arith.cmpi eq, %28, %30 : index
            %39 = arith.addi %arg9, %c1 : index
            %40 = arith.select %38, %39, %arg9 : index
            scf.yield %37, %40, %34#0, %34#1, %34#2 : index, index, f64, i1, !llvm.ptr
          }
          %26 = scf.if %25#3 -> (!llvm.ptr) {
            %c0_31 = arith.constant 0 : index
            memref.store %arg2, %alloca_29[%c0_31] : memref<?xindex>
            %c1_32 = arith.constant 1 : index
            memref.store %arg4, %alloca_29[%c1_32] : memref<?xindex>
            %c2_33 = arith.constant 2 : index
            memref.store %arg6, %alloca_29[%c2_33] : memref<?xindex>
            memref.store %25#2, %alloca_30[] : memref<f64>
            func.call @lexInsertF64(%25#4, %alloca_29, %alloca_30) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
            scf.yield %25#4 : !llvm.ptr
          } else {
            scf.yield %25#4 : !llvm.ptr
          }
          scf.yield %26 : !llvm.ptr
        } {"Emitted from" = "linalg.generic"}
        scf.yield %18 : !llvm.ptr
      } {"Emitted from" = "linalg.generic"}
      scf.yield %16 : !llvm.ptr
    } {"Emitted from" = "linalg.generic"}
    call @endLexInsert(%14) : (!llvm.ptr) -> ()
    return %14 : !llvm.ptr
  }
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @fileA : !llvm.ptr
    %1 = llvm.mlir.addressof @fileB : !llvm.ptr
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %alloca = memref.alloca(%c3) : memref<?xindex>
    %c0_3 = arith.constant 0 : index
    memref.store %c0_0, %alloca[%c0_3] : memref<?xindex>
    %c1_4 = arith.constant 1 : index
    memref.store %c0_1, %alloca[%c1_4] : memref<?xindex>
    %c2_5 = arith.constant 2 : index
    memref.store %c0_2, %alloca[%c2_5] : memref<?xindex>
    %c1_i32 = arith.constant 1 : i32
    %2 = call @createCheckedSparseTensorReader(%0, %alloca, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %3 = call @getSparseTensorReaderDimSizes(%2) : (!llvm.ptr) -> memref<?xindex>
    %c0_6 = arith.constant 0 : index
    %4 = memref.load %3[%c0_6] : memref<?xindex>
    %c1_7 = arith.constant 1 : index
    %5 = memref.load %3[%c1_7] : memref<?xindex>
    %c2_8 = arith.constant 2 : index
    %6 = memref.load %3[%c2_8] : memref<?xindex>
    %c65536_i64 = arith.constant 65536 : i64
    %c65536_i64_9 = arith.constant 65536 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c3_10 = arith.constant 3 : index
    %alloca_11 = memref.alloca(%c3_10) : memref<?xi64>
    %c0_12 = arith.constant 0 : index
    memref.store %c65536_i64, %alloca_11[%c0_12] : memref<?xi64>
    %c1_13 = arith.constant 1 : index
    memref.store %c65536_i64_9, %alloca_11[%c1_13] : memref<?xi64>
    %c2_14 = arith.constant 2 : index
    memref.store %c262144_i64, %alloca_11[%c2_14] : memref<?xi64>
    %c0_15 = arith.constant 0 : index
    %c1_16 = arith.constant 1 : index
    %c2_17 = arith.constant 2 : index
    %c3_18 = arith.constant 3 : index
    %alloca_19 = memref.alloca(%c3_18) : memref<?xindex>
    %c0_20 = arith.constant 0 : index
    memref.store %c0_15, %alloca_19[%c0_20] : memref<?xindex>
    %c1_21 = arith.constant 1 : index
    memref.store %c1_16, %alloca_19[%c1_21] : memref<?xindex>
    %c2_22 = arith.constant 2 : index
    memref.store %c2_17, %alloca_19[%c2_22] : memref<?xindex>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_23 = arith.constant 0 : i32
    %c1_i32_24 = arith.constant 1 : i32
    %c1_i32_25 = arith.constant 1 : i32
    %7 = call @newSparseTensor(%3, %3, %alloca_11, %alloca_19, %alloca_19, %c0_i32, %c0_i32_23, %c1_i32_24, %c1_i32_25, %2) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%2) : (!llvm.ptr) -> ()
    %c0_26 = arith.constant 0 : index
    %c0_27 = arith.constant 0 : index
    %c2_28 = arith.constant 2 : index
    %alloca_29 = memref.alloca(%c2_28) : memref<?xindex>
    %c0_30 = arith.constant 0 : index
    memref.store %c0_26, %alloca_29[%c0_30] : memref<?xindex>
    %c1_31 = arith.constant 1 : index
    memref.store %c0_27, %alloca_29[%c1_31] : memref<?xindex>
    %c1_i32_32 = arith.constant 1 : i32
    %8 = call @createCheckedSparseTensorReader(%1, %alloca_29, %c1_i32_32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %9 = call @getSparseTensorReaderDimSizes(%8) : (!llvm.ptr) -> memref<?xindex>
    %c0_33 = arith.constant 0 : index
    %10 = memref.load %9[%c0_33] : memref<?xindex>
    %c1_34 = arith.constant 1 : index
    %11 = memref.load %9[%c1_34] : memref<?xindex>
    %c65536_i64_35 = arith.constant 65536 : i64
    %c262144_i64_36 = arith.constant 262144 : i64
    %c2_37 = arith.constant 2 : index
    %alloca_38 = memref.alloca(%c2_37) : memref<?xi64>
    %c0_39 = arith.constant 0 : index
    memref.store %c65536_i64_35, %alloca_38[%c0_39] : memref<?xi64>
    %c1_40 = arith.constant 1 : index
    memref.store %c262144_i64_36, %alloca_38[%c1_40] : memref<?xi64>
    %c1_41 = arith.constant 1 : index
    %c0_42 = arith.constant 0 : index
    %c1_43 = arith.constant 1 : index
    %c0_44 = arith.constant 0 : index
    %c2_45 = arith.constant 2 : index
    %alloca_46 = memref.alloca(%c2_45) : memref<?xindex>
    %c0_47 = arith.constant 0 : index
    memref.store %c1_41, %alloca_46[%c0_47] : memref<?xindex>
    %c1_48 = arith.constant 1 : index
    memref.store %c0_42, %alloca_46[%c1_48] : memref<?xindex>
    %c2_49 = arith.constant 2 : index
    %alloca_50 = memref.alloca(%c2_49) : memref<?xindex>
    %c0_51 = arith.constant 0 : index
    memref.store %c1_43, %alloca_50[%c0_51] : memref<?xindex>
    %c1_52 = arith.constant 1 : index
    memref.store %c0_44, %alloca_50[%c1_52] : memref<?xindex>
    %c2_53 = arith.constant 2 : index
    %alloca_54 = memref.alloca(%c2_53) : memref<?xindex>
    %c0_55 = arith.constant 0 : index
    memref.store %11, %alloca_54[%c0_55] : memref<?xindex>
    %c1_56 = arith.constant 1 : index
    memref.store %10, %alloca_54[%c1_56] : memref<?xindex>
    %c0_i32_57 = arith.constant 0 : i32
    %c0_i32_58 = arith.constant 0 : i32
    %c1_i32_59 = arith.constant 1 : i32
    %c1_i32_60 = arith.constant 1 : i32
    %12 = call @newSparseTensor(%9, %alloca_54, %alloca_38, %alloca_46, %alloca_50, %c0_i32_57, %c0_i32_58, %c1_i32_59, %c1_i32_60, %8) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%8) : (!llvm.ptr) -> ()
    %13 = call @spttspm(%7, %12) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %c0_61 = arith.constant 0 : index
    %14 = call @sparseLvlSize(%13, %c0_61) : (!llvm.ptr, index) -> index
    %c1_62 = arith.constant 1 : index
    %15 = call @sparseLvlSize(%13, %c1_62) : (!llvm.ptr, index) -> index
    %c2_63 = arith.constant 2 : index
    %16 = call @sparseLvlSize(%13, %c2_63) : (!llvm.ptr, index) -> index
    %17 = bufferization.alloc_tensor(%14, %15, %16) : tensor<?x?x?xf64>
    %18 = linalg.fill ins(%cst : f64) outs(%17 : tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    %19 = call @sparseValuesF64(%13) : (!llvm.ptr) -> memref<?xf64>
    %c0_64 = arith.constant 0 : index
    %20 = call @sparseLvlSize(%13, %c0_64) : (!llvm.ptr, index) -> index
    %c1_65 = arith.constant 1 : index
    %21 = call @sparseLvlSize(%13, %c1_65) : (!llvm.ptr, index) -> index
    %c2_66 = arith.constant 2 : index
    %22 = call @sparsePositions0(%13, %c2_66) : (!llvm.ptr, index) -> memref<?xindex>
    %c2_67 = arith.constant 2 : index
    %23 = call @sparseCoordinates0(%13, %c2_67) : (!llvm.ptr, index) -> memref<?xindex>
    %24 = scf.for %arg0 = %c0 to %20 step %c1 iter_args(%arg1 = %18) -> (tensor<?x?x?xf64>) {
      %25 = arith.muli %arg0, %21 : index
      %26 = scf.for %arg2 = %c0 to %21 step %c1 iter_args(%arg3 = %arg1) -> (tensor<?x?x?xf64>) {
        %27 = arith.addi %arg2, %25 : index
        %28 = memref.load %22[%27] : memref<?xindex>
        %29 = arith.addi %27, %c1 : index
        %30 = memref.load %22[%29] : memref<?xindex>
        %31 = scf.for %arg4 = %28 to %30 step %c1 iter_args(%arg5 = %arg3) -> (tensor<?x?x?xf64>) {
          %32 = memref.load %23[%arg4] : memref<?xindex>
          %33 = memref.load %19[%arg4] : memref<?xf64>
          %inserted = tensor.insert %33 into %arg5[%arg0, %arg2, %32] : tensor<?x?x?xf64>
          scf.yield %inserted : tensor<?x?x?xf64>
        } {"Emitted from" = "sparse_tensor.foreach"}
        scf.yield %31 : tensor<?x?x?xf64>
      } {"Emitted from" = "sparse_tensor.foreach"}
      scf.yield %26 : tensor<?x?x?xf64>
    } {"Emitted from" = "sparse_tensor.foreach"}
    %cast = tensor.cast %24 : tensor<?x?x?xf64> to tensor<*xf64>
    call @printMemrefF64(%cast) : (tensor<*xf64>) -> ()
    call @delSparseTensor(%7) : (!llvm.ptr) -> ()
    call @delSparseTensor(%12) : (!llvm.ptr) -> ()
    call @delSparseTensor(%13) : (!llvm.ptr) -> ()
    bufferization.dealloc_tensor %24 : tensor<?x?x?xf64>
    return
  }
}

