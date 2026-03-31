#sparse = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : compressed) }>
#sparse1 = #sparse_tensor.encoding<{ map = (d0, d1, d2, d3) -> (d0 : dense, d1 : compressed, d2 : compressed, d3 : compressed) }>
#sparse2 = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : compressed, d1 : singleton, d2 : singleton) }>
#sparse3 = #sparse_tensor.encoding<{ map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : singleton, d2 : singleton, d3 : singleton) }>
module {
  func.func private @delSparseTensorReader(!llvm.ptr)
  func.func private @getSparseTensorReaderReadToBuffers0F64(!llvm.ptr, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>) -> i1 attributes {llvm.emit_c_interface}
  func.func private @getSparseTensorReaderNSE(!llvm.ptr) -> index
  func.func private @getSparseTensorReaderDimSizes(!llvm.ptr) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @createCheckedSparseTensorReader(!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @printF64(f64)
  llvm.mlir.global internal constant @tensorL("tensor_L.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @tensorBra("tensor_Bra.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @tensorW("tensor_W.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @tensorKet("tensor_Ket.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @tensorR("tensor_R.tns\00") {addr_space = 0 : i32}
  func.func @dmrg_expectation_value(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xf64>, %arg5: !sparse_tensor.storage_specifier<#sparse>, %arg6: memref<?xindex>, %arg7: memref<?xindex>, %arg8: memref<?xindex>, %arg9: memref<?xindex>, %arg10: memref<?xf64>, %arg11: !sparse_tensor.storage_specifier<#sparse>, %arg12: memref<?xindex>, %arg13: memref<?xindex>, %arg14: memref<?xindex>, %arg15: memref<?xindex>, %arg16: memref<?xindex>, %arg17: memref<?xindex>, %arg18: memref<?xf64>, %arg19: !sparse_tensor.storage_specifier<#sparse1>, %arg20: memref<?xindex>, %arg21: memref<?xindex>, %arg22: memref<?xindex>, %arg23: memref<?xindex>, %arg24: memref<?xf64>, %arg25: !sparse_tensor.storage_specifier<#sparse>, %arg26: memref<?xindex>, %arg27: memref<?xindex>, %arg28: memref<?xindex>, %arg29: memref<?xindex>, %arg30: memref<?xf64>, %arg31: !sparse_tensor.storage_specifier<#sparse>) -> memref<f64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    linalg.fill ins(%cst : f64) outs(%alloc : memref<f64>)
    %0 = sparse_tensor.storage_specifier.get %arg5  val_mem_sz : !sparse_tensor.storage_specifier<#sparse>
    %subview = memref.subview %arg4[0] [%0] [1] : memref<?xf64> to memref<?xf64>
    %1 = sparse_tensor.storage_specifier.get %arg11  val_mem_sz : !sparse_tensor.storage_specifier<#sparse>
    %subview_0 = memref.subview %arg10[0] [%1] [1] : memref<?xf64> to memref<?xf64>
    %2 = sparse_tensor.storage_specifier.get %arg19  val_mem_sz : !sparse_tensor.storage_specifier<#sparse1>
    %subview_1 = memref.subview %arg18[0] [%2] [1] : memref<?xf64> to memref<?xf64>
    %3 = sparse_tensor.storage_specifier.get %arg25  val_mem_sz : !sparse_tensor.storage_specifier<#sparse>
    %subview_2 = memref.subview %arg24[0] [%3] [1] : memref<?xf64> to memref<?xf64>
    %4 = sparse_tensor.storage_specifier.get %arg31  val_mem_sz : !sparse_tensor.storage_specifier<#sparse>
    %subview_3 = memref.subview %arg30[0] [%4] [1] : memref<?xf64> to memref<?xf64>
    %5 = sparse_tensor.storage_specifier.get %arg5  lvl_sz at 0 : !sparse_tensor.storage_specifier<#sparse>
    %6 = sparse_tensor.storage_specifier.get %arg5  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %subview_4 = memref.subview %arg0[0] [%6] [1] : memref<?xindex> to memref<?xindex>
    %7 = sparse_tensor.storage_specifier.get %arg5  crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %subview_5 = memref.subview %arg1[0] [%7] [1] : memref<?xindex> to memref<?xindex>
    %8 = sparse_tensor.storage_specifier.get %arg5  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %subview_6 = memref.subview %arg2[0] [%8] [1] : memref<?xindex> to memref<?xindex>
    %9 = sparse_tensor.storage_specifier.get %arg5  crd_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %subview_7 = memref.subview %arg3[0] [%9] [1] : memref<?xindex> to memref<?xindex>
    %10 = sparse_tensor.storage_specifier.get %arg11  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %subview_8 = memref.subview %arg6[0] [%10] [1] : memref<?xindex> to memref<?xindex>
    %11 = sparse_tensor.storage_specifier.get %arg11  crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %subview_9 = memref.subview %arg7[0] [%11] [1] : memref<?xindex> to memref<?xindex>
    %12 = sparse_tensor.storage_specifier.get %arg11  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %subview_10 = memref.subview %arg8[0] [%12] [1] : memref<?xindex> to memref<?xindex>
    %13 = sparse_tensor.storage_specifier.get %arg11  crd_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %subview_11 = memref.subview %arg9[0] [%13] [1] : memref<?xindex> to memref<?xindex>
    %14 = sparse_tensor.storage_specifier.get %arg19  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse1>
    %subview_12 = memref.subview %arg12[0] [%14] [1] : memref<?xindex> to memref<?xindex>
    %15 = sparse_tensor.storage_specifier.get %arg19  crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse1>
    %subview_13 = memref.subview %arg13[0] [%15] [1] : memref<?xindex> to memref<?xindex>
    %16 = sparse_tensor.storage_specifier.get %arg19  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse1>
    %subview_14 = memref.subview %arg14[0] [%16] [1] : memref<?xindex> to memref<?xindex>
    %17 = sparse_tensor.storage_specifier.get %arg19  crd_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse1>
    %subview_15 = memref.subview %arg15[0] [%17] [1] : memref<?xindex> to memref<?xindex>
    %18 = sparse_tensor.storage_specifier.get %arg19  pos_mem_sz at 3 : !sparse_tensor.storage_specifier<#sparse1>
    %subview_16 = memref.subview %arg16[0] [%18] [1] : memref<?xindex> to memref<?xindex>
    %19 = sparse_tensor.storage_specifier.get %arg19  crd_mem_sz at 3 : !sparse_tensor.storage_specifier<#sparse1>
    %subview_17 = memref.subview %arg17[0] [%19] [1] : memref<?xindex> to memref<?xindex>
    %20 = sparse_tensor.storage_specifier.get %arg25  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %subview_18 = memref.subview %arg20[0] [%20] [1] : memref<?xindex> to memref<?xindex>
    %21 = sparse_tensor.storage_specifier.get %arg25  crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %subview_19 = memref.subview %arg21[0] [%21] [1] : memref<?xindex> to memref<?xindex>
    %22 = sparse_tensor.storage_specifier.get %arg25  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %subview_20 = memref.subview %arg22[0] [%22] [1] : memref<?xindex> to memref<?xindex>
    %23 = sparse_tensor.storage_specifier.get %arg25  crd_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %subview_21 = memref.subview %arg23[0] [%23] [1] : memref<?xindex> to memref<?xindex>
    %24 = sparse_tensor.storage_specifier.get %arg31  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %subview_22 = memref.subview %arg26[0] [%24] [1] : memref<?xindex> to memref<?xindex>
    %25 = sparse_tensor.storage_specifier.get %arg31  crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %subview_23 = memref.subview %arg27[0] [%25] [1] : memref<?xindex> to memref<?xindex>
    %26 = sparse_tensor.storage_specifier.get %arg31  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %subview_24 = memref.subview %arg28[0] [%26] [1] : memref<?xindex> to memref<?xindex>
    %27 = sparse_tensor.storage_specifier.get %arg31  crd_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %subview_25 = memref.subview %arg29[0] [%27] [1] : memref<?xindex> to memref<?xindex>
    %28 = memref.load %alloc[] : memref<f64>
    %29 = scf.for %arg32 = %c0 to %5 step %c1 iter_args(%arg33 = %28) -> (f64) {
      %30 = memref.load %subview_4[%arg32] : memref<?xindex>
      %31 = arith.addi %arg32, %c1 : index
      %32 = memref.load %subview_4[%31] : memref<?xindex>
      %33 = arith.addi %arg32, %c1 : index
      %34 = scf.for %arg34 = %30 to %32 step %c1 iter_args(%arg35 = %arg33) -> (f64) {
        %35 = memref.load %subview_5[%arg34] : memref<?xindex>
        %36 = memref.load %subview_6[%arg34] : memref<?xindex>
        %37 = arith.addi %arg34, %c1 : index
        %38 = memref.load %subview_6[%37] : memref<?xindex>
        %39 = arith.addi %35, %c1 : index
        %40 = scf.for %arg36 = %36 to %38 step %c1 iter_args(%arg37 = %arg35) -> (f64) {
          %41 = memref.load %subview_7[%arg36] : memref<?xindex>
          %42 = memref.load %subview[%arg36] : memref<?xf64>
          %43 = memref.load %subview_8[%arg32] : memref<?xindex>
          %44 = memref.load %subview_8[%33] : memref<?xindex>
          %45 = scf.for %arg38 = %43 to %44 step %c1 iter_args(%arg39 = %arg37) -> (f64) {
            %46 = memref.load %subview_9[%arg38] : memref<?xindex>
            %47 = memref.load %subview_12[%35] : memref<?xindex>
            %48 = memref.load %subview_12[%39] : memref<?xindex>
            %49 = memref.load %subview_22[%46] : memref<?xindex>
            %50 = arith.addi %46, %c1 : index
            %51 = memref.load %subview_22[%50] : memref<?xindex>
            %52:3 = scf.while (%arg40 = %47, %arg41 = %49, %arg42 = %arg39) : (index, index, f64) -> (index, index, f64) {
              %53 = arith.cmpi ult, %arg40, %48 : index
              %54 = arith.cmpi ult, %arg41, %51 : index
              %55 = arith.andi %53, %54 : i1
              scf.condition(%55) %arg40, %arg41, %arg42 : index, index, f64
            } do {
            ^bb0(%arg40: index, %arg41: index, %arg42: f64):
              %53 = memref.load %subview_13[%arg40] : memref<?xindex>
              %54 = memref.load %subview_23[%arg41] : memref<?xindex>
              %55 = arith.cmpi ult, %54, %53 : index
              %56 = arith.select %55, %54, %53 : index
              %57 = arith.cmpi eq, %53, %56 : index
              %58 = arith.cmpi eq, %54, %56 : index
              %59 = arith.andi %57, %58 : i1
              %60 = scf.if %59 -> (f64) {
                %67 = memref.load %subview_18[%41] : memref<?xindex>
                %68 = arith.addi %41, %c1 : index
                %69 = memref.load %subview_18[%68] : memref<?xindex>
                %70 = memref.load %subview_24[%arg41] : memref<?xindex>
                %71 = arith.addi %arg41, %c1 : index
                %72 = memref.load %subview_24[%71] : memref<?xindex>
                %73:3 = scf.while (%arg43 = %67, %arg44 = %70, %arg45 = %arg42) : (index, index, f64) -> (index, index, f64) {
                  %74 = arith.cmpi ult, %arg43, %69 : index
                  %75 = arith.cmpi ult, %arg44, %72 : index
                  %76 = arith.andi %74, %75 : i1
                  scf.condition(%76) %arg43, %arg44, %arg45 : index, index, f64
                } do {
                ^bb0(%arg43: index, %arg44: index, %arg45: f64):
                  %74 = memref.load %subview_19[%arg43] : memref<?xindex>
                  %75 = memref.load %subview_25[%arg44] : memref<?xindex>
                  %76 = arith.cmpi ult, %75, %74 : index
                  %77 = arith.select %76, %75, %74 : index
                  %78 = arith.cmpi eq, %74, %77 : index
                  %79 = arith.cmpi eq, %75, %77 : index
                  %80 = arith.andi %78, %79 : i1
                  %81 = scf.if %80 -> (f64) {
                    %88 = memref.load %subview_3[%arg44] : memref<?xf64>
                    %89 = memref.load %subview_10[%arg38] : memref<?xindex>
                    %90 = arith.addi %arg38, %c1 : index
                    %91 = memref.load %subview_10[%90] : memref<?xindex>
                    %92 = memref.load %subview_14[%arg40] : memref<?xindex>
                    %93 = arith.addi %arg40, %c1 : index
                    %94 = memref.load %subview_14[%93] : memref<?xindex>
                    %95:3 = scf.while (%arg46 = %89, %arg47 = %92, %arg48 = %arg45) : (index, index, f64) -> (index, index, f64) {
                      %96 = arith.cmpi ult, %arg46, %91 : index
                      %97 = arith.cmpi ult, %arg47, %94 : index
                      %98 = arith.andi %96, %97 : i1
                      scf.condition(%98) %arg46, %arg47, %arg48 : index, index, f64
                    } do {
                    ^bb0(%arg46: index, %arg47: index, %arg48: f64):
                      %96 = memref.load %subview_11[%arg46] : memref<?xindex>
                      %97 = memref.load %subview_15[%arg47] : memref<?xindex>
                      %98 = arith.cmpi ult, %97, %96 : index
                      %99 = arith.select %98, %97, %96 : index
                      %100 = arith.cmpi eq, %96, %99 : index
                      %101 = arith.cmpi eq, %97, %99 : index
                      %102 = arith.andi %100, %101 : i1
                      %103 = scf.if %102 -> (f64) {
                        %110 = memref.load %subview_0[%arg46] : memref<?xf64>
                        %111 = memref.load %subview_16[%arg47] : memref<?xindex>
                        %112 = arith.addi %arg47, %c1 : index
                        %113 = memref.load %subview_16[%112] : memref<?xindex>
                        %114 = memref.load %subview_20[%arg43] : memref<?xindex>
                        %115 = arith.addi %arg43, %c1 : index
                        %116 = memref.load %subview_20[%115] : memref<?xindex>
                        %117:3 = scf.while (%arg49 = %111, %arg50 = %114, %arg51 = %arg48) : (index, index, f64) -> (index, index, f64) {
                          %118 = arith.cmpi ult, %arg49, %113 : index
                          %119 = arith.cmpi ult, %arg50, %116 : index
                          %120 = arith.andi %118, %119 : i1
                          scf.condition(%120) %arg49, %arg50, %arg51 : index, index, f64
                        } do {
                        ^bb0(%arg49: index, %arg50: index, %arg51: f64):
                          %118 = memref.load %subview_17[%arg49] : memref<?xindex>
                          %119 = memref.load %subview_21[%arg50] : memref<?xindex>
                          %120 = arith.cmpi ult, %119, %118 : index
                          %121 = arith.select %120, %119, %118 : index
                          %122 = arith.cmpi eq, %118, %121 : index
                          %123 = arith.cmpi eq, %119, %121 : index
                          %124 = arith.andi %122, %123 : i1
                          %125 = scf.if %124 -> (f64) {
                            %132 = arith.mulf %42, %110 : f64
                            %133 = memref.load %subview_1[%arg49] : memref<?xf64>
                            %134 = arith.mulf %132, %133 : f64
                            %135 = memref.load %subview_2[%arg50] : memref<?xf64>
                            %136 = arith.mulf %134, %135 : f64
                            %137 = arith.mulf %136, %88 : f64
                            %138 = arith.addf %arg51, %137 : f64
                            scf.yield %138 : f64
                          } else {
                            scf.yield %arg51 : f64
                          }
                          %126 = arith.cmpi eq, %118, %121 : index
                          %127 = arith.addi %arg49, %c1 : index
                          %128 = arith.select %126, %127, %arg49 : index
                          %129 = arith.cmpi eq, %119, %121 : index
                          %130 = arith.addi %arg50, %c1 : index
                          %131 = arith.select %129, %130, %arg50 : index
                          scf.yield %128, %131, %125 : index, index, f64
                        } attributes {"Emitted from" = "linalg.generic"}
                        scf.yield %117#2 : f64
                      } else {
                        scf.yield %arg48 : f64
                      }
                      %104 = arith.cmpi eq, %96, %99 : index
                      %105 = arith.addi %arg46, %c1 : index
                      %106 = arith.select %104, %105, %arg46 : index
                      %107 = arith.cmpi eq, %97, %99 : index
                      %108 = arith.addi %arg47, %c1 : index
                      %109 = arith.select %107, %108, %arg47 : index
                      scf.yield %106, %109, %103 : index, index, f64
                    } attributes {"Emitted from" = "linalg.generic"}
                    scf.yield %95#2 : f64
                  } else {
                    scf.yield %arg45 : f64
                  }
                  %82 = arith.cmpi eq, %74, %77 : index
                  %83 = arith.addi %arg43, %c1 : index
                  %84 = arith.select %82, %83, %arg43 : index
                  %85 = arith.cmpi eq, %75, %77 : index
                  %86 = arith.addi %arg44, %c1 : index
                  %87 = arith.select %85, %86, %arg44 : index
                  scf.yield %84, %87, %81 : index, index, f64
                } attributes {"Emitted from" = "linalg.generic"}
                scf.yield %73#2 : f64
              } else {
                scf.yield %arg42 : f64
              }
              %61 = arith.cmpi eq, %53, %56 : index
              %62 = arith.addi %arg40, %c1 : index
              %63 = arith.select %61, %62, %arg40 : index
              %64 = arith.cmpi eq, %54, %56 : index
              %65 = arith.addi %arg41, %c1 : index
              %66 = arith.select %64, %65, %arg41 : index
              scf.yield %63, %66, %60 : index, index, f64
            } attributes {"Emitted from" = "linalg.generic"}
            scf.yield %52#2 : f64
          } {"Emitted from" = "linalg.generic"}
          scf.yield %45 : f64
        } {"Emitted from" = "linalg.generic"}
        scf.yield %40 : f64
      } {"Emitted from" = "linalg.generic"}
      scf.yield %34 : f64
    } {"Emitted from" = "linalg.generic"}
    memref.store %29, %alloc[] : memref<f64>
    return %alloc : memref<f64>
  }
  func.func private @"_insert_dense_compressed_compressed_-9223372036854775808_-9223372036854775808_-9223372036854775808_f64_0_0"(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xf64>, %arg5: !sparse_tensor.storage_specifier<#sparse>, %arg6: index, %arg7: index, %arg8: index, %arg9: f64) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
    %c2 = arith.constant 2 : index
    %false = arith.constant false
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.addi %arg6, %c1 : index
    %1 = memref.load %arg0[%arg6] : memref<?xindex>
    %2 = memref.load %arg0[%0] : memref<?xindex>
    %3 = sparse_tensor.storage_specifier.get %arg5  crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %4 = arith.subi %2, %c1 : index
    %5 = arith.cmpi ult, %1, %2 : index
    %6 = scf.if %5 -> (i1) {
      %21 = memref.load %arg1[%4] : memref<?xindex>
      %22 = arith.cmpi eq, %21, %arg7 : index
      scf.yield %22 : i1
    } else {
      memref.store %3, %arg0[%arg6] : memref<?xindex>
      scf.yield %false : i1
    }
    %7:7 = scf.if %6 -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index) {
      scf.yield %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %4 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index
    } else {
      %21 = arith.addi %3, %c1 : index
      memref.store %21, %arg0[%0] : memref<?xindex>
      %22 = sparse_tensor.storage_specifier.get %arg5  crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
      %dim_0 = memref.dim %arg1, %c0 : memref<?xindex>
      %23 = arith.addi %22, %c1 : index
      %24 = arith.cmpi ugt, %23, %dim_0 : index
      %25 = scf.if %24 -> (memref<?xindex>) {
        %32 = arith.muli %dim_0, %c2 : index
        %33 = memref.realloc %arg1(%32) : memref<?xindex> to memref<?xindex>
        scf.yield %33 : memref<?xindex>
      } else {
        scf.yield %arg1 : memref<?xindex>
      }
      memref.store %arg7, %25[%22] : memref<?xindex>
      %26 = sparse_tensor.storage_specifier.set %arg5  crd_mem_sz at 1 with %23 : !sparse_tensor.storage_specifier<#sparse>
      %27 = sparse_tensor.storage_specifier.get %26  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
      %dim_1 = memref.dim %arg2, %c0 : memref<?xindex>
      %28 = arith.addi %27, %c1 : index
      %29 = arith.cmpi ugt, %28, %dim_1 : index
      %30 = scf.if %29 -> (memref<?xindex>) {
        %32 = arith.muli %dim_1, %c2 : index
        %33 = memref.realloc %arg2(%32) : memref<?xindex> to memref<?xindex>
        scf.yield %33 : memref<?xindex>
      } else {
        scf.yield %arg2 : memref<?xindex>
      }
      memref.store %c0, %30[%27] : memref<?xindex>
      %31 = sparse_tensor.storage_specifier.set %26  pos_mem_sz at 2 with %28 : !sparse_tensor.storage_specifier<#sparse>
      scf.yield %arg0, %25, %30, %arg3, %arg4, %31, %3 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index
    }
    %8 = arith.addi %7#6, %c1 : index
    %9 = memref.load %7#2[%7#6] : memref<?xindex>
    %10 = memref.load %7#2[%8] : memref<?xindex>
    %11 = sparse_tensor.storage_specifier.get %7#5  crd_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %12 = arith.subi %10, %c1 : index
    %13 = arith.cmpi ult, %9, %10 : index
    %14 = scf.if %13 -> (i1) {
      %21 = memref.load %7#3[%12] : memref<?xindex>
      %22 = arith.cmpi eq, %21, %arg8 : index
      scf.yield %22 : i1
    } else {
      memref.store %11, %7#2[%7#6] : memref<?xindex>
      scf.yield %false : i1
    }
    %15:7 = scf.if %14 -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index) {
      scf.yield %7#0, %7#1, %7#2, %7#3, %7#4, %7#5, %12 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index
    } else {
      %21 = arith.addi %11, %c1 : index
      memref.store %21, %7#2[%8] : memref<?xindex>
      %22 = sparse_tensor.storage_specifier.get %7#5  crd_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
      %dim_0 = memref.dim %7#3, %c0 : memref<?xindex>
      %23 = arith.addi %22, %c1 : index
      %24 = arith.cmpi ugt, %23, %dim_0 : index
      %25 = scf.if %24 -> (memref<?xindex>) {
        %27 = arith.muli %dim_0, %c2 : index
        %28 = memref.realloc %7#3(%27) : memref<?xindex> to memref<?xindex>
        scf.yield %28 : memref<?xindex>
      } else {
        scf.yield %7#3 : memref<?xindex>
      }
      memref.store %arg8, %25[%22] : memref<?xindex>
      %26 = sparse_tensor.storage_specifier.set %7#5  crd_mem_sz at 2 with %23 : !sparse_tensor.storage_specifier<#sparse>
      scf.yield %7#0, %7#1, %7#2, %25, %7#4, %26, %11 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index
    }
    %16 = sparse_tensor.storage_specifier.get %15#5  val_mem_sz : !sparse_tensor.storage_specifier<#sparse>
    %dim = memref.dim %15#4, %c0 : memref<?xf64>
    %17 = arith.addi %16, %c1 : index
    %18 = arith.cmpi ugt, %17, %dim : index
    %19 = scf.if %18 -> (memref<?xf64>) {
      %21 = arith.muli %dim, %c2 : index
      %22 = memref.realloc %15#4(%21) : memref<?xf64> to memref<?xf64>
      scf.yield %22 : memref<?xf64>
    } else {
      scf.yield %15#4 : memref<?xf64>
    }
    memref.store %arg9, %19[%16] : memref<?xf64>
    %20 = sparse_tensor.storage_specifier.set %15#5  val_mem_sz with %17 : !sparse_tensor.storage_specifier<#sparse>
    return %15#0, %15#1, %15#2, %15#3, %19, %20 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
  }
  func.func private @"_insert_dense_compressed_compressed_compressed_-9223372036854775808_-9223372036854775808_-9223372036854775808_-9223372036854775808_f64_0_0"(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: memref<?xindex>, %arg6: memref<?xf64>, %arg7: !sparse_tensor.storage_specifier<#sparse1>, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: f64) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>) {
    %c2 = arith.constant 2 : index
    %false = arith.constant false
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.addi %arg8, %c1 : index
    %1 = memref.load %arg0[%arg8] : memref<?xindex>
    %2 = memref.load %arg0[%0] : memref<?xindex>
    %3 = sparse_tensor.storage_specifier.get %arg7  crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse1>
    %4 = arith.subi %2, %c1 : index
    %5 = arith.cmpi ult, %1, %2 : index
    %6 = scf.if %5 -> (i1) {
      %29 = memref.load %arg1[%4] : memref<?xindex>
      %30 = arith.cmpi eq, %29, %arg9 : index
      scf.yield %30 : i1
    } else {
      memref.store %3, %arg0[%arg8] : memref<?xindex>
      scf.yield %false : i1
    }
    %7:9 = scf.if %6 -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index) {
      scf.yield %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %4 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index
    } else {
      %29 = arith.addi %3, %c1 : index
      memref.store %29, %arg0[%0] : memref<?xindex>
      %30 = sparse_tensor.storage_specifier.get %arg7  crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse1>
      %dim_0 = memref.dim %arg1, %c0 : memref<?xindex>
      %31 = arith.addi %30, %c1 : index
      %32 = arith.cmpi ugt, %31, %dim_0 : index
      %33 = scf.if %32 -> (memref<?xindex>) {
        %40 = arith.muli %dim_0, %c2 : index
        %41 = memref.realloc %arg1(%40) : memref<?xindex> to memref<?xindex>
        scf.yield %41 : memref<?xindex>
      } else {
        scf.yield %arg1 : memref<?xindex>
      }
      memref.store %arg9, %33[%30] : memref<?xindex>
      %34 = sparse_tensor.storage_specifier.set %arg7  crd_mem_sz at 1 with %31 : !sparse_tensor.storage_specifier<#sparse1>
      %35 = sparse_tensor.storage_specifier.get %34  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse1>
      %dim_1 = memref.dim %arg2, %c0 : memref<?xindex>
      %36 = arith.addi %35, %c1 : index
      %37 = arith.cmpi ugt, %36, %dim_1 : index
      %38 = scf.if %37 -> (memref<?xindex>) {
        %40 = arith.muli %dim_1, %c2 : index
        %41 = memref.realloc %arg2(%40) : memref<?xindex> to memref<?xindex>
        scf.yield %41 : memref<?xindex>
      } else {
        scf.yield %arg2 : memref<?xindex>
      }
      memref.store %c0, %38[%35] : memref<?xindex>
      %39 = sparse_tensor.storage_specifier.set %34  pos_mem_sz at 2 with %36 : !sparse_tensor.storage_specifier<#sparse1>
      scf.yield %arg0, %33, %38, %arg3, %arg4, %arg5, %arg6, %39, %3 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index
    }
    %8 = arith.addi %7#8, %c1 : index
    %9 = memref.load %7#2[%7#8] : memref<?xindex>
    %10 = memref.load %7#2[%8] : memref<?xindex>
    %11 = sparse_tensor.storage_specifier.get %7#7  crd_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse1>
    %12 = arith.subi %10, %c1 : index
    %13 = arith.cmpi ult, %9, %10 : index
    %14 = scf.if %13 -> (i1) {
      %29 = memref.load %7#3[%12] : memref<?xindex>
      %30 = arith.cmpi eq, %29, %arg10 : index
      scf.yield %30 : i1
    } else {
      memref.store %11, %7#2[%7#8] : memref<?xindex>
      scf.yield %false : i1
    }
    %15:9 = scf.if %14 -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index) {
      scf.yield %7#0, %7#1, %7#2, %7#3, %7#4, %7#5, %7#6, %7#7, %12 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index
    } else {
      %29 = arith.addi %11, %c1 : index
      memref.store %29, %7#2[%8] : memref<?xindex>
      %30 = sparse_tensor.storage_specifier.get %7#7  crd_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse1>
      %dim_0 = memref.dim %7#3, %c0 : memref<?xindex>
      %31 = arith.addi %30, %c1 : index
      %32 = arith.cmpi ugt, %31, %dim_0 : index
      %33 = scf.if %32 -> (memref<?xindex>) {
        %40 = arith.muli %dim_0, %c2 : index
        %41 = memref.realloc %7#3(%40) : memref<?xindex> to memref<?xindex>
        scf.yield %41 : memref<?xindex>
      } else {
        scf.yield %7#3 : memref<?xindex>
      }
      memref.store %arg10, %33[%30] : memref<?xindex>
      %34 = sparse_tensor.storage_specifier.set %7#7  crd_mem_sz at 2 with %31 : !sparse_tensor.storage_specifier<#sparse1>
      %35 = sparse_tensor.storage_specifier.get %34  pos_mem_sz at 3 : !sparse_tensor.storage_specifier<#sparse1>
      %dim_1 = memref.dim %7#4, %c0 : memref<?xindex>
      %36 = arith.addi %35, %c1 : index
      %37 = arith.cmpi ugt, %36, %dim_1 : index
      %38 = scf.if %37 -> (memref<?xindex>) {
        %40 = arith.muli %dim_1, %c2 : index
        %41 = memref.realloc %7#4(%40) : memref<?xindex> to memref<?xindex>
        scf.yield %41 : memref<?xindex>
      } else {
        scf.yield %7#4 : memref<?xindex>
      }
      memref.store %c0, %38[%35] : memref<?xindex>
      %39 = sparse_tensor.storage_specifier.set %34  pos_mem_sz at 3 with %36 : !sparse_tensor.storage_specifier<#sparse1>
      scf.yield %7#0, %7#1, %7#2, %33, %38, %7#5, %7#6, %39, %11 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index
    }
    %16 = arith.addi %15#8, %c1 : index
    %17 = memref.load %15#4[%15#8] : memref<?xindex>
    %18 = memref.load %15#4[%16] : memref<?xindex>
    %19 = sparse_tensor.storage_specifier.get %15#7  crd_mem_sz at 3 : !sparse_tensor.storage_specifier<#sparse1>
    %20 = arith.subi %18, %c1 : index
    %21 = arith.cmpi ult, %17, %18 : index
    %22 = scf.if %21 -> (i1) {
      %29 = memref.load %15#5[%20] : memref<?xindex>
      %30 = arith.cmpi eq, %29, %arg11 : index
      scf.yield %30 : i1
    } else {
      memref.store %19, %15#4[%15#8] : memref<?xindex>
      scf.yield %false : i1
    }
    %23:9 = scf.if %22 -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index) {
      scf.yield %15#0, %15#1, %15#2, %15#3, %15#4, %15#5, %15#6, %15#7, %20 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index
    } else {
      %29 = arith.addi %19, %c1 : index
      memref.store %29, %15#4[%16] : memref<?xindex>
      %30 = sparse_tensor.storage_specifier.get %15#7  crd_mem_sz at 3 : !sparse_tensor.storage_specifier<#sparse1>
      %dim_0 = memref.dim %15#5, %c0 : memref<?xindex>
      %31 = arith.addi %30, %c1 : index
      %32 = arith.cmpi ugt, %31, %dim_0 : index
      %33 = scf.if %32 -> (memref<?xindex>) {
        %35 = arith.muli %dim_0, %c2 : index
        %36 = memref.realloc %15#5(%35) : memref<?xindex> to memref<?xindex>
        scf.yield %36 : memref<?xindex>
      } else {
        scf.yield %15#5 : memref<?xindex>
      }
      memref.store %arg11, %33[%30] : memref<?xindex>
      %34 = sparse_tensor.storage_specifier.set %15#7  crd_mem_sz at 3 with %31 : !sparse_tensor.storage_specifier<#sparse1>
      scf.yield %15#0, %15#1, %15#2, %15#3, %15#4, %33, %15#6, %34, %19 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index
    }
    %24 = sparse_tensor.storage_specifier.get %23#7  val_mem_sz : !sparse_tensor.storage_specifier<#sparse1>
    %dim = memref.dim %23#6, %c0 : memref<?xf64>
    %25 = arith.addi %24, %c1 : index
    %26 = arith.cmpi ugt, %25, %dim : index
    %27 = scf.if %26 -> (memref<?xf64>) {
      %29 = arith.muli %dim, %c2 : index
      %30 = memref.realloc %23#6(%29) : memref<?xf64> to memref<?xf64>
      scf.yield %30 : memref<?xf64>
    } else {
      scf.yield %23#6 : memref<?xf64>
    }
    memref.store %arg12, %27[%24] : memref<?xf64>
    %28 = sparse_tensor.storage_specifier.set %23#7  val_mem_sz with %25 : !sparse_tensor.storage_specifier<#sparse1>
    return %23#0, %23#1, %23#2, %23#3, %23#4, %23#5, %27, %28 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>
  }
  func.func private @_sparse_binary_search_0_1_2_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>) -> index {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %0:2 = scf.while (%arg4 = %arg0, %arg5 = %arg1) : (index, index) -> (index, index) {
      %1 = arith.cmpi ult, %arg4, %arg5 : index
      scf.condition(%1) %arg4, %arg5 : index, index
    } do {
    ^bb0(%arg4: index, %arg5: index):
      %1 = arith.addi %arg4, %arg5 : index
      %2 = arith.shrui %1, %c1 : index
      %3 = arith.addi %2, %c1 : index
      %4 = arith.muli %arg1, %c3 : index
      %5 = arith.muli %2, %c3 : index
      %6 = memref.load %arg2[%4] : memref<?xindex>
      %7 = memref.load %arg2[%5] : memref<?xindex>
      %8 = arith.cmpi ne, %6, %7 : index
      %9 = scf.if %8 -> (i1) {
        %12 = arith.cmpi ult, %6, %7 : index
        scf.yield %12 : i1
      } else {
        %12 = arith.addi %4, %c1 : index
        %13 = arith.addi %5, %c1 : index
        %14 = memref.load %arg2[%12] : memref<?xindex>
        %15 = memref.load %arg2[%13] : memref<?xindex>
        %16 = arith.cmpi ne, %14, %15 : index
        %17 = scf.if %16 -> (i1) {
          %18 = arith.cmpi ult, %14, %15 : index
          scf.yield %18 : i1
        } else {
          %18 = arith.addi %4, %c2 : index
          %19 = arith.addi %5, %c2 : index
          %20 = memref.load %arg2[%18] : memref<?xindex>
          %21 = memref.load %arg2[%19] : memref<?xindex>
          %22 = arith.cmpi ult, %20, %21 : index
          scf.yield %22 : i1
        }
        scf.yield %17 : i1
      }
      %10 = arith.select %9, %arg4, %3 : index
      %11 = arith.select %9, %2, %arg5 : index
      scf.yield %10, %11 : index, index
    }
    return %0#0 : index
  }
  func.func private @_sparse_sort_stable_0_1_2_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg0, %c1 : index
    scf.for %arg4 = %0 to %arg1 step %c1 {
      %1 = func.call @_sparse_binary_search_0_1_2_index_coo_0_f64(%arg0, %arg4, %arg2, %arg3) : (index, index, memref<?xindex>, memref<?xf64>) -> index
      %2 = arith.muli %arg4, %c3 : index
      %3 = memref.load %arg2[%2] : memref<?xindex>
      %4 = arith.addi %2, %c1 : index
      %5 = memref.load %arg2[%4] : memref<?xindex>
      %6 = arith.addi %2, %c2 : index
      %7 = memref.load %arg2[%6] : memref<?xindex>
      %8 = memref.load %arg3[%arg4] : memref<?xf64>
      %9 = arith.subi %arg4, %1 : index
      scf.for %arg5 = %c0 to %9 step %c1 {
        %13 = arith.subi %arg4, %arg5 : index
        %14 = arith.subi %13, %c1 : index
        %15 = arith.muli %14, %c3 : index
        %16 = arith.muli %13, %c3 : index
        %17 = memref.load %arg2[%15] : memref<?xindex>
        memref.store %17, %arg2[%16] : memref<?xindex>
        %18 = arith.addi %15, %c1 : index
        %19 = arith.addi %16, %c1 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        memref.store %20, %arg2[%19] : memref<?xindex>
        %21 = arith.addi %15, %c2 : index
        %22 = arith.addi %16, %c2 : index
        %23 = memref.load %arg2[%21] : memref<?xindex>
        memref.store %23, %arg2[%22] : memref<?xindex>
        %24 = memref.load %arg3[%14] : memref<?xf64>
        memref.store %24, %arg3[%13] : memref<?xf64>
      }
      %10 = arith.muli %1, %c3 : index
      memref.store %3, %arg2[%10] : memref<?xindex>
      %11 = arith.addi %10, %c1 : index
      memref.store %5, %arg2[%11] : memref<?xindex>
      %12 = arith.addi %10, %c2 : index
      memref.store %7, %arg2[%12] : memref<?xindex>
      memref.store %8, %arg3[%1] : memref<?xf64>
    }
    return
  }
  func.func private @_sparse_shift_down_0_1_2_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>, %arg4: index) {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.cmpi uge, %arg4, %c2 : index
    scf.if %0 {
      %1 = arith.subi %arg1, %arg0 : index
      %2 = arith.subi %arg4, %c2 : index
      %3 = arith.shrui %2, %c1 : index
      %4 = arith.cmpi uge, %3, %1 : index
      scf.if %4 {
        %5 = arith.shli %1, %c1 : index
        %6 = arith.addi %5, %c1 : index
        %7 = arith.addi %6, %arg0 : index
        %8 = arith.addi %6, %c1 : index
        %9 = arith.cmpi ult, %8, %arg4 : index
        %10:2 = scf.if %9 -> (index, index) {
          %12 = arith.addi %8, %arg0 : index
          %13 = arith.muli %7, %c3 : index
          %14 = arith.muli %12, %c3 : index
          %15 = memref.load %arg2[%13] : memref<?xindex>
          %16 = memref.load %arg2[%14] : memref<?xindex>
          %17 = arith.cmpi ne, %15, %16 : index
          %18 = scf.if %17 -> (i1) {
            %20 = arith.cmpi ult, %15, %16 : index
            scf.yield %20 : i1
          } else {
            %20 = arith.addi %13, %c1 : index
            %21 = arith.addi %14, %c1 : index
            %22 = memref.load %arg2[%20] : memref<?xindex>
            %23 = memref.load %arg2[%21] : memref<?xindex>
            %24 = arith.cmpi ne, %22, %23 : index
            %25 = scf.if %24 -> (i1) {
              %26 = arith.cmpi ult, %22, %23 : index
              scf.yield %26 : i1
            } else {
              %26 = arith.addi %13, %c2 : index
              %27 = arith.addi %14, %c2 : index
              %28 = memref.load %arg2[%26] : memref<?xindex>
              %29 = memref.load %arg2[%27] : memref<?xindex>
              %30 = arith.cmpi ult, %28, %29 : index
              scf.yield %30 : i1
            }
            scf.yield %25 : i1
          }
          %19:2 = scf.if %18 -> (index, index) {
            scf.yield %8, %12 : index, index
          } else {
            scf.yield %6, %7 : index, index
          }
          scf.yield %19#0, %19#1 : index, index
        } else {
          scf.yield %6, %7 : index, index
        }
        %11:3 = scf.while (%arg5 = %arg1, %arg6 = %10#0, %arg7 = %10#1) : (index, index, index) -> (index, index, index) {
          %12 = arith.muli %arg5, %c3 : index
          %13 = arith.muli %arg7, %c3 : index
          %14 = memref.load %arg2[%12] : memref<?xindex>
          %15 = memref.load %arg2[%13] : memref<?xindex>
          %16 = arith.cmpi ne, %14, %15 : index
          %17 = scf.if %16 -> (i1) {
            %18 = arith.cmpi ult, %14, %15 : index
            scf.yield %18 : i1
          } else {
            %18 = arith.addi %12, %c1 : index
            %19 = arith.addi %13, %c1 : index
            %20 = memref.load %arg2[%18] : memref<?xindex>
            %21 = memref.load %arg2[%19] : memref<?xindex>
            %22 = arith.cmpi ne, %20, %21 : index
            %23 = scf.if %22 -> (i1) {
              %24 = arith.cmpi ult, %20, %21 : index
              scf.yield %24 : i1
            } else {
              %24 = arith.addi %12, %c2 : index
              %25 = arith.addi %13, %c2 : index
              %26 = memref.load %arg2[%24] : memref<?xindex>
              %27 = memref.load %arg2[%25] : memref<?xindex>
              %28 = arith.cmpi ult, %26, %27 : index
              scf.yield %28 : i1
            }
            scf.yield %23 : i1
          }
          scf.condition(%17) %arg5, %arg6, %arg7 : index, index, index
        } do {
        ^bb0(%arg5: index, %arg6: index, %arg7: index):
          %12 = arith.muli %arg5, %c3 : index
          %13 = arith.muli %arg7, %c3 : index
          %14 = memref.load %arg2[%12] : memref<?xindex>
          %15 = memref.load %arg2[%13] : memref<?xindex>
          memref.store %15, %arg2[%12] : memref<?xindex>
          memref.store %14, %arg2[%13] : memref<?xindex>
          %16 = arith.addi %12, %c1 : index
          %17 = arith.addi %13, %c1 : index
          %18 = memref.load %arg2[%16] : memref<?xindex>
          %19 = memref.load %arg2[%17] : memref<?xindex>
          memref.store %19, %arg2[%16] : memref<?xindex>
          memref.store %18, %arg2[%17] : memref<?xindex>
          %20 = arith.addi %12, %c2 : index
          %21 = arith.addi %13, %c2 : index
          %22 = memref.load %arg2[%20] : memref<?xindex>
          %23 = memref.load %arg2[%21] : memref<?xindex>
          memref.store %23, %arg2[%20] : memref<?xindex>
          memref.store %22, %arg2[%21] : memref<?xindex>
          %24 = memref.load %arg3[%arg5] : memref<?xf64>
          %25 = memref.load %arg3[%arg7] : memref<?xf64>
          memref.store %25, %arg3[%arg5] : memref<?xf64>
          memref.store %24, %arg3[%arg7] : memref<?xf64>
          %26 = arith.cmpi uge, %3, %arg6 : index
          %27:2 = scf.if %26 -> (index, index) {
            %28 = arith.shli %arg6, %c1 : index
            %29 = arith.addi %28, %c1 : index
            %30 = arith.addi %29, %arg0 : index
            %31 = arith.addi %29, %c1 : index
            %32 = arith.cmpi ult, %31, %arg4 : index
            %33:2 = scf.if %32 -> (index, index) {
              %34 = arith.addi %31, %arg0 : index
              %35 = arith.muli %30, %c3 : index
              %36 = arith.muli %34, %c3 : index
              %37 = memref.load %arg2[%35] : memref<?xindex>
              %38 = memref.load %arg2[%36] : memref<?xindex>
              %39 = arith.cmpi ne, %37, %38 : index
              %40 = scf.if %39 -> (i1) {
                %42 = arith.cmpi ult, %37, %38 : index
                scf.yield %42 : i1
              } else {
                %42 = arith.addi %35, %c1 : index
                %43 = arith.addi %36, %c1 : index
                %44 = memref.load %arg2[%42] : memref<?xindex>
                %45 = memref.load %arg2[%43] : memref<?xindex>
                %46 = arith.cmpi ne, %44, %45 : index
                %47 = scf.if %46 -> (i1) {
                  %48 = arith.cmpi ult, %44, %45 : index
                  scf.yield %48 : i1
                } else {
                  %48 = arith.addi %35, %c2 : index
                  %49 = arith.addi %36, %c2 : index
                  %50 = memref.load %arg2[%48] : memref<?xindex>
                  %51 = memref.load %arg2[%49] : memref<?xindex>
                  %52 = arith.cmpi ult, %50, %51 : index
                  scf.yield %52 : i1
                }
                scf.yield %47 : i1
              }
              %41:2 = scf.if %40 -> (index, index) {
                scf.yield %31, %34 : index, index
              } else {
                scf.yield %29, %30 : index, index
              }
              scf.yield %41#0, %41#1 : index, index
            } else {
              scf.yield %29, %30 : index, index
            }
            scf.yield %33#0, %33#1 : index, index
          } else {
            scf.yield %arg6, %arg7 : index, index
          }
          scf.yield %arg7, %27#0, %27#1 : index, index, index
        }
      }
    }
    return
  }
  func.func private @_sparse_heap_sort_0_1_2_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>) {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.subi %arg1, %arg0 : index
    %1 = arith.subi %0, %c2 : index
    %2 = arith.shrui %1, %c1 : index
    %3 = arith.addi %2, %c1 : index
    scf.for %arg4 = %c0 to %3 step %c1 {
      %5 = arith.subi %2, %arg4 : index
      %6 = arith.addi %arg0, %5 : index
      func.call @_sparse_shift_down_0_1_2_index_coo_0_f64(%arg0, %6, %arg2, %arg3, %0) : (index, index, memref<?xindex>, memref<?xf64>, index) -> ()
    }
    %4 = arith.subi %0, %c1 : index
    scf.for %arg4 = %c0 to %4 step %c1 {
      %5 = arith.subi %0, %arg4 : index
      %6 = arith.addi %arg0, %5 : index
      %7 = arith.subi %6, %c1 : index
      %8 = arith.muli %arg0, %c3 : index
      %9 = arith.muli %7, %c3 : index
      %10 = memref.load %arg2[%8] : memref<?xindex>
      %11 = memref.load %arg2[%9] : memref<?xindex>
      memref.store %11, %arg2[%8] : memref<?xindex>
      memref.store %10, %arg2[%9] : memref<?xindex>
      %12 = arith.addi %8, %c1 : index
      %13 = arith.addi %9, %c1 : index
      %14 = memref.load %arg2[%12] : memref<?xindex>
      %15 = memref.load %arg2[%13] : memref<?xindex>
      memref.store %15, %arg2[%12] : memref<?xindex>
      memref.store %14, %arg2[%13] : memref<?xindex>
      %16 = arith.addi %8, %c2 : index
      %17 = arith.addi %9, %c2 : index
      %18 = memref.load %arg2[%16] : memref<?xindex>
      %19 = memref.load %arg2[%17] : memref<?xindex>
      memref.store %19, %arg2[%16] : memref<?xindex>
      memref.store %18, %arg2[%17] : memref<?xindex>
      %20 = memref.load %arg3[%arg0] : memref<?xf64>
      %21 = memref.load %arg3[%7] : memref<?xf64>
      memref.store %21, %arg3[%arg0] : memref<?xf64>
      memref.store %20, %arg3[%7] : memref<?xf64>
      %22 = arith.subi %5, %c1 : index
      func.call @_sparse_shift_down_0_1_2_index_coo_0_f64(%arg0, %arg0, %arg2, %arg3, %22) : (index, index, memref<?xindex>, memref<?xf64>, index) -> ()
    }
    return
  }
  func.func private @_sparse_partition_0_1_2_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>) -> index {
    %c-1 = arith.constant -1 : index
    %false = arith.constant false
    %true = arith.constant true
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg0, %arg1 : index
    %1 = arith.shrui %0, %c1 : index
    %2 = arith.subi %arg1, %c1 : index
    %3 = arith.subi %arg1, %arg0 : index
    %4 = arith.cmpi ult, %3, %c1000 : index
    scf.if %4 {
      %6 = arith.muli %1, %c3 : index
      %7 = arith.muli %arg0, %c3 : index
      %8 = memref.load %arg2[%6] : memref<?xindex>
      %9 = memref.load %arg2[%7] : memref<?xindex>
      %10 = arith.cmpi ne, %8, %9 : index
      %11 = scf.if %10 -> (i1) {
        %18 = arith.cmpi ult, %8, %9 : index
        scf.yield %18 : i1
      } else {
        %18 = arith.addi %6, %c1 : index
        %19 = arith.addi %7, %c1 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        %21 = memref.load %arg2[%19] : memref<?xindex>
        %22 = arith.cmpi ne, %20, %21 : index
        %23 = scf.if %22 -> (i1) {
          %24 = arith.cmpi ult, %20, %21 : index
          scf.yield %24 : i1
        } else {
          %24 = arith.addi %6, %c2 : index
          %25 = arith.addi %7, %c2 : index
          %26 = memref.load %arg2[%24] : memref<?xindex>
          %27 = memref.load %arg2[%25] : memref<?xindex>
          %28 = arith.cmpi ult, %26, %27 : index
          scf.yield %28 : i1
        }
        scf.yield %23 : i1
      }
      scf.if %11 {
        %18 = arith.muli %1, %c3 : index
        %19 = arith.muli %arg0, %c3 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        %21 = memref.load %arg2[%19] : memref<?xindex>
        memref.store %21, %arg2[%18] : memref<?xindex>
        memref.store %20, %arg2[%19] : memref<?xindex>
        %22 = arith.addi %18, %c1 : index
        %23 = arith.addi %19, %c1 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        memref.store %25, %arg2[%22] : memref<?xindex>
        memref.store %24, %arg2[%23] : memref<?xindex>
        %26 = arith.addi %18, %c2 : index
        %27 = arith.addi %19, %c2 : index
        %28 = memref.load %arg2[%26] : memref<?xindex>
        %29 = memref.load %arg2[%27] : memref<?xindex>
        memref.store %29, %arg2[%26] : memref<?xindex>
        memref.store %28, %arg2[%27] : memref<?xindex>
        %30 = memref.load %arg3[%1] : memref<?xf64>
        %31 = memref.load %arg3[%arg0] : memref<?xf64>
        memref.store %31, %arg3[%1] : memref<?xf64>
        memref.store %30, %arg3[%arg0] : memref<?xf64>
      }
      %12 = arith.muli %2, %c3 : index
      %13 = arith.muli %1, %c3 : index
      %14 = memref.load %arg2[%12] : memref<?xindex>
      %15 = memref.load %arg2[%13] : memref<?xindex>
      %16 = arith.cmpi ne, %14, %15 : index
      %17 = scf.if %16 -> (i1) {
        %18 = arith.cmpi ult, %14, %15 : index
        scf.yield %18 : i1
      } else {
        %18 = arith.addi %12, %c1 : index
        %19 = arith.addi %13, %c1 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        %21 = memref.load %arg2[%19] : memref<?xindex>
        %22 = arith.cmpi ne, %20, %21 : index
        %23 = scf.if %22 -> (i1) {
          %24 = arith.cmpi ult, %20, %21 : index
          scf.yield %24 : i1
        } else {
          %24 = arith.addi %12, %c2 : index
          %25 = arith.addi %13, %c2 : index
          %26 = memref.load %arg2[%24] : memref<?xindex>
          %27 = memref.load %arg2[%25] : memref<?xindex>
          %28 = arith.cmpi ult, %26, %27 : index
          scf.yield %28 : i1
        }
        scf.yield %23 : i1
      }
      scf.if %17 {
        %18 = arith.muli %2, %c3 : index
        %19 = arith.muli %1, %c3 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        %21 = memref.load %arg2[%19] : memref<?xindex>
        memref.store %21, %arg2[%18] : memref<?xindex>
        memref.store %20, %arg2[%19] : memref<?xindex>
        %22 = arith.addi %18, %c1 : index
        %23 = arith.addi %19, %c1 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        memref.store %25, %arg2[%22] : memref<?xindex>
        memref.store %24, %arg2[%23] : memref<?xindex>
        %26 = arith.addi %18, %c2 : index
        %27 = arith.addi %19, %c2 : index
        %28 = memref.load %arg2[%26] : memref<?xindex>
        %29 = memref.load %arg2[%27] : memref<?xindex>
        memref.store %29, %arg2[%26] : memref<?xindex>
        memref.store %28, %arg2[%27] : memref<?xindex>
        %30 = memref.load %arg3[%2] : memref<?xf64>
        %31 = memref.load %arg3[%1] : memref<?xf64>
        memref.store %31, %arg3[%2] : memref<?xf64>
        memref.store %30, %arg3[%1] : memref<?xf64>
        %32 = arith.muli %1, %c3 : index
        %33 = arith.muli %arg0, %c3 : index
        %34 = memref.load %arg2[%32] : memref<?xindex>
        %35 = memref.load %arg2[%33] : memref<?xindex>
        %36 = arith.cmpi ne, %34, %35 : index
        %37 = scf.if %36 -> (i1) {
          %38 = arith.cmpi ult, %34, %35 : index
          scf.yield %38 : i1
        } else {
          %38 = arith.addi %32, %c1 : index
          %39 = arith.addi %33, %c1 : index
          %40 = memref.load %arg2[%38] : memref<?xindex>
          %41 = memref.load %arg2[%39] : memref<?xindex>
          %42 = arith.cmpi ne, %40, %41 : index
          %43 = scf.if %42 -> (i1) {
            %44 = arith.cmpi ult, %40, %41 : index
            scf.yield %44 : i1
          } else {
            %44 = arith.addi %32, %c2 : index
            %45 = arith.addi %33, %c2 : index
            %46 = memref.load %arg2[%44] : memref<?xindex>
            %47 = memref.load %arg2[%45] : memref<?xindex>
            %48 = arith.cmpi ult, %46, %47 : index
            scf.yield %48 : i1
          }
          scf.yield %43 : i1
        }
        scf.if %37 {
          %38 = arith.muli %1, %c3 : index
          %39 = arith.muli %arg0, %c3 : index
          %40 = memref.load %arg2[%38] : memref<?xindex>
          %41 = memref.load %arg2[%39] : memref<?xindex>
          memref.store %41, %arg2[%38] : memref<?xindex>
          memref.store %40, %arg2[%39] : memref<?xindex>
          %42 = arith.addi %38, %c1 : index
          %43 = arith.addi %39, %c1 : index
          %44 = memref.load %arg2[%42] : memref<?xindex>
          %45 = memref.load %arg2[%43] : memref<?xindex>
          memref.store %45, %arg2[%42] : memref<?xindex>
          memref.store %44, %arg2[%43] : memref<?xindex>
          %46 = arith.addi %38, %c2 : index
          %47 = arith.addi %39, %c2 : index
          %48 = memref.load %arg2[%46] : memref<?xindex>
          %49 = memref.load %arg2[%47] : memref<?xindex>
          memref.store %49, %arg2[%46] : memref<?xindex>
          memref.store %48, %arg2[%47] : memref<?xindex>
          %50 = memref.load %arg3[%1] : memref<?xf64>
          %51 = memref.load %arg3[%arg0] : memref<?xf64>
          memref.store %51, %arg3[%1] : memref<?xf64>
          memref.store %50, %arg3[%arg0] : memref<?xf64>
        }
      }
    } else {
      %6 = arith.addi %arg0, %arg1 : index
      %7 = arith.shrui %6, %c1 : index
      %8 = arith.addi %1, %arg1 : index
      %9 = arith.shrui %8, %c1 : index
      %10 = arith.muli %7, %c3 : index
      %11 = arith.muli %arg0, %c3 : index
      %12 = memref.load %arg2[%10] : memref<?xindex>
      %13 = memref.load %arg2[%11] : memref<?xindex>
      %14 = arith.cmpi ne, %12, %13 : index
      %15 = scf.if %14 -> (i1) {
        %34 = arith.cmpi ult, %12, %13 : index
        scf.yield %34 : i1
      } else {
        %34 = arith.addi %10, %c1 : index
        %35 = arith.addi %11, %c1 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        %38 = arith.cmpi ne, %36, %37 : index
        %39 = scf.if %38 -> (i1) {
          %40 = arith.cmpi ult, %36, %37 : index
          scf.yield %40 : i1
        } else {
          %40 = arith.addi %10, %c2 : index
          %41 = arith.addi %11, %c2 : index
          %42 = memref.load %arg2[%40] : memref<?xindex>
          %43 = memref.load %arg2[%41] : memref<?xindex>
          %44 = arith.cmpi ult, %42, %43 : index
          scf.yield %44 : i1
        }
        scf.yield %39 : i1
      }
      scf.if %15 {
        %34 = arith.muli %7, %c3 : index
        %35 = arith.muli %arg0, %c3 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        memref.store %37, %arg2[%34] : memref<?xindex>
        memref.store %36, %arg2[%35] : memref<?xindex>
        %38 = arith.addi %34, %c1 : index
        %39 = arith.addi %35, %c1 : index
        %40 = memref.load %arg2[%38] : memref<?xindex>
        %41 = memref.load %arg2[%39] : memref<?xindex>
        memref.store %41, %arg2[%38] : memref<?xindex>
        memref.store %40, %arg2[%39] : memref<?xindex>
        %42 = arith.addi %34, %c2 : index
        %43 = arith.addi %35, %c2 : index
        %44 = memref.load %arg2[%42] : memref<?xindex>
        %45 = memref.load %arg2[%43] : memref<?xindex>
        memref.store %45, %arg2[%42] : memref<?xindex>
        memref.store %44, %arg2[%43] : memref<?xindex>
        %46 = memref.load %arg3[%7] : memref<?xf64>
        %47 = memref.load %arg3[%arg0] : memref<?xf64>
        memref.store %47, %arg3[%7] : memref<?xf64>
        memref.store %46, %arg3[%arg0] : memref<?xf64>
      }
      %16 = arith.muli %1, %c3 : index
      %17 = arith.muli %7, %c3 : index
      %18 = memref.load %arg2[%16] : memref<?xindex>
      %19 = memref.load %arg2[%17] : memref<?xindex>
      %20 = arith.cmpi ne, %18, %19 : index
      %21 = scf.if %20 -> (i1) {
        %34 = arith.cmpi ult, %18, %19 : index
        scf.yield %34 : i1
      } else {
        %34 = arith.addi %16, %c1 : index
        %35 = arith.addi %17, %c1 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        %38 = arith.cmpi ne, %36, %37 : index
        %39 = scf.if %38 -> (i1) {
          %40 = arith.cmpi ult, %36, %37 : index
          scf.yield %40 : i1
        } else {
          %40 = arith.addi %16, %c2 : index
          %41 = arith.addi %17, %c2 : index
          %42 = memref.load %arg2[%40] : memref<?xindex>
          %43 = memref.load %arg2[%41] : memref<?xindex>
          %44 = arith.cmpi ult, %42, %43 : index
          scf.yield %44 : i1
        }
        scf.yield %39 : i1
      }
      scf.if %21 {
        %34 = arith.muli %1, %c3 : index
        %35 = arith.muli %7, %c3 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        memref.store %37, %arg2[%34] : memref<?xindex>
        memref.store %36, %arg2[%35] : memref<?xindex>
        %38 = arith.addi %34, %c1 : index
        %39 = arith.addi %35, %c1 : index
        %40 = memref.load %arg2[%38] : memref<?xindex>
        %41 = memref.load %arg2[%39] : memref<?xindex>
        memref.store %41, %arg2[%38] : memref<?xindex>
        memref.store %40, %arg2[%39] : memref<?xindex>
        %42 = arith.addi %34, %c2 : index
        %43 = arith.addi %35, %c2 : index
        %44 = memref.load %arg2[%42] : memref<?xindex>
        %45 = memref.load %arg2[%43] : memref<?xindex>
        memref.store %45, %arg2[%42] : memref<?xindex>
        memref.store %44, %arg2[%43] : memref<?xindex>
        %46 = memref.load %arg3[%1] : memref<?xf64>
        %47 = memref.load %arg3[%7] : memref<?xf64>
        memref.store %47, %arg3[%1] : memref<?xf64>
        memref.store %46, %arg3[%7] : memref<?xf64>
        %48 = arith.muli %7, %c3 : index
        %49 = arith.muli %arg0, %c3 : index
        %50 = memref.load %arg2[%48] : memref<?xindex>
        %51 = memref.load %arg2[%49] : memref<?xindex>
        %52 = arith.cmpi ne, %50, %51 : index
        %53 = scf.if %52 -> (i1) {
          %54 = arith.cmpi ult, %50, %51 : index
          scf.yield %54 : i1
        } else {
          %54 = arith.addi %48, %c1 : index
          %55 = arith.addi %49, %c1 : index
          %56 = memref.load %arg2[%54] : memref<?xindex>
          %57 = memref.load %arg2[%55] : memref<?xindex>
          %58 = arith.cmpi ne, %56, %57 : index
          %59 = scf.if %58 -> (i1) {
            %60 = arith.cmpi ult, %56, %57 : index
            scf.yield %60 : i1
          } else {
            %60 = arith.addi %48, %c2 : index
            %61 = arith.addi %49, %c2 : index
            %62 = memref.load %arg2[%60] : memref<?xindex>
            %63 = memref.load %arg2[%61] : memref<?xindex>
            %64 = arith.cmpi ult, %62, %63 : index
            scf.yield %64 : i1
          }
          scf.yield %59 : i1
        }
        scf.if %53 {
          %54 = arith.muli %7, %c3 : index
          %55 = arith.muli %arg0, %c3 : index
          %56 = memref.load %arg2[%54] : memref<?xindex>
          %57 = memref.load %arg2[%55] : memref<?xindex>
          memref.store %57, %arg2[%54] : memref<?xindex>
          memref.store %56, %arg2[%55] : memref<?xindex>
          %58 = arith.addi %54, %c1 : index
          %59 = arith.addi %55, %c1 : index
          %60 = memref.load %arg2[%58] : memref<?xindex>
          %61 = memref.load %arg2[%59] : memref<?xindex>
          memref.store %61, %arg2[%58] : memref<?xindex>
          memref.store %60, %arg2[%59] : memref<?xindex>
          %62 = arith.addi %54, %c2 : index
          %63 = arith.addi %55, %c2 : index
          %64 = memref.load %arg2[%62] : memref<?xindex>
          %65 = memref.load %arg2[%63] : memref<?xindex>
          memref.store %65, %arg2[%62] : memref<?xindex>
          memref.store %64, %arg2[%63] : memref<?xindex>
          %66 = memref.load %arg3[%7] : memref<?xf64>
          %67 = memref.load %arg3[%arg0] : memref<?xf64>
          memref.store %67, %arg3[%7] : memref<?xf64>
          memref.store %66, %arg3[%arg0] : memref<?xf64>
        }
      }
      %22 = arith.muli %9, %c3 : index
      %23 = arith.muli %1, %c3 : index
      %24 = memref.load %arg2[%22] : memref<?xindex>
      %25 = memref.load %arg2[%23] : memref<?xindex>
      %26 = arith.cmpi ne, %24, %25 : index
      %27 = scf.if %26 -> (i1) {
        %34 = arith.cmpi ult, %24, %25 : index
        scf.yield %34 : i1
      } else {
        %34 = arith.addi %22, %c1 : index
        %35 = arith.addi %23, %c1 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        %38 = arith.cmpi ne, %36, %37 : index
        %39 = scf.if %38 -> (i1) {
          %40 = arith.cmpi ult, %36, %37 : index
          scf.yield %40 : i1
        } else {
          %40 = arith.addi %22, %c2 : index
          %41 = arith.addi %23, %c2 : index
          %42 = memref.load %arg2[%40] : memref<?xindex>
          %43 = memref.load %arg2[%41] : memref<?xindex>
          %44 = arith.cmpi ult, %42, %43 : index
          scf.yield %44 : i1
        }
        scf.yield %39 : i1
      }
      scf.if %27 {
        %34 = arith.muli %9, %c3 : index
        %35 = arith.muli %1, %c3 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        memref.store %37, %arg2[%34] : memref<?xindex>
        memref.store %36, %arg2[%35] : memref<?xindex>
        %38 = arith.addi %34, %c1 : index
        %39 = arith.addi %35, %c1 : index
        %40 = memref.load %arg2[%38] : memref<?xindex>
        %41 = memref.load %arg2[%39] : memref<?xindex>
        memref.store %41, %arg2[%38] : memref<?xindex>
        memref.store %40, %arg2[%39] : memref<?xindex>
        %42 = arith.addi %34, %c2 : index
        %43 = arith.addi %35, %c2 : index
        %44 = memref.load %arg2[%42] : memref<?xindex>
        %45 = memref.load %arg2[%43] : memref<?xindex>
        memref.store %45, %arg2[%42] : memref<?xindex>
        memref.store %44, %arg2[%43] : memref<?xindex>
        %46 = memref.load %arg3[%9] : memref<?xf64>
        %47 = memref.load %arg3[%1] : memref<?xf64>
        memref.store %47, %arg3[%9] : memref<?xf64>
        memref.store %46, %arg3[%1] : memref<?xf64>
        %48 = arith.muli %1, %c3 : index
        %49 = arith.muli %7, %c3 : index
        %50 = memref.load %arg2[%48] : memref<?xindex>
        %51 = memref.load %arg2[%49] : memref<?xindex>
        %52 = arith.cmpi ne, %50, %51 : index
        %53 = scf.if %52 -> (i1) {
          %54 = arith.cmpi ult, %50, %51 : index
          scf.yield %54 : i1
        } else {
          %54 = arith.addi %48, %c1 : index
          %55 = arith.addi %49, %c1 : index
          %56 = memref.load %arg2[%54] : memref<?xindex>
          %57 = memref.load %arg2[%55] : memref<?xindex>
          %58 = arith.cmpi ne, %56, %57 : index
          %59 = scf.if %58 -> (i1) {
            %60 = arith.cmpi ult, %56, %57 : index
            scf.yield %60 : i1
          } else {
            %60 = arith.addi %48, %c2 : index
            %61 = arith.addi %49, %c2 : index
            %62 = memref.load %arg2[%60] : memref<?xindex>
            %63 = memref.load %arg2[%61] : memref<?xindex>
            %64 = arith.cmpi ult, %62, %63 : index
            scf.yield %64 : i1
          }
          scf.yield %59 : i1
        }
        scf.if %53 {
          %54 = arith.muli %1, %c3 : index
          %55 = arith.muli %7, %c3 : index
          %56 = memref.load %arg2[%54] : memref<?xindex>
          %57 = memref.load %arg2[%55] : memref<?xindex>
          memref.store %57, %arg2[%54] : memref<?xindex>
          memref.store %56, %arg2[%55] : memref<?xindex>
          %58 = arith.addi %54, %c1 : index
          %59 = arith.addi %55, %c1 : index
          %60 = memref.load %arg2[%58] : memref<?xindex>
          %61 = memref.load %arg2[%59] : memref<?xindex>
          memref.store %61, %arg2[%58] : memref<?xindex>
          memref.store %60, %arg2[%59] : memref<?xindex>
          %62 = arith.addi %54, %c2 : index
          %63 = arith.addi %55, %c2 : index
          %64 = memref.load %arg2[%62] : memref<?xindex>
          %65 = memref.load %arg2[%63] : memref<?xindex>
          memref.store %65, %arg2[%62] : memref<?xindex>
          memref.store %64, %arg2[%63] : memref<?xindex>
          %66 = memref.load %arg3[%1] : memref<?xf64>
          %67 = memref.load %arg3[%7] : memref<?xf64>
          memref.store %67, %arg3[%1] : memref<?xf64>
          memref.store %66, %arg3[%7] : memref<?xf64>
          %68 = arith.muli %7, %c3 : index
          %69 = arith.muli %arg0, %c3 : index
          %70 = memref.load %arg2[%68] : memref<?xindex>
          %71 = memref.load %arg2[%69] : memref<?xindex>
          %72 = arith.cmpi ne, %70, %71 : index
          %73 = scf.if %72 -> (i1) {
            %74 = arith.cmpi ult, %70, %71 : index
            scf.yield %74 : i1
          } else {
            %74 = arith.addi %68, %c1 : index
            %75 = arith.addi %69, %c1 : index
            %76 = memref.load %arg2[%74] : memref<?xindex>
            %77 = memref.load %arg2[%75] : memref<?xindex>
            %78 = arith.cmpi ne, %76, %77 : index
            %79 = scf.if %78 -> (i1) {
              %80 = arith.cmpi ult, %76, %77 : index
              scf.yield %80 : i1
            } else {
              %80 = arith.addi %68, %c2 : index
              %81 = arith.addi %69, %c2 : index
              %82 = memref.load %arg2[%80] : memref<?xindex>
              %83 = memref.load %arg2[%81] : memref<?xindex>
              %84 = arith.cmpi ult, %82, %83 : index
              scf.yield %84 : i1
            }
            scf.yield %79 : i1
          }
          scf.if %73 {
            %74 = arith.muli %7, %c3 : index
            %75 = arith.muli %arg0, %c3 : index
            %76 = memref.load %arg2[%74] : memref<?xindex>
            %77 = memref.load %arg2[%75] : memref<?xindex>
            memref.store %77, %arg2[%74] : memref<?xindex>
            memref.store %76, %arg2[%75] : memref<?xindex>
            %78 = arith.addi %74, %c1 : index
            %79 = arith.addi %75, %c1 : index
            %80 = memref.load %arg2[%78] : memref<?xindex>
            %81 = memref.load %arg2[%79] : memref<?xindex>
            memref.store %81, %arg2[%78] : memref<?xindex>
            memref.store %80, %arg2[%79] : memref<?xindex>
            %82 = arith.addi %74, %c2 : index
            %83 = arith.addi %75, %c2 : index
            %84 = memref.load %arg2[%82] : memref<?xindex>
            %85 = memref.load %arg2[%83] : memref<?xindex>
            memref.store %85, %arg2[%82] : memref<?xindex>
            memref.store %84, %arg2[%83] : memref<?xindex>
            %86 = memref.load %arg3[%7] : memref<?xf64>
            %87 = memref.load %arg3[%arg0] : memref<?xf64>
            memref.store %87, %arg3[%7] : memref<?xf64>
            memref.store %86, %arg3[%arg0] : memref<?xf64>
          }
        }
      }
      %28 = arith.muli %2, %c3 : index
      %29 = arith.muli %9, %c3 : index
      %30 = memref.load %arg2[%28] : memref<?xindex>
      %31 = memref.load %arg2[%29] : memref<?xindex>
      %32 = arith.cmpi ne, %30, %31 : index
      %33 = scf.if %32 -> (i1) {
        %34 = arith.cmpi ult, %30, %31 : index
        scf.yield %34 : i1
      } else {
        %34 = arith.addi %28, %c1 : index
        %35 = arith.addi %29, %c1 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        %38 = arith.cmpi ne, %36, %37 : index
        %39 = scf.if %38 -> (i1) {
          %40 = arith.cmpi ult, %36, %37 : index
          scf.yield %40 : i1
        } else {
          %40 = arith.addi %28, %c2 : index
          %41 = arith.addi %29, %c2 : index
          %42 = memref.load %arg2[%40] : memref<?xindex>
          %43 = memref.load %arg2[%41] : memref<?xindex>
          %44 = arith.cmpi ult, %42, %43 : index
          scf.yield %44 : i1
        }
        scf.yield %39 : i1
      }
      scf.if %33 {
        %34 = arith.muli %2, %c3 : index
        %35 = arith.muli %9, %c3 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        memref.store %37, %arg2[%34] : memref<?xindex>
        memref.store %36, %arg2[%35] : memref<?xindex>
        %38 = arith.addi %34, %c1 : index
        %39 = arith.addi %35, %c1 : index
        %40 = memref.load %arg2[%38] : memref<?xindex>
        %41 = memref.load %arg2[%39] : memref<?xindex>
        memref.store %41, %arg2[%38] : memref<?xindex>
        memref.store %40, %arg2[%39] : memref<?xindex>
        %42 = arith.addi %34, %c2 : index
        %43 = arith.addi %35, %c2 : index
        %44 = memref.load %arg2[%42] : memref<?xindex>
        %45 = memref.load %arg2[%43] : memref<?xindex>
        memref.store %45, %arg2[%42] : memref<?xindex>
        memref.store %44, %arg2[%43] : memref<?xindex>
        %46 = memref.load %arg3[%2] : memref<?xf64>
        %47 = memref.load %arg3[%9] : memref<?xf64>
        memref.store %47, %arg3[%2] : memref<?xf64>
        memref.store %46, %arg3[%9] : memref<?xf64>
        %48 = arith.muli %9, %c3 : index
        %49 = arith.muli %1, %c3 : index
        %50 = memref.load %arg2[%48] : memref<?xindex>
        %51 = memref.load %arg2[%49] : memref<?xindex>
        %52 = arith.cmpi ne, %50, %51 : index
        %53 = scf.if %52 -> (i1) {
          %54 = arith.cmpi ult, %50, %51 : index
          scf.yield %54 : i1
        } else {
          %54 = arith.addi %48, %c1 : index
          %55 = arith.addi %49, %c1 : index
          %56 = memref.load %arg2[%54] : memref<?xindex>
          %57 = memref.load %arg2[%55] : memref<?xindex>
          %58 = arith.cmpi ne, %56, %57 : index
          %59 = scf.if %58 -> (i1) {
            %60 = arith.cmpi ult, %56, %57 : index
            scf.yield %60 : i1
          } else {
            %60 = arith.addi %48, %c2 : index
            %61 = arith.addi %49, %c2 : index
            %62 = memref.load %arg2[%60] : memref<?xindex>
            %63 = memref.load %arg2[%61] : memref<?xindex>
            %64 = arith.cmpi ult, %62, %63 : index
            scf.yield %64 : i1
          }
          scf.yield %59 : i1
        }
        scf.if %53 {
          %54 = arith.muli %9, %c3 : index
          %55 = arith.muli %1, %c3 : index
          %56 = memref.load %arg2[%54] : memref<?xindex>
          %57 = memref.load %arg2[%55] : memref<?xindex>
          memref.store %57, %arg2[%54] : memref<?xindex>
          memref.store %56, %arg2[%55] : memref<?xindex>
          %58 = arith.addi %54, %c1 : index
          %59 = arith.addi %55, %c1 : index
          %60 = memref.load %arg2[%58] : memref<?xindex>
          %61 = memref.load %arg2[%59] : memref<?xindex>
          memref.store %61, %arg2[%58] : memref<?xindex>
          memref.store %60, %arg2[%59] : memref<?xindex>
          %62 = arith.addi %54, %c2 : index
          %63 = arith.addi %55, %c2 : index
          %64 = memref.load %arg2[%62] : memref<?xindex>
          %65 = memref.load %arg2[%63] : memref<?xindex>
          memref.store %65, %arg2[%62] : memref<?xindex>
          memref.store %64, %arg2[%63] : memref<?xindex>
          %66 = memref.load %arg3[%9] : memref<?xf64>
          %67 = memref.load %arg3[%1] : memref<?xf64>
          memref.store %67, %arg3[%9] : memref<?xf64>
          memref.store %66, %arg3[%1] : memref<?xf64>
          %68 = arith.muli %1, %c3 : index
          %69 = arith.muli %7, %c3 : index
          %70 = memref.load %arg2[%68] : memref<?xindex>
          %71 = memref.load %arg2[%69] : memref<?xindex>
          %72 = arith.cmpi ne, %70, %71 : index
          %73 = scf.if %72 -> (i1) {
            %74 = arith.cmpi ult, %70, %71 : index
            scf.yield %74 : i1
          } else {
            %74 = arith.addi %68, %c1 : index
            %75 = arith.addi %69, %c1 : index
            %76 = memref.load %arg2[%74] : memref<?xindex>
            %77 = memref.load %arg2[%75] : memref<?xindex>
            %78 = arith.cmpi ne, %76, %77 : index
            %79 = scf.if %78 -> (i1) {
              %80 = arith.cmpi ult, %76, %77 : index
              scf.yield %80 : i1
            } else {
              %80 = arith.addi %68, %c2 : index
              %81 = arith.addi %69, %c2 : index
              %82 = memref.load %arg2[%80] : memref<?xindex>
              %83 = memref.load %arg2[%81] : memref<?xindex>
              %84 = arith.cmpi ult, %82, %83 : index
              scf.yield %84 : i1
            }
            scf.yield %79 : i1
          }
          scf.if %73 {
            %74 = arith.muli %1, %c3 : index
            %75 = arith.muli %7, %c3 : index
            %76 = memref.load %arg2[%74] : memref<?xindex>
            %77 = memref.load %arg2[%75] : memref<?xindex>
            memref.store %77, %arg2[%74] : memref<?xindex>
            memref.store %76, %arg2[%75] : memref<?xindex>
            %78 = arith.addi %74, %c1 : index
            %79 = arith.addi %75, %c1 : index
            %80 = memref.load %arg2[%78] : memref<?xindex>
            %81 = memref.load %arg2[%79] : memref<?xindex>
            memref.store %81, %arg2[%78] : memref<?xindex>
            memref.store %80, %arg2[%79] : memref<?xindex>
            %82 = arith.addi %74, %c2 : index
            %83 = arith.addi %75, %c2 : index
            %84 = memref.load %arg2[%82] : memref<?xindex>
            %85 = memref.load %arg2[%83] : memref<?xindex>
            memref.store %85, %arg2[%82] : memref<?xindex>
            memref.store %84, %arg2[%83] : memref<?xindex>
            %86 = memref.load %arg3[%1] : memref<?xf64>
            %87 = memref.load %arg3[%7] : memref<?xf64>
            memref.store %87, %arg3[%1] : memref<?xf64>
            memref.store %86, %arg3[%7] : memref<?xf64>
            %88 = arith.muli %7, %c3 : index
            %89 = arith.muli %arg0, %c3 : index
            %90 = memref.load %arg2[%88] : memref<?xindex>
            %91 = memref.load %arg2[%89] : memref<?xindex>
            %92 = arith.cmpi ne, %90, %91 : index
            %93 = scf.if %92 -> (i1) {
              %94 = arith.cmpi ult, %90, %91 : index
              scf.yield %94 : i1
            } else {
              %94 = arith.addi %88, %c1 : index
              %95 = arith.addi %89, %c1 : index
              %96 = memref.load %arg2[%94] : memref<?xindex>
              %97 = memref.load %arg2[%95] : memref<?xindex>
              %98 = arith.cmpi ne, %96, %97 : index
              %99 = scf.if %98 -> (i1) {
                %100 = arith.cmpi ult, %96, %97 : index
                scf.yield %100 : i1
              } else {
                %100 = arith.addi %88, %c2 : index
                %101 = arith.addi %89, %c2 : index
                %102 = memref.load %arg2[%100] : memref<?xindex>
                %103 = memref.load %arg2[%101] : memref<?xindex>
                %104 = arith.cmpi ult, %102, %103 : index
                scf.yield %104 : i1
              }
              scf.yield %99 : i1
            }
            scf.if %93 {
              %94 = arith.muli %7, %c3 : index
              %95 = arith.muli %arg0, %c3 : index
              %96 = memref.load %arg2[%94] : memref<?xindex>
              %97 = memref.load %arg2[%95] : memref<?xindex>
              memref.store %97, %arg2[%94] : memref<?xindex>
              memref.store %96, %arg2[%95] : memref<?xindex>
              %98 = arith.addi %94, %c1 : index
              %99 = arith.addi %95, %c1 : index
              %100 = memref.load %arg2[%98] : memref<?xindex>
              %101 = memref.load %arg2[%99] : memref<?xindex>
              memref.store %101, %arg2[%98] : memref<?xindex>
              memref.store %100, %arg2[%99] : memref<?xindex>
              %102 = arith.addi %94, %c2 : index
              %103 = arith.addi %95, %c2 : index
              %104 = memref.load %arg2[%102] : memref<?xindex>
              %105 = memref.load %arg2[%103] : memref<?xindex>
              memref.store %105, %arg2[%102] : memref<?xindex>
              memref.store %104, %arg2[%103] : memref<?xindex>
              %106 = memref.load %arg3[%7] : memref<?xf64>
              %107 = memref.load %arg3[%arg0] : memref<?xf64>
              memref.store %107, %arg3[%7] : memref<?xf64>
              memref.store %106, %arg3[%arg0] : memref<?xf64>
            }
          }
        }
      }
    }
    %5:4 = scf.while (%arg4 = %arg0, %arg5 = %2, %arg6 = %1, %arg7 = %true) : (index, index, index, i1) -> (index, index, index, i1) {
      scf.condition(%arg7) %arg4, %arg5, %arg6, %arg7 : index, index, index, i1
    } do {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: i1):
      %6 = scf.while (%arg8 = %arg4) : (index) -> index {
        %22 = arith.muli %arg8, %c3 : index
        %23 = arith.muli %arg6, %c3 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        %26 = arith.cmpi ne, %24, %25 : index
        %27 = scf.if %26 -> (i1) {
          %28 = arith.cmpi ult, %24, %25 : index
          scf.yield %28 : i1
        } else {
          %28 = arith.addi %22, %c1 : index
          %29 = arith.addi %23, %c1 : index
          %30 = memref.load %arg2[%28] : memref<?xindex>
          %31 = memref.load %arg2[%29] : memref<?xindex>
          %32 = arith.cmpi ne, %30, %31 : index
          %33 = scf.if %32 -> (i1) {
            %34 = arith.cmpi ult, %30, %31 : index
            scf.yield %34 : i1
          } else {
            %34 = arith.addi %22, %c2 : index
            %35 = arith.addi %23, %c2 : index
            %36 = memref.load %arg2[%34] : memref<?xindex>
            %37 = memref.load %arg2[%35] : memref<?xindex>
            %38 = arith.cmpi ult, %36, %37 : index
            scf.yield %38 : i1
          }
          scf.yield %33 : i1
        }
        scf.condition(%27) %arg8 : index
      } do {
      ^bb0(%arg8: index):
        %22 = arith.addi %arg8, %c1 : index
        scf.yield %22 : index
      }
      %7 = arith.muli %6, %c3 : index
      %8 = arith.muli %arg6, %c3 : index
      %9 = memref.load %arg2[%7] : memref<?xindex>
      %10 = memref.load %arg2[%8] : memref<?xindex>
      %11 = arith.cmpi ne, %9, %10 : index
      %12 = scf.if %11 -> (i1) {
        scf.yield %false : i1
      } else {
        %22 = arith.addi %7, %c1 : index
        %23 = arith.addi %8, %c1 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        %26 = arith.cmpi ne, %24, %25 : index
        %27 = scf.if %26 -> (i1) {
          scf.yield %false : i1
        } else {
          %28 = arith.addi %7, %c2 : index
          %29 = arith.addi %8, %c2 : index
          %30 = memref.load %arg2[%28] : memref<?xindex>
          %31 = memref.load %arg2[%29] : memref<?xindex>
          %32 = arith.cmpi eq, %30, %31 : index
          scf.yield %32 : i1
        }
        scf.yield %27 : i1
      }
      %13 = scf.while (%arg8 = %arg5) : (index) -> index {
        %22 = arith.muli %arg6, %c3 : index
        %23 = arith.muli %arg8, %c3 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        %26 = arith.cmpi ne, %24, %25 : index
        %27 = scf.if %26 -> (i1) {
          %28 = arith.cmpi ult, %24, %25 : index
          scf.yield %28 : i1
        } else {
          %28 = arith.addi %22, %c1 : index
          %29 = arith.addi %23, %c1 : index
          %30 = memref.load %arg2[%28] : memref<?xindex>
          %31 = memref.load %arg2[%29] : memref<?xindex>
          %32 = arith.cmpi ne, %30, %31 : index
          %33 = scf.if %32 -> (i1) {
            %34 = arith.cmpi ult, %30, %31 : index
            scf.yield %34 : i1
          } else {
            %34 = arith.addi %22, %c2 : index
            %35 = arith.addi %23, %c2 : index
            %36 = memref.load %arg2[%34] : memref<?xindex>
            %37 = memref.load %arg2[%35] : memref<?xindex>
            %38 = arith.cmpi ult, %36, %37 : index
            scf.yield %38 : i1
          }
          scf.yield %33 : i1
        }
        scf.condition(%27) %arg8 : index
      } do {
      ^bb0(%arg8: index):
        %22 = arith.addi %arg8, %c-1 : index
        scf.yield %22 : index
      }
      %14 = arith.muli %13, %c3 : index
      %15 = arith.muli %arg6, %c3 : index
      %16 = memref.load %arg2[%14] : memref<?xindex>
      %17 = memref.load %arg2[%15] : memref<?xindex>
      %18 = arith.cmpi ne, %16, %17 : index
      %19 = scf.if %18 -> (i1) {
        scf.yield %false : i1
      } else {
        %22 = arith.addi %14, %c1 : index
        %23 = arith.addi %15, %c1 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        %26 = arith.cmpi ne, %24, %25 : index
        %27 = scf.if %26 -> (i1) {
          scf.yield %false : i1
        } else {
          %28 = arith.addi %14, %c2 : index
          %29 = arith.addi %15, %c2 : index
          %30 = memref.load %arg2[%28] : memref<?xindex>
          %31 = memref.load %arg2[%29] : memref<?xindex>
          %32 = arith.cmpi eq, %30, %31 : index
          scf.yield %32 : i1
        }
        scf.yield %27 : i1
      }
      %20 = arith.cmpi ult, %6, %13 : index
      %21:4 = scf.if %20 -> (index, index, index, i1) {
        %22 = arith.muli %6, %c3 : index
        %23 = arith.muli %13, %c3 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        memref.store %25, %arg2[%22] : memref<?xindex>
        memref.store %24, %arg2[%23] : memref<?xindex>
        %26 = arith.addi %22, %c1 : index
        %27 = arith.addi %23, %c1 : index
        %28 = memref.load %arg2[%26] : memref<?xindex>
        %29 = memref.load %arg2[%27] : memref<?xindex>
        memref.store %29, %arg2[%26] : memref<?xindex>
        memref.store %28, %arg2[%27] : memref<?xindex>
        %30 = arith.addi %22, %c2 : index
        %31 = arith.addi %23, %c2 : index
        %32 = memref.load %arg2[%30] : memref<?xindex>
        %33 = memref.load %arg2[%31] : memref<?xindex>
        memref.store %33, %arg2[%30] : memref<?xindex>
        memref.store %32, %arg2[%31] : memref<?xindex>
        %34 = memref.load %arg3[%6] : memref<?xf64>
        %35 = memref.load %arg3[%13] : memref<?xf64>
        memref.store %35, %arg3[%6] : memref<?xf64>
        memref.store %34, %arg3[%13] : memref<?xf64>
        %36 = arith.cmpi eq, %6, %arg6 : index
        %37 = scf.if %36 -> (index) {
          scf.yield %13 : index
        } else {
          %40 = arith.cmpi eq, %13, %arg6 : index
          %41 = scf.if %40 -> (index) {
            scf.yield %6 : index
          } else {
            scf.yield %arg6 : index
          }
          scf.yield %41 : index
        }
        %38 = arith.andi %12, %19 : i1
        %39:2 = scf.if %38 -> (index, index) {
          %40 = arith.addi %6, %c1 : index
          %41 = arith.subi %13, %c1 : index
          scf.yield %40, %41 : index, index
        } else {
          scf.yield %6, %13 : index, index
        }
        scf.yield %39#0, %39#1, %37, %true : index, index, index, i1
      } else {
        %22 = arith.addi %13, %c1 : index
        scf.yield %6, %13, %22, %false : index, index, index, i1
      }
      scf.yield %21#0, %21#1, %21#2, %21#3 : index, index, index, i1
    }
    return %5#2 : index
  }
  func.func private @_sparse_hybrid_qsort_0_1_2_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>, %arg4: i64) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c30 = arith.constant 30 : index
    %c1 = arith.constant 1 : index
    %0:2 = scf.while (%arg5 = %arg0, %arg6 = %arg1) : (index, index) -> (index, index) {
      %1 = arith.addi %arg5, %c1 : index
      %2 = arith.cmpi ult, %1, %arg6 : index
      scf.condition(%2) %arg5, %arg6 : index, index
    } do {
    ^bb0(%arg5: index, %arg6: index):
      %1 = arith.subi %arg6, %arg5 : index
      %2 = arith.cmpi ule, %1, %c30 : index
      %3:2 = scf.if %2 -> (index, index) {
        func.call @_sparse_sort_stable_0_1_2_index_coo_0_f64(%arg5, %arg6, %arg2, %arg3) : (index, index, memref<?xindex>, memref<?xf64>) -> ()
        scf.yield %arg5, %arg5 : index, index
      } else {
        %4 = arith.subi %arg4, %c1_i64 : i64
        %5 = arith.cmpi ule, %4, %c0_i64 : i64
        %6:2 = scf.if %5 -> (index, index) {
          func.call @_sparse_heap_sort_0_1_2_index_coo_0_f64(%arg5, %arg6, %arg2, %arg3) : (index, index, memref<?xindex>, memref<?xf64>) -> ()
          scf.yield %arg5, %arg5 : index, index
        } else {
          %7 = func.call @_sparse_partition_0_1_2_index_coo_0_f64(%arg5, %arg6, %arg2, %arg3) : (index, index, memref<?xindex>, memref<?xf64>) -> index
          %8 = arith.subi %7, %arg5 : index
          %9 = arith.subi %arg6, %7 : index
          %10 = arith.subi %arg6, %arg5 : index
          %11 = arith.cmpi ugt, %10, %c2 : index
          %12:2 = scf.if %11 -> (index, index) {
            %13 = arith.cmpi ule, %8, %9 : index
            %14:2 = scf.if %13 -> (index, index) {
              %15 = arith.cmpi ne, %8, %c0 : index
              scf.if %15 {
                func.call @_sparse_hybrid_qsort_0_1_2_index_coo_0_f64(%arg5, %7, %arg2, %arg3, %4) : (index, index, memref<?xindex>, memref<?xf64>, i64) -> ()
              }
              scf.yield %7, %arg6 : index, index
            } else {
              %15 = arith.cmpi ne, %9, %c0 : index
              scf.if %15 {
                func.call @_sparse_hybrid_qsort_0_1_2_index_coo_0_f64(%7, %arg6, %arg2, %arg3, %4) : (index, index, memref<?xindex>, memref<?xf64>, i64) -> ()
              }
              scf.yield %arg5, %7 : index, index
            }
            scf.yield %14#0, %14#1 : index, index
          } else {
            scf.yield %arg5, %arg5 : index, index
          }
          scf.yield %12#0, %12#1 : index, index
        }
        scf.yield %6#0, %6#1 : index, index
      }
      scf.yield %3#0, %3#1 : index, index
    }
    return
  }
  func.func private @_sparse_binary_search_0_1_2_3_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>) -> index {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = scf.while (%arg4 = %arg0, %arg5 = %arg1) : (index, index) -> (index, index) {
      %1 = arith.cmpi ult, %arg4, %arg5 : index
      scf.condition(%1) %arg4, %arg5 : index, index
    } do {
    ^bb0(%arg4: index, %arg5: index):
      %1 = arith.addi %arg4, %arg5 : index
      %2 = arith.shrui %1, %c1 : index
      %3 = arith.addi %2, %c1 : index
      %4 = arith.muli %arg1, %c4 : index
      %5 = arith.muli %2, %c4 : index
      %6 = memref.load %arg2[%4] : memref<?xindex>
      %7 = memref.load %arg2[%5] : memref<?xindex>
      %8 = arith.cmpi ne, %6, %7 : index
      %9 = scf.if %8 -> (i1) {
        %12 = arith.cmpi ult, %6, %7 : index
        scf.yield %12 : i1
      } else {
        %12 = arith.addi %4, %c1 : index
        %13 = arith.addi %5, %c1 : index
        %14 = memref.load %arg2[%12] : memref<?xindex>
        %15 = memref.load %arg2[%13] : memref<?xindex>
        %16 = arith.cmpi ne, %14, %15 : index
        %17 = scf.if %16 -> (i1) {
          %18 = arith.cmpi ult, %14, %15 : index
          scf.yield %18 : i1
        } else {
          %18 = arith.addi %4, %c2 : index
          %19 = arith.addi %5, %c2 : index
          %20 = memref.load %arg2[%18] : memref<?xindex>
          %21 = memref.load %arg2[%19] : memref<?xindex>
          %22 = arith.cmpi ne, %20, %21 : index
          %23 = scf.if %22 -> (i1) {
            %24 = arith.cmpi ult, %20, %21 : index
            scf.yield %24 : i1
          } else {
            %24 = arith.addi %4, %c3 : index
            %25 = arith.addi %5, %c3 : index
            %26 = memref.load %arg2[%24] : memref<?xindex>
            %27 = memref.load %arg2[%25] : memref<?xindex>
            %28 = arith.cmpi ult, %26, %27 : index
            scf.yield %28 : i1
          }
          scf.yield %23 : i1
        }
        scf.yield %17 : i1
      }
      %10 = arith.select %9, %arg4, %3 : index
      %11 = arith.select %9, %2, %arg5 : index
      scf.yield %10, %11 : index, index
    }
    return %0#0 : index
  }
  func.func private @_sparse_sort_stable_0_1_2_3_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg0, %c1 : index
    scf.for %arg4 = %0 to %arg1 step %c1 {
      %1 = func.call @_sparse_binary_search_0_1_2_3_index_coo_0_f64(%arg0, %arg4, %arg2, %arg3) : (index, index, memref<?xindex>, memref<?xf64>) -> index
      %2 = arith.muli %arg4, %c4 : index
      %3 = memref.load %arg2[%2] : memref<?xindex>
      %4 = arith.addi %2, %c1 : index
      %5 = memref.load %arg2[%4] : memref<?xindex>
      %6 = arith.addi %2, %c2 : index
      %7 = memref.load %arg2[%6] : memref<?xindex>
      %8 = arith.addi %2, %c3 : index
      %9 = memref.load %arg2[%8] : memref<?xindex>
      %10 = memref.load %arg3[%arg4] : memref<?xf64>
      %11 = arith.subi %arg4, %1 : index
      scf.for %arg5 = %c0 to %11 step %c1 {
        %16 = arith.subi %arg4, %arg5 : index
        %17 = arith.subi %16, %c1 : index
        %18 = arith.muli %17, %c4 : index
        %19 = arith.muli %16, %c4 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        memref.store %20, %arg2[%19] : memref<?xindex>
        %21 = arith.addi %18, %c1 : index
        %22 = arith.addi %19, %c1 : index
        %23 = memref.load %arg2[%21] : memref<?xindex>
        memref.store %23, %arg2[%22] : memref<?xindex>
        %24 = arith.addi %18, %c2 : index
        %25 = arith.addi %19, %c2 : index
        %26 = memref.load %arg2[%24] : memref<?xindex>
        memref.store %26, %arg2[%25] : memref<?xindex>
        %27 = arith.addi %18, %c3 : index
        %28 = arith.addi %19, %c3 : index
        %29 = memref.load %arg2[%27] : memref<?xindex>
        memref.store %29, %arg2[%28] : memref<?xindex>
        %30 = memref.load %arg3[%17] : memref<?xf64>
        memref.store %30, %arg3[%16] : memref<?xf64>
      }
      %12 = arith.muli %1, %c4 : index
      memref.store %3, %arg2[%12] : memref<?xindex>
      %13 = arith.addi %12, %c1 : index
      memref.store %5, %arg2[%13] : memref<?xindex>
      %14 = arith.addi %12, %c2 : index
      memref.store %7, %arg2[%14] : memref<?xindex>
      %15 = arith.addi %12, %c3 : index
      memref.store %9, %arg2[%15] : memref<?xindex>
      memref.store %10, %arg3[%1] : memref<?xf64>
    }
    return
  }
  func.func private @_sparse_shift_down_0_1_2_3_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>, %arg4: index) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.cmpi uge, %arg4, %c2 : index
    scf.if %0 {
      %1 = arith.subi %arg1, %arg0 : index
      %2 = arith.subi %arg4, %c2 : index
      %3 = arith.shrui %2, %c1 : index
      %4 = arith.cmpi uge, %3, %1 : index
      scf.if %4 {
        %5 = arith.shli %1, %c1 : index
        %6 = arith.addi %5, %c1 : index
        %7 = arith.addi %6, %arg0 : index
        %8 = arith.addi %6, %c1 : index
        %9 = arith.cmpi ult, %8, %arg4 : index
        %10:2 = scf.if %9 -> (index, index) {
          %12 = arith.addi %8, %arg0 : index
          %13 = arith.muli %7, %c4 : index
          %14 = arith.muli %12, %c4 : index
          %15 = memref.load %arg2[%13] : memref<?xindex>
          %16 = memref.load %arg2[%14] : memref<?xindex>
          %17 = arith.cmpi ne, %15, %16 : index
          %18 = scf.if %17 -> (i1) {
            %20 = arith.cmpi ult, %15, %16 : index
            scf.yield %20 : i1
          } else {
            %20 = arith.addi %13, %c1 : index
            %21 = arith.addi %14, %c1 : index
            %22 = memref.load %arg2[%20] : memref<?xindex>
            %23 = memref.load %arg2[%21] : memref<?xindex>
            %24 = arith.cmpi ne, %22, %23 : index
            %25 = scf.if %24 -> (i1) {
              %26 = arith.cmpi ult, %22, %23 : index
              scf.yield %26 : i1
            } else {
              %26 = arith.addi %13, %c2 : index
              %27 = arith.addi %14, %c2 : index
              %28 = memref.load %arg2[%26] : memref<?xindex>
              %29 = memref.load %arg2[%27] : memref<?xindex>
              %30 = arith.cmpi ne, %28, %29 : index
              %31 = scf.if %30 -> (i1) {
                %32 = arith.cmpi ult, %28, %29 : index
                scf.yield %32 : i1
              } else {
                %32 = arith.addi %13, %c3 : index
                %33 = arith.addi %14, %c3 : index
                %34 = memref.load %arg2[%32] : memref<?xindex>
                %35 = memref.load %arg2[%33] : memref<?xindex>
                %36 = arith.cmpi ult, %34, %35 : index
                scf.yield %36 : i1
              }
              scf.yield %31 : i1
            }
            scf.yield %25 : i1
          }
          %19:2 = scf.if %18 -> (index, index) {
            scf.yield %8, %12 : index, index
          } else {
            scf.yield %6, %7 : index, index
          }
          scf.yield %19#0, %19#1 : index, index
        } else {
          scf.yield %6, %7 : index, index
        }
        %11:3 = scf.while (%arg5 = %arg1, %arg6 = %10#0, %arg7 = %10#1) : (index, index, index) -> (index, index, index) {
          %12 = arith.muli %arg5, %c4 : index
          %13 = arith.muli %arg7, %c4 : index
          %14 = memref.load %arg2[%12] : memref<?xindex>
          %15 = memref.load %arg2[%13] : memref<?xindex>
          %16 = arith.cmpi ne, %14, %15 : index
          %17 = scf.if %16 -> (i1) {
            %18 = arith.cmpi ult, %14, %15 : index
            scf.yield %18 : i1
          } else {
            %18 = arith.addi %12, %c1 : index
            %19 = arith.addi %13, %c1 : index
            %20 = memref.load %arg2[%18] : memref<?xindex>
            %21 = memref.load %arg2[%19] : memref<?xindex>
            %22 = arith.cmpi ne, %20, %21 : index
            %23 = scf.if %22 -> (i1) {
              %24 = arith.cmpi ult, %20, %21 : index
              scf.yield %24 : i1
            } else {
              %24 = arith.addi %12, %c2 : index
              %25 = arith.addi %13, %c2 : index
              %26 = memref.load %arg2[%24] : memref<?xindex>
              %27 = memref.load %arg2[%25] : memref<?xindex>
              %28 = arith.cmpi ne, %26, %27 : index
              %29 = scf.if %28 -> (i1) {
                %30 = arith.cmpi ult, %26, %27 : index
                scf.yield %30 : i1
              } else {
                %30 = arith.addi %12, %c3 : index
                %31 = arith.addi %13, %c3 : index
                %32 = memref.load %arg2[%30] : memref<?xindex>
                %33 = memref.load %arg2[%31] : memref<?xindex>
                %34 = arith.cmpi ult, %32, %33 : index
                scf.yield %34 : i1
              }
              scf.yield %29 : i1
            }
            scf.yield %23 : i1
          }
          scf.condition(%17) %arg5, %arg6, %arg7 : index, index, index
        } do {
        ^bb0(%arg5: index, %arg6: index, %arg7: index):
          %12 = arith.muli %arg5, %c4 : index
          %13 = arith.muli %arg7, %c4 : index
          %14 = memref.load %arg2[%12] : memref<?xindex>
          %15 = memref.load %arg2[%13] : memref<?xindex>
          memref.store %15, %arg2[%12] : memref<?xindex>
          memref.store %14, %arg2[%13] : memref<?xindex>
          %16 = arith.addi %12, %c1 : index
          %17 = arith.addi %13, %c1 : index
          %18 = memref.load %arg2[%16] : memref<?xindex>
          %19 = memref.load %arg2[%17] : memref<?xindex>
          memref.store %19, %arg2[%16] : memref<?xindex>
          memref.store %18, %arg2[%17] : memref<?xindex>
          %20 = arith.addi %12, %c2 : index
          %21 = arith.addi %13, %c2 : index
          %22 = memref.load %arg2[%20] : memref<?xindex>
          %23 = memref.load %arg2[%21] : memref<?xindex>
          memref.store %23, %arg2[%20] : memref<?xindex>
          memref.store %22, %arg2[%21] : memref<?xindex>
          %24 = arith.addi %12, %c3 : index
          %25 = arith.addi %13, %c3 : index
          %26 = memref.load %arg2[%24] : memref<?xindex>
          %27 = memref.load %arg2[%25] : memref<?xindex>
          memref.store %27, %arg2[%24] : memref<?xindex>
          memref.store %26, %arg2[%25] : memref<?xindex>
          %28 = memref.load %arg3[%arg5] : memref<?xf64>
          %29 = memref.load %arg3[%arg7] : memref<?xf64>
          memref.store %29, %arg3[%arg5] : memref<?xf64>
          memref.store %28, %arg3[%arg7] : memref<?xf64>
          %30 = arith.cmpi uge, %3, %arg6 : index
          %31:2 = scf.if %30 -> (index, index) {
            %32 = arith.shli %arg6, %c1 : index
            %33 = arith.addi %32, %c1 : index
            %34 = arith.addi %33, %arg0 : index
            %35 = arith.addi %33, %c1 : index
            %36 = arith.cmpi ult, %35, %arg4 : index
            %37:2 = scf.if %36 -> (index, index) {
              %38 = arith.addi %35, %arg0 : index
              %39 = arith.muli %34, %c4 : index
              %40 = arith.muli %38, %c4 : index
              %41 = memref.load %arg2[%39] : memref<?xindex>
              %42 = memref.load %arg2[%40] : memref<?xindex>
              %43 = arith.cmpi ne, %41, %42 : index
              %44 = scf.if %43 -> (i1) {
                %46 = arith.cmpi ult, %41, %42 : index
                scf.yield %46 : i1
              } else {
                %46 = arith.addi %39, %c1 : index
                %47 = arith.addi %40, %c1 : index
                %48 = memref.load %arg2[%46] : memref<?xindex>
                %49 = memref.load %arg2[%47] : memref<?xindex>
                %50 = arith.cmpi ne, %48, %49 : index
                %51 = scf.if %50 -> (i1) {
                  %52 = arith.cmpi ult, %48, %49 : index
                  scf.yield %52 : i1
                } else {
                  %52 = arith.addi %39, %c2 : index
                  %53 = arith.addi %40, %c2 : index
                  %54 = memref.load %arg2[%52] : memref<?xindex>
                  %55 = memref.load %arg2[%53] : memref<?xindex>
                  %56 = arith.cmpi ne, %54, %55 : index
                  %57 = scf.if %56 -> (i1) {
                    %58 = arith.cmpi ult, %54, %55 : index
                    scf.yield %58 : i1
                  } else {
                    %58 = arith.addi %39, %c3 : index
                    %59 = arith.addi %40, %c3 : index
                    %60 = memref.load %arg2[%58] : memref<?xindex>
                    %61 = memref.load %arg2[%59] : memref<?xindex>
                    %62 = arith.cmpi ult, %60, %61 : index
                    scf.yield %62 : i1
                  }
                  scf.yield %57 : i1
                }
                scf.yield %51 : i1
              }
              %45:2 = scf.if %44 -> (index, index) {
                scf.yield %35, %38 : index, index
              } else {
                scf.yield %33, %34 : index, index
              }
              scf.yield %45#0, %45#1 : index, index
            } else {
              scf.yield %33, %34 : index, index
            }
            scf.yield %37#0, %37#1 : index, index
          } else {
            scf.yield %arg6, %arg7 : index, index
          }
          scf.yield %arg7, %31#0, %31#1 : index, index, index
        }
      }
    }
    return
  }
  func.func private @_sparse_heap_sort_0_1_2_3_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.subi %arg1, %arg0 : index
    %1 = arith.subi %0, %c2 : index
    %2 = arith.shrui %1, %c1 : index
    %3 = arith.addi %2, %c1 : index
    scf.for %arg4 = %c0 to %3 step %c1 {
      %5 = arith.subi %2, %arg4 : index
      %6 = arith.addi %arg0, %5 : index
      func.call @_sparse_shift_down_0_1_2_3_index_coo_0_f64(%arg0, %6, %arg2, %arg3, %0) : (index, index, memref<?xindex>, memref<?xf64>, index) -> ()
    }
    %4 = arith.subi %0, %c1 : index
    scf.for %arg4 = %c0 to %4 step %c1 {
      %5 = arith.subi %0, %arg4 : index
      %6 = arith.addi %arg0, %5 : index
      %7 = arith.subi %6, %c1 : index
      %8 = arith.muli %arg0, %c4 : index
      %9 = arith.muli %7, %c4 : index
      %10 = memref.load %arg2[%8] : memref<?xindex>
      %11 = memref.load %arg2[%9] : memref<?xindex>
      memref.store %11, %arg2[%8] : memref<?xindex>
      memref.store %10, %arg2[%9] : memref<?xindex>
      %12 = arith.addi %8, %c1 : index
      %13 = arith.addi %9, %c1 : index
      %14 = memref.load %arg2[%12] : memref<?xindex>
      %15 = memref.load %arg2[%13] : memref<?xindex>
      memref.store %15, %arg2[%12] : memref<?xindex>
      memref.store %14, %arg2[%13] : memref<?xindex>
      %16 = arith.addi %8, %c2 : index
      %17 = arith.addi %9, %c2 : index
      %18 = memref.load %arg2[%16] : memref<?xindex>
      %19 = memref.load %arg2[%17] : memref<?xindex>
      memref.store %19, %arg2[%16] : memref<?xindex>
      memref.store %18, %arg2[%17] : memref<?xindex>
      %20 = arith.addi %8, %c3 : index
      %21 = arith.addi %9, %c3 : index
      %22 = memref.load %arg2[%20] : memref<?xindex>
      %23 = memref.load %arg2[%21] : memref<?xindex>
      memref.store %23, %arg2[%20] : memref<?xindex>
      memref.store %22, %arg2[%21] : memref<?xindex>
      %24 = memref.load %arg3[%arg0] : memref<?xf64>
      %25 = memref.load %arg3[%7] : memref<?xf64>
      memref.store %25, %arg3[%arg0] : memref<?xf64>
      memref.store %24, %arg3[%7] : memref<?xf64>
      %26 = arith.subi %5, %c1 : index
      func.call @_sparse_shift_down_0_1_2_3_index_coo_0_f64(%arg0, %arg0, %arg2, %arg3, %26) : (index, index, memref<?xindex>, memref<?xf64>, index) -> ()
    }
    return
  }
  func.func private @_sparse_partition_0_1_2_3_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>) -> index {
    %c-1 = arith.constant -1 : index
    %false = arith.constant false
    %true = arith.constant true
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg0, %arg1 : index
    %1 = arith.shrui %0, %c1 : index
    %2 = arith.subi %arg1, %c1 : index
    %3 = arith.subi %arg1, %arg0 : index
    %4 = arith.cmpi ult, %3, %c1000 : index
    scf.if %4 {
      %6 = arith.muli %1, %c4 : index
      %7 = arith.muli %arg0, %c4 : index
      %8 = memref.load %arg2[%6] : memref<?xindex>
      %9 = memref.load %arg2[%7] : memref<?xindex>
      %10 = arith.cmpi ne, %8, %9 : index
      %11 = scf.if %10 -> (i1) {
        %18 = arith.cmpi ult, %8, %9 : index
        scf.yield %18 : i1
      } else {
        %18 = arith.addi %6, %c1 : index
        %19 = arith.addi %7, %c1 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        %21 = memref.load %arg2[%19] : memref<?xindex>
        %22 = arith.cmpi ne, %20, %21 : index
        %23 = scf.if %22 -> (i1) {
          %24 = arith.cmpi ult, %20, %21 : index
          scf.yield %24 : i1
        } else {
          %24 = arith.addi %6, %c2 : index
          %25 = arith.addi %7, %c2 : index
          %26 = memref.load %arg2[%24] : memref<?xindex>
          %27 = memref.load %arg2[%25] : memref<?xindex>
          %28 = arith.cmpi ne, %26, %27 : index
          %29 = scf.if %28 -> (i1) {
            %30 = arith.cmpi ult, %26, %27 : index
            scf.yield %30 : i1
          } else {
            %30 = arith.addi %6, %c3 : index
            %31 = arith.addi %7, %c3 : index
            %32 = memref.load %arg2[%30] : memref<?xindex>
            %33 = memref.load %arg2[%31] : memref<?xindex>
            %34 = arith.cmpi ult, %32, %33 : index
            scf.yield %34 : i1
          }
          scf.yield %29 : i1
        }
        scf.yield %23 : i1
      }
      scf.if %11 {
        %18 = arith.muli %1, %c4 : index
        %19 = arith.muli %arg0, %c4 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        %21 = memref.load %arg2[%19] : memref<?xindex>
        memref.store %21, %arg2[%18] : memref<?xindex>
        memref.store %20, %arg2[%19] : memref<?xindex>
        %22 = arith.addi %18, %c1 : index
        %23 = arith.addi %19, %c1 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        memref.store %25, %arg2[%22] : memref<?xindex>
        memref.store %24, %arg2[%23] : memref<?xindex>
        %26 = arith.addi %18, %c2 : index
        %27 = arith.addi %19, %c2 : index
        %28 = memref.load %arg2[%26] : memref<?xindex>
        %29 = memref.load %arg2[%27] : memref<?xindex>
        memref.store %29, %arg2[%26] : memref<?xindex>
        memref.store %28, %arg2[%27] : memref<?xindex>
        %30 = arith.addi %18, %c3 : index
        %31 = arith.addi %19, %c3 : index
        %32 = memref.load %arg2[%30] : memref<?xindex>
        %33 = memref.load %arg2[%31] : memref<?xindex>
        memref.store %33, %arg2[%30] : memref<?xindex>
        memref.store %32, %arg2[%31] : memref<?xindex>
        %34 = memref.load %arg3[%1] : memref<?xf64>
        %35 = memref.load %arg3[%arg0] : memref<?xf64>
        memref.store %35, %arg3[%1] : memref<?xf64>
        memref.store %34, %arg3[%arg0] : memref<?xf64>
      }
      %12 = arith.muli %2, %c4 : index
      %13 = arith.muli %1, %c4 : index
      %14 = memref.load %arg2[%12] : memref<?xindex>
      %15 = memref.load %arg2[%13] : memref<?xindex>
      %16 = arith.cmpi ne, %14, %15 : index
      %17 = scf.if %16 -> (i1) {
        %18 = arith.cmpi ult, %14, %15 : index
        scf.yield %18 : i1
      } else {
        %18 = arith.addi %12, %c1 : index
        %19 = arith.addi %13, %c1 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        %21 = memref.load %arg2[%19] : memref<?xindex>
        %22 = arith.cmpi ne, %20, %21 : index
        %23 = scf.if %22 -> (i1) {
          %24 = arith.cmpi ult, %20, %21 : index
          scf.yield %24 : i1
        } else {
          %24 = arith.addi %12, %c2 : index
          %25 = arith.addi %13, %c2 : index
          %26 = memref.load %arg2[%24] : memref<?xindex>
          %27 = memref.load %arg2[%25] : memref<?xindex>
          %28 = arith.cmpi ne, %26, %27 : index
          %29 = scf.if %28 -> (i1) {
            %30 = arith.cmpi ult, %26, %27 : index
            scf.yield %30 : i1
          } else {
            %30 = arith.addi %12, %c3 : index
            %31 = arith.addi %13, %c3 : index
            %32 = memref.load %arg2[%30] : memref<?xindex>
            %33 = memref.load %arg2[%31] : memref<?xindex>
            %34 = arith.cmpi ult, %32, %33 : index
            scf.yield %34 : i1
          }
          scf.yield %29 : i1
        }
        scf.yield %23 : i1
      }
      scf.if %17 {
        %18 = arith.muli %2, %c4 : index
        %19 = arith.muli %1, %c4 : index
        %20 = memref.load %arg2[%18] : memref<?xindex>
        %21 = memref.load %arg2[%19] : memref<?xindex>
        memref.store %21, %arg2[%18] : memref<?xindex>
        memref.store %20, %arg2[%19] : memref<?xindex>
        %22 = arith.addi %18, %c1 : index
        %23 = arith.addi %19, %c1 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        memref.store %25, %arg2[%22] : memref<?xindex>
        memref.store %24, %arg2[%23] : memref<?xindex>
        %26 = arith.addi %18, %c2 : index
        %27 = arith.addi %19, %c2 : index
        %28 = memref.load %arg2[%26] : memref<?xindex>
        %29 = memref.load %arg2[%27] : memref<?xindex>
        memref.store %29, %arg2[%26] : memref<?xindex>
        memref.store %28, %arg2[%27] : memref<?xindex>
        %30 = arith.addi %18, %c3 : index
        %31 = arith.addi %19, %c3 : index
        %32 = memref.load %arg2[%30] : memref<?xindex>
        %33 = memref.load %arg2[%31] : memref<?xindex>
        memref.store %33, %arg2[%30] : memref<?xindex>
        memref.store %32, %arg2[%31] : memref<?xindex>
        %34 = memref.load %arg3[%2] : memref<?xf64>
        %35 = memref.load %arg3[%1] : memref<?xf64>
        memref.store %35, %arg3[%2] : memref<?xf64>
        memref.store %34, %arg3[%1] : memref<?xf64>
        %36 = arith.muli %1, %c4 : index
        %37 = arith.muli %arg0, %c4 : index
        %38 = memref.load %arg2[%36] : memref<?xindex>
        %39 = memref.load %arg2[%37] : memref<?xindex>
        %40 = arith.cmpi ne, %38, %39 : index
        %41 = scf.if %40 -> (i1) {
          %42 = arith.cmpi ult, %38, %39 : index
          scf.yield %42 : i1
        } else {
          %42 = arith.addi %36, %c1 : index
          %43 = arith.addi %37, %c1 : index
          %44 = memref.load %arg2[%42] : memref<?xindex>
          %45 = memref.load %arg2[%43] : memref<?xindex>
          %46 = arith.cmpi ne, %44, %45 : index
          %47 = scf.if %46 -> (i1) {
            %48 = arith.cmpi ult, %44, %45 : index
            scf.yield %48 : i1
          } else {
            %48 = arith.addi %36, %c2 : index
            %49 = arith.addi %37, %c2 : index
            %50 = memref.load %arg2[%48] : memref<?xindex>
            %51 = memref.load %arg2[%49] : memref<?xindex>
            %52 = arith.cmpi ne, %50, %51 : index
            %53 = scf.if %52 -> (i1) {
              %54 = arith.cmpi ult, %50, %51 : index
              scf.yield %54 : i1
            } else {
              %54 = arith.addi %36, %c3 : index
              %55 = arith.addi %37, %c3 : index
              %56 = memref.load %arg2[%54] : memref<?xindex>
              %57 = memref.load %arg2[%55] : memref<?xindex>
              %58 = arith.cmpi ult, %56, %57 : index
              scf.yield %58 : i1
            }
            scf.yield %53 : i1
          }
          scf.yield %47 : i1
        }
        scf.if %41 {
          %42 = arith.muli %1, %c4 : index
          %43 = arith.muli %arg0, %c4 : index
          %44 = memref.load %arg2[%42] : memref<?xindex>
          %45 = memref.load %arg2[%43] : memref<?xindex>
          memref.store %45, %arg2[%42] : memref<?xindex>
          memref.store %44, %arg2[%43] : memref<?xindex>
          %46 = arith.addi %42, %c1 : index
          %47 = arith.addi %43, %c1 : index
          %48 = memref.load %arg2[%46] : memref<?xindex>
          %49 = memref.load %arg2[%47] : memref<?xindex>
          memref.store %49, %arg2[%46] : memref<?xindex>
          memref.store %48, %arg2[%47] : memref<?xindex>
          %50 = arith.addi %42, %c2 : index
          %51 = arith.addi %43, %c2 : index
          %52 = memref.load %arg2[%50] : memref<?xindex>
          %53 = memref.load %arg2[%51] : memref<?xindex>
          memref.store %53, %arg2[%50] : memref<?xindex>
          memref.store %52, %arg2[%51] : memref<?xindex>
          %54 = arith.addi %42, %c3 : index
          %55 = arith.addi %43, %c3 : index
          %56 = memref.load %arg2[%54] : memref<?xindex>
          %57 = memref.load %arg2[%55] : memref<?xindex>
          memref.store %57, %arg2[%54] : memref<?xindex>
          memref.store %56, %arg2[%55] : memref<?xindex>
          %58 = memref.load %arg3[%1] : memref<?xf64>
          %59 = memref.load %arg3[%arg0] : memref<?xf64>
          memref.store %59, %arg3[%1] : memref<?xf64>
          memref.store %58, %arg3[%arg0] : memref<?xf64>
        }
      }
    } else {
      %6 = arith.addi %arg0, %arg1 : index
      %7 = arith.shrui %6, %c1 : index
      %8 = arith.addi %1, %arg1 : index
      %9 = arith.shrui %8, %c1 : index
      %10 = arith.muli %7, %c4 : index
      %11 = arith.muli %arg0, %c4 : index
      %12 = memref.load %arg2[%10] : memref<?xindex>
      %13 = memref.load %arg2[%11] : memref<?xindex>
      %14 = arith.cmpi ne, %12, %13 : index
      %15 = scf.if %14 -> (i1) {
        %34 = arith.cmpi ult, %12, %13 : index
        scf.yield %34 : i1
      } else {
        %34 = arith.addi %10, %c1 : index
        %35 = arith.addi %11, %c1 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        %38 = arith.cmpi ne, %36, %37 : index
        %39 = scf.if %38 -> (i1) {
          %40 = arith.cmpi ult, %36, %37 : index
          scf.yield %40 : i1
        } else {
          %40 = arith.addi %10, %c2 : index
          %41 = arith.addi %11, %c2 : index
          %42 = memref.load %arg2[%40] : memref<?xindex>
          %43 = memref.load %arg2[%41] : memref<?xindex>
          %44 = arith.cmpi ne, %42, %43 : index
          %45 = scf.if %44 -> (i1) {
            %46 = arith.cmpi ult, %42, %43 : index
            scf.yield %46 : i1
          } else {
            %46 = arith.addi %10, %c3 : index
            %47 = arith.addi %11, %c3 : index
            %48 = memref.load %arg2[%46] : memref<?xindex>
            %49 = memref.load %arg2[%47] : memref<?xindex>
            %50 = arith.cmpi ult, %48, %49 : index
            scf.yield %50 : i1
          }
          scf.yield %45 : i1
        }
        scf.yield %39 : i1
      }
      scf.if %15 {
        %34 = arith.muli %7, %c4 : index
        %35 = arith.muli %arg0, %c4 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        memref.store %37, %arg2[%34] : memref<?xindex>
        memref.store %36, %arg2[%35] : memref<?xindex>
        %38 = arith.addi %34, %c1 : index
        %39 = arith.addi %35, %c1 : index
        %40 = memref.load %arg2[%38] : memref<?xindex>
        %41 = memref.load %arg2[%39] : memref<?xindex>
        memref.store %41, %arg2[%38] : memref<?xindex>
        memref.store %40, %arg2[%39] : memref<?xindex>
        %42 = arith.addi %34, %c2 : index
        %43 = arith.addi %35, %c2 : index
        %44 = memref.load %arg2[%42] : memref<?xindex>
        %45 = memref.load %arg2[%43] : memref<?xindex>
        memref.store %45, %arg2[%42] : memref<?xindex>
        memref.store %44, %arg2[%43] : memref<?xindex>
        %46 = arith.addi %34, %c3 : index
        %47 = arith.addi %35, %c3 : index
        %48 = memref.load %arg2[%46] : memref<?xindex>
        %49 = memref.load %arg2[%47] : memref<?xindex>
        memref.store %49, %arg2[%46] : memref<?xindex>
        memref.store %48, %arg2[%47] : memref<?xindex>
        %50 = memref.load %arg3[%7] : memref<?xf64>
        %51 = memref.load %arg3[%arg0] : memref<?xf64>
        memref.store %51, %arg3[%7] : memref<?xf64>
        memref.store %50, %arg3[%arg0] : memref<?xf64>
      }
      %16 = arith.muli %1, %c4 : index
      %17 = arith.muli %7, %c4 : index
      %18 = memref.load %arg2[%16] : memref<?xindex>
      %19 = memref.load %arg2[%17] : memref<?xindex>
      %20 = arith.cmpi ne, %18, %19 : index
      %21 = scf.if %20 -> (i1) {
        %34 = arith.cmpi ult, %18, %19 : index
        scf.yield %34 : i1
      } else {
        %34 = arith.addi %16, %c1 : index
        %35 = arith.addi %17, %c1 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        %38 = arith.cmpi ne, %36, %37 : index
        %39 = scf.if %38 -> (i1) {
          %40 = arith.cmpi ult, %36, %37 : index
          scf.yield %40 : i1
        } else {
          %40 = arith.addi %16, %c2 : index
          %41 = arith.addi %17, %c2 : index
          %42 = memref.load %arg2[%40] : memref<?xindex>
          %43 = memref.load %arg2[%41] : memref<?xindex>
          %44 = arith.cmpi ne, %42, %43 : index
          %45 = scf.if %44 -> (i1) {
            %46 = arith.cmpi ult, %42, %43 : index
            scf.yield %46 : i1
          } else {
            %46 = arith.addi %16, %c3 : index
            %47 = arith.addi %17, %c3 : index
            %48 = memref.load %arg2[%46] : memref<?xindex>
            %49 = memref.load %arg2[%47] : memref<?xindex>
            %50 = arith.cmpi ult, %48, %49 : index
            scf.yield %50 : i1
          }
          scf.yield %45 : i1
        }
        scf.yield %39 : i1
      }
      scf.if %21 {
        %34 = arith.muli %1, %c4 : index
        %35 = arith.muli %7, %c4 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        memref.store %37, %arg2[%34] : memref<?xindex>
        memref.store %36, %arg2[%35] : memref<?xindex>
        %38 = arith.addi %34, %c1 : index
        %39 = arith.addi %35, %c1 : index
        %40 = memref.load %arg2[%38] : memref<?xindex>
        %41 = memref.load %arg2[%39] : memref<?xindex>
        memref.store %41, %arg2[%38] : memref<?xindex>
        memref.store %40, %arg2[%39] : memref<?xindex>
        %42 = arith.addi %34, %c2 : index
        %43 = arith.addi %35, %c2 : index
        %44 = memref.load %arg2[%42] : memref<?xindex>
        %45 = memref.load %arg2[%43] : memref<?xindex>
        memref.store %45, %arg2[%42] : memref<?xindex>
        memref.store %44, %arg2[%43] : memref<?xindex>
        %46 = arith.addi %34, %c3 : index
        %47 = arith.addi %35, %c3 : index
        %48 = memref.load %arg2[%46] : memref<?xindex>
        %49 = memref.load %arg2[%47] : memref<?xindex>
        memref.store %49, %arg2[%46] : memref<?xindex>
        memref.store %48, %arg2[%47] : memref<?xindex>
        %50 = memref.load %arg3[%1] : memref<?xf64>
        %51 = memref.load %arg3[%7] : memref<?xf64>
        memref.store %51, %arg3[%1] : memref<?xf64>
        memref.store %50, %arg3[%7] : memref<?xf64>
        %52 = arith.muli %7, %c4 : index
        %53 = arith.muli %arg0, %c4 : index
        %54 = memref.load %arg2[%52] : memref<?xindex>
        %55 = memref.load %arg2[%53] : memref<?xindex>
        %56 = arith.cmpi ne, %54, %55 : index
        %57 = scf.if %56 -> (i1) {
          %58 = arith.cmpi ult, %54, %55 : index
          scf.yield %58 : i1
        } else {
          %58 = arith.addi %52, %c1 : index
          %59 = arith.addi %53, %c1 : index
          %60 = memref.load %arg2[%58] : memref<?xindex>
          %61 = memref.load %arg2[%59] : memref<?xindex>
          %62 = arith.cmpi ne, %60, %61 : index
          %63 = scf.if %62 -> (i1) {
            %64 = arith.cmpi ult, %60, %61 : index
            scf.yield %64 : i1
          } else {
            %64 = arith.addi %52, %c2 : index
            %65 = arith.addi %53, %c2 : index
            %66 = memref.load %arg2[%64] : memref<?xindex>
            %67 = memref.load %arg2[%65] : memref<?xindex>
            %68 = arith.cmpi ne, %66, %67 : index
            %69 = scf.if %68 -> (i1) {
              %70 = arith.cmpi ult, %66, %67 : index
              scf.yield %70 : i1
            } else {
              %70 = arith.addi %52, %c3 : index
              %71 = arith.addi %53, %c3 : index
              %72 = memref.load %arg2[%70] : memref<?xindex>
              %73 = memref.load %arg2[%71] : memref<?xindex>
              %74 = arith.cmpi ult, %72, %73 : index
              scf.yield %74 : i1
            }
            scf.yield %69 : i1
          }
          scf.yield %63 : i1
        }
        scf.if %57 {
          %58 = arith.muli %7, %c4 : index
          %59 = arith.muli %arg0, %c4 : index
          %60 = memref.load %arg2[%58] : memref<?xindex>
          %61 = memref.load %arg2[%59] : memref<?xindex>
          memref.store %61, %arg2[%58] : memref<?xindex>
          memref.store %60, %arg2[%59] : memref<?xindex>
          %62 = arith.addi %58, %c1 : index
          %63 = arith.addi %59, %c1 : index
          %64 = memref.load %arg2[%62] : memref<?xindex>
          %65 = memref.load %arg2[%63] : memref<?xindex>
          memref.store %65, %arg2[%62] : memref<?xindex>
          memref.store %64, %arg2[%63] : memref<?xindex>
          %66 = arith.addi %58, %c2 : index
          %67 = arith.addi %59, %c2 : index
          %68 = memref.load %arg2[%66] : memref<?xindex>
          %69 = memref.load %arg2[%67] : memref<?xindex>
          memref.store %69, %arg2[%66] : memref<?xindex>
          memref.store %68, %arg2[%67] : memref<?xindex>
          %70 = arith.addi %58, %c3 : index
          %71 = arith.addi %59, %c3 : index
          %72 = memref.load %arg2[%70] : memref<?xindex>
          %73 = memref.load %arg2[%71] : memref<?xindex>
          memref.store %73, %arg2[%70] : memref<?xindex>
          memref.store %72, %arg2[%71] : memref<?xindex>
          %74 = memref.load %arg3[%7] : memref<?xf64>
          %75 = memref.load %arg3[%arg0] : memref<?xf64>
          memref.store %75, %arg3[%7] : memref<?xf64>
          memref.store %74, %arg3[%arg0] : memref<?xf64>
        }
      }
      %22 = arith.muli %9, %c4 : index
      %23 = arith.muli %1, %c4 : index
      %24 = memref.load %arg2[%22] : memref<?xindex>
      %25 = memref.load %arg2[%23] : memref<?xindex>
      %26 = arith.cmpi ne, %24, %25 : index
      %27 = scf.if %26 -> (i1) {
        %34 = arith.cmpi ult, %24, %25 : index
        scf.yield %34 : i1
      } else {
        %34 = arith.addi %22, %c1 : index
        %35 = arith.addi %23, %c1 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        %38 = arith.cmpi ne, %36, %37 : index
        %39 = scf.if %38 -> (i1) {
          %40 = arith.cmpi ult, %36, %37 : index
          scf.yield %40 : i1
        } else {
          %40 = arith.addi %22, %c2 : index
          %41 = arith.addi %23, %c2 : index
          %42 = memref.load %arg2[%40] : memref<?xindex>
          %43 = memref.load %arg2[%41] : memref<?xindex>
          %44 = arith.cmpi ne, %42, %43 : index
          %45 = scf.if %44 -> (i1) {
            %46 = arith.cmpi ult, %42, %43 : index
            scf.yield %46 : i1
          } else {
            %46 = arith.addi %22, %c3 : index
            %47 = arith.addi %23, %c3 : index
            %48 = memref.load %arg2[%46] : memref<?xindex>
            %49 = memref.load %arg2[%47] : memref<?xindex>
            %50 = arith.cmpi ult, %48, %49 : index
            scf.yield %50 : i1
          }
          scf.yield %45 : i1
        }
        scf.yield %39 : i1
      }
      scf.if %27 {
        %34 = arith.muli %9, %c4 : index
        %35 = arith.muli %1, %c4 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        memref.store %37, %arg2[%34] : memref<?xindex>
        memref.store %36, %arg2[%35] : memref<?xindex>
        %38 = arith.addi %34, %c1 : index
        %39 = arith.addi %35, %c1 : index
        %40 = memref.load %arg2[%38] : memref<?xindex>
        %41 = memref.load %arg2[%39] : memref<?xindex>
        memref.store %41, %arg2[%38] : memref<?xindex>
        memref.store %40, %arg2[%39] : memref<?xindex>
        %42 = arith.addi %34, %c2 : index
        %43 = arith.addi %35, %c2 : index
        %44 = memref.load %arg2[%42] : memref<?xindex>
        %45 = memref.load %arg2[%43] : memref<?xindex>
        memref.store %45, %arg2[%42] : memref<?xindex>
        memref.store %44, %arg2[%43] : memref<?xindex>
        %46 = arith.addi %34, %c3 : index
        %47 = arith.addi %35, %c3 : index
        %48 = memref.load %arg2[%46] : memref<?xindex>
        %49 = memref.load %arg2[%47] : memref<?xindex>
        memref.store %49, %arg2[%46] : memref<?xindex>
        memref.store %48, %arg2[%47] : memref<?xindex>
        %50 = memref.load %arg3[%9] : memref<?xf64>
        %51 = memref.load %arg3[%1] : memref<?xf64>
        memref.store %51, %arg3[%9] : memref<?xf64>
        memref.store %50, %arg3[%1] : memref<?xf64>
        %52 = arith.muli %1, %c4 : index
        %53 = arith.muli %7, %c4 : index
        %54 = memref.load %arg2[%52] : memref<?xindex>
        %55 = memref.load %arg2[%53] : memref<?xindex>
        %56 = arith.cmpi ne, %54, %55 : index
        %57 = scf.if %56 -> (i1) {
          %58 = arith.cmpi ult, %54, %55 : index
          scf.yield %58 : i1
        } else {
          %58 = arith.addi %52, %c1 : index
          %59 = arith.addi %53, %c1 : index
          %60 = memref.load %arg2[%58] : memref<?xindex>
          %61 = memref.load %arg2[%59] : memref<?xindex>
          %62 = arith.cmpi ne, %60, %61 : index
          %63 = scf.if %62 -> (i1) {
            %64 = arith.cmpi ult, %60, %61 : index
            scf.yield %64 : i1
          } else {
            %64 = arith.addi %52, %c2 : index
            %65 = arith.addi %53, %c2 : index
            %66 = memref.load %arg2[%64] : memref<?xindex>
            %67 = memref.load %arg2[%65] : memref<?xindex>
            %68 = arith.cmpi ne, %66, %67 : index
            %69 = scf.if %68 -> (i1) {
              %70 = arith.cmpi ult, %66, %67 : index
              scf.yield %70 : i1
            } else {
              %70 = arith.addi %52, %c3 : index
              %71 = arith.addi %53, %c3 : index
              %72 = memref.load %arg2[%70] : memref<?xindex>
              %73 = memref.load %arg2[%71] : memref<?xindex>
              %74 = arith.cmpi ult, %72, %73 : index
              scf.yield %74 : i1
            }
            scf.yield %69 : i1
          }
          scf.yield %63 : i1
        }
        scf.if %57 {
          %58 = arith.muli %1, %c4 : index
          %59 = arith.muli %7, %c4 : index
          %60 = memref.load %arg2[%58] : memref<?xindex>
          %61 = memref.load %arg2[%59] : memref<?xindex>
          memref.store %61, %arg2[%58] : memref<?xindex>
          memref.store %60, %arg2[%59] : memref<?xindex>
          %62 = arith.addi %58, %c1 : index
          %63 = arith.addi %59, %c1 : index
          %64 = memref.load %arg2[%62] : memref<?xindex>
          %65 = memref.load %arg2[%63] : memref<?xindex>
          memref.store %65, %arg2[%62] : memref<?xindex>
          memref.store %64, %arg2[%63] : memref<?xindex>
          %66 = arith.addi %58, %c2 : index
          %67 = arith.addi %59, %c2 : index
          %68 = memref.load %arg2[%66] : memref<?xindex>
          %69 = memref.load %arg2[%67] : memref<?xindex>
          memref.store %69, %arg2[%66] : memref<?xindex>
          memref.store %68, %arg2[%67] : memref<?xindex>
          %70 = arith.addi %58, %c3 : index
          %71 = arith.addi %59, %c3 : index
          %72 = memref.load %arg2[%70] : memref<?xindex>
          %73 = memref.load %arg2[%71] : memref<?xindex>
          memref.store %73, %arg2[%70] : memref<?xindex>
          memref.store %72, %arg2[%71] : memref<?xindex>
          %74 = memref.load %arg3[%1] : memref<?xf64>
          %75 = memref.load %arg3[%7] : memref<?xf64>
          memref.store %75, %arg3[%1] : memref<?xf64>
          memref.store %74, %arg3[%7] : memref<?xf64>
          %76 = arith.muli %7, %c4 : index
          %77 = arith.muli %arg0, %c4 : index
          %78 = memref.load %arg2[%76] : memref<?xindex>
          %79 = memref.load %arg2[%77] : memref<?xindex>
          %80 = arith.cmpi ne, %78, %79 : index
          %81 = scf.if %80 -> (i1) {
            %82 = arith.cmpi ult, %78, %79 : index
            scf.yield %82 : i1
          } else {
            %82 = arith.addi %76, %c1 : index
            %83 = arith.addi %77, %c1 : index
            %84 = memref.load %arg2[%82] : memref<?xindex>
            %85 = memref.load %arg2[%83] : memref<?xindex>
            %86 = arith.cmpi ne, %84, %85 : index
            %87 = scf.if %86 -> (i1) {
              %88 = arith.cmpi ult, %84, %85 : index
              scf.yield %88 : i1
            } else {
              %88 = arith.addi %76, %c2 : index
              %89 = arith.addi %77, %c2 : index
              %90 = memref.load %arg2[%88] : memref<?xindex>
              %91 = memref.load %arg2[%89] : memref<?xindex>
              %92 = arith.cmpi ne, %90, %91 : index
              %93 = scf.if %92 -> (i1) {
                %94 = arith.cmpi ult, %90, %91 : index
                scf.yield %94 : i1
              } else {
                %94 = arith.addi %76, %c3 : index
                %95 = arith.addi %77, %c3 : index
                %96 = memref.load %arg2[%94] : memref<?xindex>
                %97 = memref.load %arg2[%95] : memref<?xindex>
                %98 = arith.cmpi ult, %96, %97 : index
                scf.yield %98 : i1
              }
              scf.yield %93 : i1
            }
            scf.yield %87 : i1
          }
          scf.if %81 {
            %82 = arith.muli %7, %c4 : index
            %83 = arith.muli %arg0, %c4 : index
            %84 = memref.load %arg2[%82] : memref<?xindex>
            %85 = memref.load %arg2[%83] : memref<?xindex>
            memref.store %85, %arg2[%82] : memref<?xindex>
            memref.store %84, %arg2[%83] : memref<?xindex>
            %86 = arith.addi %82, %c1 : index
            %87 = arith.addi %83, %c1 : index
            %88 = memref.load %arg2[%86] : memref<?xindex>
            %89 = memref.load %arg2[%87] : memref<?xindex>
            memref.store %89, %arg2[%86] : memref<?xindex>
            memref.store %88, %arg2[%87] : memref<?xindex>
            %90 = arith.addi %82, %c2 : index
            %91 = arith.addi %83, %c2 : index
            %92 = memref.load %arg2[%90] : memref<?xindex>
            %93 = memref.load %arg2[%91] : memref<?xindex>
            memref.store %93, %arg2[%90] : memref<?xindex>
            memref.store %92, %arg2[%91] : memref<?xindex>
            %94 = arith.addi %82, %c3 : index
            %95 = arith.addi %83, %c3 : index
            %96 = memref.load %arg2[%94] : memref<?xindex>
            %97 = memref.load %arg2[%95] : memref<?xindex>
            memref.store %97, %arg2[%94] : memref<?xindex>
            memref.store %96, %arg2[%95] : memref<?xindex>
            %98 = memref.load %arg3[%7] : memref<?xf64>
            %99 = memref.load %arg3[%arg0] : memref<?xf64>
            memref.store %99, %arg3[%7] : memref<?xf64>
            memref.store %98, %arg3[%arg0] : memref<?xf64>
          }
        }
      }
      %28 = arith.muli %2, %c4 : index
      %29 = arith.muli %9, %c4 : index
      %30 = memref.load %arg2[%28] : memref<?xindex>
      %31 = memref.load %arg2[%29] : memref<?xindex>
      %32 = arith.cmpi ne, %30, %31 : index
      %33 = scf.if %32 -> (i1) {
        %34 = arith.cmpi ult, %30, %31 : index
        scf.yield %34 : i1
      } else {
        %34 = arith.addi %28, %c1 : index
        %35 = arith.addi %29, %c1 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        %38 = arith.cmpi ne, %36, %37 : index
        %39 = scf.if %38 -> (i1) {
          %40 = arith.cmpi ult, %36, %37 : index
          scf.yield %40 : i1
        } else {
          %40 = arith.addi %28, %c2 : index
          %41 = arith.addi %29, %c2 : index
          %42 = memref.load %arg2[%40] : memref<?xindex>
          %43 = memref.load %arg2[%41] : memref<?xindex>
          %44 = arith.cmpi ne, %42, %43 : index
          %45 = scf.if %44 -> (i1) {
            %46 = arith.cmpi ult, %42, %43 : index
            scf.yield %46 : i1
          } else {
            %46 = arith.addi %28, %c3 : index
            %47 = arith.addi %29, %c3 : index
            %48 = memref.load %arg2[%46] : memref<?xindex>
            %49 = memref.load %arg2[%47] : memref<?xindex>
            %50 = arith.cmpi ult, %48, %49 : index
            scf.yield %50 : i1
          }
          scf.yield %45 : i1
        }
        scf.yield %39 : i1
      }
      scf.if %33 {
        %34 = arith.muli %2, %c4 : index
        %35 = arith.muli %9, %c4 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        memref.store %37, %arg2[%34] : memref<?xindex>
        memref.store %36, %arg2[%35] : memref<?xindex>
        %38 = arith.addi %34, %c1 : index
        %39 = arith.addi %35, %c1 : index
        %40 = memref.load %arg2[%38] : memref<?xindex>
        %41 = memref.load %arg2[%39] : memref<?xindex>
        memref.store %41, %arg2[%38] : memref<?xindex>
        memref.store %40, %arg2[%39] : memref<?xindex>
        %42 = arith.addi %34, %c2 : index
        %43 = arith.addi %35, %c2 : index
        %44 = memref.load %arg2[%42] : memref<?xindex>
        %45 = memref.load %arg2[%43] : memref<?xindex>
        memref.store %45, %arg2[%42] : memref<?xindex>
        memref.store %44, %arg2[%43] : memref<?xindex>
        %46 = arith.addi %34, %c3 : index
        %47 = arith.addi %35, %c3 : index
        %48 = memref.load %arg2[%46] : memref<?xindex>
        %49 = memref.load %arg2[%47] : memref<?xindex>
        memref.store %49, %arg2[%46] : memref<?xindex>
        memref.store %48, %arg2[%47] : memref<?xindex>
        %50 = memref.load %arg3[%2] : memref<?xf64>
        %51 = memref.load %arg3[%9] : memref<?xf64>
        memref.store %51, %arg3[%2] : memref<?xf64>
        memref.store %50, %arg3[%9] : memref<?xf64>
        %52 = arith.muli %9, %c4 : index
        %53 = arith.muli %1, %c4 : index
        %54 = memref.load %arg2[%52] : memref<?xindex>
        %55 = memref.load %arg2[%53] : memref<?xindex>
        %56 = arith.cmpi ne, %54, %55 : index
        %57 = scf.if %56 -> (i1) {
          %58 = arith.cmpi ult, %54, %55 : index
          scf.yield %58 : i1
        } else {
          %58 = arith.addi %52, %c1 : index
          %59 = arith.addi %53, %c1 : index
          %60 = memref.load %arg2[%58] : memref<?xindex>
          %61 = memref.load %arg2[%59] : memref<?xindex>
          %62 = arith.cmpi ne, %60, %61 : index
          %63 = scf.if %62 -> (i1) {
            %64 = arith.cmpi ult, %60, %61 : index
            scf.yield %64 : i1
          } else {
            %64 = arith.addi %52, %c2 : index
            %65 = arith.addi %53, %c2 : index
            %66 = memref.load %arg2[%64] : memref<?xindex>
            %67 = memref.load %arg2[%65] : memref<?xindex>
            %68 = arith.cmpi ne, %66, %67 : index
            %69 = scf.if %68 -> (i1) {
              %70 = arith.cmpi ult, %66, %67 : index
              scf.yield %70 : i1
            } else {
              %70 = arith.addi %52, %c3 : index
              %71 = arith.addi %53, %c3 : index
              %72 = memref.load %arg2[%70] : memref<?xindex>
              %73 = memref.load %arg2[%71] : memref<?xindex>
              %74 = arith.cmpi ult, %72, %73 : index
              scf.yield %74 : i1
            }
            scf.yield %69 : i1
          }
          scf.yield %63 : i1
        }
        scf.if %57 {
          %58 = arith.muli %9, %c4 : index
          %59 = arith.muli %1, %c4 : index
          %60 = memref.load %arg2[%58] : memref<?xindex>
          %61 = memref.load %arg2[%59] : memref<?xindex>
          memref.store %61, %arg2[%58] : memref<?xindex>
          memref.store %60, %arg2[%59] : memref<?xindex>
          %62 = arith.addi %58, %c1 : index
          %63 = arith.addi %59, %c1 : index
          %64 = memref.load %arg2[%62] : memref<?xindex>
          %65 = memref.load %arg2[%63] : memref<?xindex>
          memref.store %65, %arg2[%62] : memref<?xindex>
          memref.store %64, %arg2[%63] : memref<?xindex>
          %66 = arith.addi %58, %c2 : index
          %67 = arith.addi %59, %c2 : index
          %68 = memref.load %arg2[%66] : memref<?xindex>
          %69 = memref.load %arg2[%67] : memref<?xindex>
          memref.store %69, %arg2[%66] : memref<?xindex>
          memref.store %68, %arg2[%67] : memref<?xindex>
          %70 = arith.addi %58, %c3 : index
          %71 = arith.addi %59, %c3 : index
          %72 = memref.load %arg2[%70] : memref<?xindex>
          %73 = memref.load %arg2[%71] : memref<?xindex>
          memref.store %73, %arg2[%70] : memref<?xindex>
          memref.store %72, %arg2[%71] : memref<?xindex>
          %74 = memref.load %arg3[%9] : memref<?xf64>
          %75 = memref.load %arg3[%1] : memref<?xf64>
          memref.store %75, %arg3[%9] : memref<?xf64>
          memref.store %74, %arg3[%1] : memref<?xf64>
          %76 = arith.muli %1, %c4 : index
          %77 = arith.muli %7, %c4 : index
          %78 = memref.load %arg2[%76] : memref<?xindex>
          %79 = memref.load %arg2[%77] : memref<?xindex>
          %80 = arith.cmpi ne, %78, %79 : index
          %81 = scf.if %80 -> (i1) {
            %82 = arith.cmpi ult, %78, %79 : index
            scf.yield %82 : i1
          } else {
            %82 = arith.addi %76, %c1 : index
            %83 = arith.addi %77, %c1 : index
            %84 = memref.load %arg2[%82] : memref<?xindex>
            %85 = memref.load %arg2[%83] : memref<?xindex>
            %86 = arith.cmpi ne, %84, %85 : index
            %87 = scf.if %86 -> (i1) {
              %88 = arith.cmpi ult, %84, %85 : index
              scf.yield %88 : i1
            } else {
              %88 = arith.addi %76, %c2 : index
              %89 = arith.addi %77, %c2 : index
              %90 = memref.load %arg2[%88] : memref<?xindex>
              %91 = memref.load %arg2[%89] : memref<?xindex>
              %92 = arith.cmpi ne, %90, %91 : index
              %93 = scf.if %92 -> (i1) {
                %94 = arith.cmpi ult, %90, %91 : index
                scf.yield %94 : i1
              } else {
                %94 = arith.addi %76, %c3 : index
                %95 = arith.addi %77, %c3 : index
                %96 = memref.load %arg2[%94] : memref<?xindex>
                %97 = memref.load %arg2[%95] : memref<?xindex>
                %98 = arith.cmpi ult, %96, %97 : index
                scf.yield %98 : i1
              }
              scf.yield %93 : i1
            }
            scf.yield %87 : i1
          }
          scf.if %81 {
            %82 = arith.muli %1, %c4 : index
            %83 = arith.muli %7, %c4 : index
            %84 = memref.load %arg2[%82] : memref<?xindex>
            %85 = memref.load %arg2[%83] : memref<?xindex>
            memref.store %85, %arg2[%82] : memref<?xindex>
            memref.store %84, %arg2[%83] : memref<?xindex>
            %86 = arith.addi %82, %c1 : index
            %87 = arith.addi %83, %c1 : index
            %88 = memref.load %arg2[%86] : memref<?xindex>
            %89 = memref.load %arg2[%87] : memref<?xindex>
            memref.store %89, %arg2[%86] : memref<?xindex>
            memref.store %88, %arg2[%87] : memref<?xindex>
            %90 = arith.addi %82, %c2 : index
            %91 = arith.addi %83, %c2 : index
            %92 = memref.load %arg2[%90] : memref<?xindex>
            %93 = memref.load %arg2[%91] : memref<?xindex>
            memref.store %93, %arg2[%90] : memref<?xindex>
            memref.store %92, %arg2[%91] : memref<?xindex>
            %94 = arith.addi %82, %c3 : index
            %95 = arith.addi %83, %c3 : index
            %96 = memref.load %arg2[%94] : memref<?xindex>
            %97 = memref.load %arg2[%95] : memref<?xindex>
            memref.store %97, %arg2[%94] : memref<?xindex>
            memref.store %96, %arg2[%95] : memref<?xindex>
            %98 = memref.load %arg3[%1] : memref<?xf64>
            %99 = memref.load %arg3[%7] : memref<?xf64>
            memref.store %99, %arg3[%1] : memref<?xf64>
            memref.store %98, %arg3[%7] : memref<?xf64>
            %100 = arith.muli %7, %c4 : index
            %101 = arith.muli %arg0, %c4 : index
            %102 = memref.load %arg2[%100] : memref<?xindex>
            %103 = memref.load %arg2[%101] : memref<?xindex>
            %104 = arith.cmpi ne, %102, %103 : index
            %105 = scf.if %104 -> (i1) {
              %106 = arith.cmpi ult, %102, %103 : index
              scf.yield %106 : i1
            } else {
              %106 = arith.addi %100, %c1 : index
              %107 = arith.addi %101, %c1 : index
              %108 = memref.load %arg2[%106] : memref<?xindex>
              %109 = memref.load %arg2[%107] : memref<?xindex>
              %110 = arith.cmpi ne, %108, %109 : index
              %111 = scf.if %110 -> (i1) {
                %112 = arith.cmpi ult, %108, %109 : index
                scf.yield %112 : i1
              } else {
                %112 = arith.addi %100, %c2 : index
                %113 = arith.addi %101, %c2 : index
                %114 = memref.load %arg2[%112] : memref<?xindex>
                %115 = memref.load %arg2[%113] : memref<?xindex>
                %116 = arith.cmpi ne, %114, %115 : index
                %117 = scf.if %116 -> (i1) {
                  %118 = arith.cmpi ult, %114, %115 : index
                  scf.yield %118 : i1
                } else {
                  %118 = arith.addi %100, %c3 : index
                  %119 = arith.addi %101, %c3 : index
                  %120 = memref.load %arg2[%118] : memref<?xindex>
                  %121 = memref.load %arg2[%119] : memref<?xindex>
                  %122 = arith.cmpi ult, %120, %121 : index
                  scf.yield %122 : i1
                }
                scf.yield %117 : i1
              }
              scf.yield %111 : i1
            }
            scf.if %105 {
              %106 = arith.muli %7, %c4 : index
              %107 = arith.muli %arg0, %c4 : index
              %108 = memref.load %arg2[%106] : memref<?xindex>
              %109 = memref.load %arg2[%107] : memref<?xindex>
              memref.store %109, %arg2[%106] : memref<?xindex>
              memref.store %108, %arg2[%107] : memref<?xindex>
              %110 = arith.addi %106, %c1 : index
              %111 = arith.addi %107, %c1 : index
              %112 = memref.load %arg2[%110] : memref<?xindex>
              %113 = memref.load %arg2[%111] : memref<?xindex>
              memref.store %113, %arg2[%110] : memref<?xindex>
              memref.store %112, %arg2[%111] : memref<?xindex>
              %114 = arith.addi %106, %c2 : index
              %115 = arith.addi %107, %c2 : index
              %116 = memref.load %arg2[%114] : memref<?xindex>
              %117 = memref.load %arg2[%115] : memref<?xindex>
              memref.store %117, %arg2[%114] : memref<?xindex>
              memref.store %116, %arg2[%115] : memref<?xindex>
              %118 = arith.addi %106, %c3 : index
              %119 = arith.addi %107, %c3 : index
              %120 = memref.load %arg2[%118] : memref<?xindex>
              %121 = memref.load %arg2[%119] : memref<?xindex>
              memref.store %121, %arg2[%118] : memref<?xindex>
              memref.store %120, %arg2[%119] : memref<?xindex>
              %122 = memref.load %arg3[%7] : memref<?xf64>
              %123 = memref.load %arg3[%arg0] : memref<?xf64>
              memref.store %123, %arg3[%7] : memref<?xf64>
              memref.store %122, %arg3[%arg0] : memref<?xf64>
            }
          }
        }
      }
    }
    %5:4 = scf.while (%arg4 = %arg0, %arg5 = %2, %arg6 = %1, %arg7 = %true) : (index, index, index, i1) -> (index, index, index, i1) {
      scf.condition(%arg7) %arg4, %arg5, %arg6, %arg7 : index, index, index, i1
    } do {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: i1):
      %6 = scf.while (%arg8 = %arg4) : (index) -> index {
        %22 = arith.muli %arg8, %c4 : index
        %23 = arith.muli %arg6, %c4 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        %26 = arith.cmpi ne, %24, %25 : index
        %27 = scf.if %26 -> (i1) {
          %28 = arith.cmpi ult, %24, %25 : index
          scf.yield %28 : i1
        } else {
          %28 = arith.addi %22, %c1 : index
          %29 = arith.addi %23, %c1 : index
          %30 = memref.load %arg2[%28] : memref<?xindex>
          %31 = memref.load %arg2[%29] : memref<?xindex>
          %32 = arith.cmpi ne, %30, %31 : index
          %33 = scf.if %32 -> (i1) {
            %34 = arith.cmpi ult, %30, %31 : index
            scf.yield %34 : i1
          } else {
            %34 = arith.addi %22, %c2 : index
            %35 = arith.addi %23, %c2 : index
            %36 = memref.load %arg2[%34] : memref<?xindex>
            %37 = memref.load %arg2[%35] : memref<?xindex>
            %38 = arith.cmpi ne, %36, %37 : index
            %39 = scf.if %38 -> (i1) {
              %40 = arith.cmpi ult, %36, %37 : index
              scf.yield %40 : i1
            } else {
              %40 = arith.addi %22, %c3 : index
              %41 = arith.addi %23, %c3 : index
              %42 = memref.load %arg2[%40] : memref<?xindex>
              %43 = memref.load %arg2[%41] : memref<?xindex>
              %44 = arith.cmpi ult, %42, %43 : index
              scf.yield %44 : i1
            }
            scf.yield %39 : i1
          }
          scf.yield %33 : i1
        }
        scf.condition(%27) %arg8 : index
      } do {
      ^bb0(%arg8: index):
        %22 = arith.addi %arg8, %c1 : index
        scf.yield %22 : index
      }
      %7 = arith.muli %6, %c4 : index
      %8 = arith.muli %arg6, %c4 : index
      %9 = memref.load %arg2[%7] : memref<?xindex>
      %10 = memref.load %arg2[%8] : memref<?xindex>
      %11 = arith.cmpi ne, %9, %10 : index
      %12 = scf.if %11 -> (i1) {
        scf.yield %false : i1
      } else {
        %22 = arith.addi %7, %c1 : index
        %23 = arith.addi %8, %c1 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        %26 = arith.cmpi ne, %24, %25 : index
        %27 = scf.if %26 -> (i1) {
          scf.yield %false : i1
        } else {
          %28 = arith.addi %7, %c2 : index
          %29 = arith.addi %8, %c2 : index
          %30 = memref.load %arg2[%28] : memref<?xindex>
          %31 = memref.load %arg2[%29] : memref<?xindex>
          %32 = arith.cmpi ne, %30, %31 : index
          %33 = scf.if %32 -> (i1) {
            scf.yield %false : i1
          } else {
            %34 = arith.addi %7, %c3 : index
            %35 = arith.addi %8, %c3 : index
            %36 = memref.load %arg2[%34] : memref<?xindex>
            %37 = memref.load %arg2[%35] : memref<?xindex>
            %38 = arith.cmpi eq, %36, %37 : index
            scf.yield %38 : i1
          }
          scf.yield %33 : i1
        }
        scf.yield %27 : i1
      }
      %13 = scf.while (%arg8 = %arg5) : (index) -> index {
        %22 = arith.muli %arg6, %c4 : index
        %23 = arith.muli %arg8, %c4 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        %26 = arith.cmpi ne, %24, %25 : index
        %27 = scf.if %26 -> (i1) {
          %28 = arith.cmpi ult, %24, %25 : index
          scf.yield %28 : i1
        } else {
          %28 = arith.addi %22, %c1 : index
          %29 = arith.addi %23, %c1 : index
          %30 = memref.load %arg2[%28] : memref<?xindex>
          %31 = memref.load %arg2[%29] : memref<?xindex>
          %32 = arith.cmpi ne, %30, %31 : index
          %33 = scf.if %32 -> (i1) {
            %34 = arith.cmpi ult, %30, %31 : index
            scf.yield %34 : i1
          } else {
            %34 = arith.addi %22, %c2 : index
            %35 = arith.addi %23, %c2 : index
            %36 = memref.load %arg2[%34] : memref<?xindex>
            %37 = memref.load %arg2[%35] : memref<?xindex>
            %38 = arith.cmpi ne, %36, %37 : index
            %39 = scf.if %38 -> (i1) {
              %40 = arith.cmpi ult, %36, %37 : index
              scf.yield %40 : i1
            } else {
              %40 = arith.addi %22, %c3 : index
              %41 = arith.addi %23, %c3 : index
              %42 = memref.load %arg2[%40] : memref<?xindex>
              %43 = memref.load %arg2[%41] : memref<?xindex>
              %44 = arith.cmpi ult, %42, %43 : index
              scf.yield %44 : i1
            }
            scf.yield %39 : i1
          }
          scf.yield %33 : i1
        }
        scf.condition(%27) %arg8 : index
      } do {
      ^bb0(%arg8: index):
        %22 = arith.addi %arg8, %c-1 : index
        scf.yield %22 : index
      }
      %14 = arith.muli %13, %c4 : index
      %15 = arith.muli %arg6, %c4 : index
      %16 = memref.load %arg2[%14] : memref<?xindex>
      %17 = memref.load %arg2[%15] : memref<?xindex>
      %18 = arith.cmpi ne, %16, %17 : index
      %19 = scf.if %18 -> (i1) {
        scf.yield %false : i1
      } else {
        %22 = arith.addi %14, %c1 : index
        %23 = arith.addi %15, %c1 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        %26 = arith.cmpi ne, %24, %25 : index
        %27 = scf.if %26 -> (i1) {
          scf.yield %false : i1
        } else {
          %28 = arith.addi %14, %c2 : index
          %29 = arith.addi %15, %c2 : index
          %30 = memref.load %arg2[%28] : memref<?xindex>
          %31 = memref.load %arg2[%29] : memref<?xindex>
          %32 = arith.cmpi ne, %30, %31 : index
          %33 = scf.if %32 -> (i1) {
            scf.yield %false : i1
          } else {
            %34 = arith.addi %14, %c3 : index
            %35 = arith.addi %15, %c3 : index
            %36 = memref.load %arg2[%34] : memref<?xindex>
            %37 = memref.load %arg2[%35] : memref<?xindex>
            %38 = arith.cmpi eq, %36, %37 : index
            scf.yield %38 : i1
          }
          scf.yield %33 : i1
        }
        scf.yield %27 : i1
      }
      %20 = arith.cmpi ult, %6, %13 : index
      %21:4 = scf.if %20 -> (index, index, index, i1) {
        %22 = arith.muli %6, %c4 : index
        %23 = arith.muli %13, %c4 : index
        %24 = memref.load %arg2[%22] : memref<?xindex>
        %25 = memref.load %arg2[%23] : memref<?xindex>
        memref.store %25, %arg2[%22] : memref<?xindex>
        memref.store %24, %arg2[%23] : memref<?xindex>
        %26 = arith.addi %22, %c1 : index
        %27 = arith.addi %23, %c1 : index
        %28 = memref.load %arg2[%26] : memref<?xindex>
        %29 = memref.load %arg2[%27] : memref<?xindex>
        memref.store %29, %arg2[%26] : memref<?xindex>
        memref.store %28, %arg2[%27] : memref<?xindex>
        %30 = arith.addi %22, %c2 : index
        %31 = arith.addi %23, %c2 : index
        %32 = memref.load %arg2[%30] : memref<?xindex>
        %33 = memref.load %arg2[%31] : memref<?xindex>
        memref.store %33, %arg2[%30] : memref<?xindex>
        memref.store %32, %arg2[%31] : memref<?xindex>
        %34 = arith.addi %22, %c3 : index
        %35 = arith.addi %23, %c3 : index
        %36 = memref.load %arg2[%34] : memref<?xindex>
        %37 = memref.load %arg2[%35] : memref<?xindex>
        memref.store %37, %arg2[%34] : memref<?xindex>
        memref.store %36, %arg2[%35] : memref<?xindex>
        %38 = memref.load %arg3[%6] : memref<?xf64>
        %39 = memref.load %arg3[%13] : memref<?xf64>
        memref.store %39, %arg3[%6] : memref<?xf64>
        memref.store %38, %arg3[%13] : memref<?xf64>
        %40 = arith.cmpi eq, %6, %arg6 : index
        %41 = scf.if %40 -> (index) {
          scf.yield %13 : index
        } else {
          %44 = arith.cmpi eq, %13, %arg6 : index
          %45 = scf.if %44 -> (index) {
            scf.yield %6 : index
          } else {
            scf.yield %arg6 : index
          }
          scf.yield %45 : index
        }
        %42 = arith.andi %12, %19 : i1
        %43:2 = scf.if %42 -> (index, index) {
          %44 = arith.addi %6, %c1 : index
          %45 = arith.subi %13, %c1 : index
          scf.yield %44, %45 : index, index
        } else {
          scf.yield %6, %13 : index, index
        }
        scf.yield %43#0, %43#1, %41, %true : index, index, index, i1
      } else {
        %22 = arith.addi %13, %c1 : index
        scf.yield %6, %13, %22, %false : index, index, index, i1
      }
      scf.yield %21#0, %21#1, %21#2, %21#3 : index, index, index, i1
    }
    return %5#2 : index
  }
  func.func private @_sparse_hybrid_qsort_0_1_2_3_index_coo_0_f64(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf64>, %arg4: i64) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c30 = arith.constant 30 : index
    %c1 = arith.constant 1 : index
    %0:2 = scf.while (%arg5 = %arg0, %arg6 = %arg1) : (index, index) -> (index, index) {
      %1 = arith.addi %arg5, %c1 : index
      %2 = arith.cmpi ult, %1, %arg6 : index
      scf.condition(%2) %arg5, %arg6 : index, index
    } do {
    ^bb0(%arg5: index, %arg6: index):
      %1 = arith.subi %arg6, %arg5 : index
      %2 = arith.cmpi ule, %1, %c30 : index
      %3:2 = scf.if %2 -> (index, index) {
        func.call @_sparse_sort_stable_0_1_2_3_index_coo_0_f64(%arg5, %arg6, %arg2, %arg3) : (index, index, memref<?xindex>, memref<?xf64>) -> ()
        scf.yield %arg5, %arg5 : index, index
      } else {
        %4 = arith.subi %arg4, %c1_i64 : i64
        %5 = arith.cmpi ule, %4, %c0_i64 : i64
        %6:2 = scf.if %5 -> (index, index) {
          func.call @_sparse_heap_sort_0_1_2_3_index_coo_0_f64(%arg5, %arg6, %arg2, %arg3) : (index, index, memref<?xindex>, memref<?xf64>) -> ()
          scf.yield %arg5, %arg5 : index, index
        } else {
          %7 = func.call @_sparse_partition_0_1_2_3_index_coo_0_f64(%arg5, %arg6, %arg2, %arg3) : (index, index, memref<?xindex>, memref<?xf64>) -> index
          %8 = arith.subi %7, %arg5 : index
          %9 = arith.subi %arg6, %7 : index
          %10 = arith.subi %arg6, %arg5 : index
          %11 = arith.cmpi ugt, %10, %c2 : index
          %12:2 = scf.if %11 -> (index, index) {
            %13 = arith.cmpi ule, %8, %9 : index
            %14:2 = scf.if %13 -> (index, index) {
              %15 = arith.cmpi ne, %8, %c0 : index
              scf.if %15 {
                func.call @_sparse_hybrid_qsort_0_1_2_3_index_coo_0_f64(%arg5, %7, %arg2, %arg3, %4) : (index, index, memref<?xindex>, memref<?xf64>, i64) -> ()
              }
              scf.yield %7, %arg6 : index, index
            } else {
              %15 = arith.cmpi ne, %9, %c0 : index
              scf.if %15 {
                func.call @_sparse_hybrid_qsort_0_1_2_3_index_coo_0_f64(%7, %arg6, %arg2, %arg3, %4) : (index, index, memref<?xindex>, memref<?xf64>, i64) -> ()
              }
              scf.yield %arg5, %7 : index, index
            }
            scf.yield %14#0, %14#1 : index, index
          } else {
            scf.yield %arg5, %arg5 : index, index
          }
          scf.yield %12#0, %12#1 : index, index
        }
        scf.yield %6#0, %6#1 : index, index
      }
      scf.yield %3#0, %3#1 : index, index
    }
    return
  }
  func.func @main() {
    %c32 = arith.constant 32 : index
    %c64_i64 = arith.constant 64 : i64
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @tensorL : !llvm.ptr
    %1 = llvm.mlir.addressof @tensorBra : !llvm.ptr
    %2 = llvm.mlir.addressof @tensorW : !llvm.ptr
    %3 = llvm.mlir.addressof @tensorKet : !llvm.ptr
    %4 = llvm.mlir.addressof @tensorR : !llvm.ptr
    %alloca = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca[%c0] : memref<?xindex>
    memref.store %c0, %alloca[%c1] : memref<?xindex>
    memref.store %c0, %alloca[%c2] : memref<?xindex>
    %5 = call @createCheckedSparseTensorReader(%0, %alloca, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %6 = call @getSparseTensorReaderDimSizes(%5) : (!llvm.ptr) -> memref<?xindex>
    %7 = memref.load %6[%c0] : memref<?xindex>
    %8 = memref.load %6[%c1] : memref<?xindex>
    %9 = memref.load %6[%c2] : memref<?xindex>
    %10 = call @getSparseTensorReaderNSE(%5) : (!llvm.ptr) -> index
    %alloca_0 = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca_0[%c0] : memref<?xindex>
    memref.store %c1, %alloca_0[%c1] : memref<?xindex>
    memref.store %c2, %alloca_0[%c2] : memref<?xindex>
    %11 = arith.muli %10, %c3 : index
    %alloc = memref.alloc(%c2) : memref<?xindex>
    %alloc_1 = memref.alloc(%11) : memref<?xindex>
    %alloc_2 = memref.alloc(%10) : memref<?xf64>
    %12 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse2>
    %13 = sparse_tensor.storage_specifier.set %12  lvl_sz at 0 with %7 : !sparse_tensor.storage_specifier<#sparse2>
    %14 = sparse_tensor.storage_specifier.get %13  pos_mem_sz at 0 : !sparse_tensor.storage_specifier<#sparse2>
    %15 = arith.addi %14, %c1 : index
    %16 = arith.cmpi ugt, %15, %c2 : index
    %17 = scf.if %16 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc(%c4) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc : memref<?xindex>
    }
    memref.store %c0, %17[%14] : memref<?xindex>
    %dim = memref.dim %17, %c0 : memref<?xindex>
    %18 = arith.addi %15, %c1 : index
    %19 = arith.cmpi ugt, %18, %dim : index
    %20 = scf.if %19 -> (memref<?xindex>) {
      %271 = arith.muli %dim, %c2 : index
      %272 = memref.realloc %17(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %17 : memref<?xindex>
    }
    memref.store %c0, %20[%15] : memref<?xindex>
    %21 = call @getSparseTensorReaderReadToBuffers0F64(%5, %alloca_0, %alloca_0, %alloc_1, %alloc_2) : (!llvm.ptr, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>) -> i1
    %22 = arith.cmpi eq, %21, %false : i1
    scf.if %22 {
      %271 = arith.index_cast %10 : index to i64
      %272 = math.ctlz %271 : i64
      %273 = arith.subi %c64_i64, %272 : i64
      func.call @_sparse_hybrid_qsort_0_1_2_index_coo_0_f64(%c0, %10, %alloc_1, %alloc_2, %273) : (index, index, memref<?xindex>, memref<?xf64>, i64) -> ()
    }
    memref.store %10, %20[%c1] : memref<?xindex>
    %23 = arith.muli %10, %c3 : index
    call @delSparseTensorReader(%5) : (!llvm.ptr) -> ()
    %alloc_3 = memref.alloc(%c16) : memref<?xindex>
    %alloc_4 = memref.alloc(%c16) : memref<?xindex>
    %alloc_5 = memref.alloc(%c16) : memref<?xindex>
    %alloc_6 = memref.alloc(%c16) : memref<?xindex>
    %alloc_7 = memref.alloc(%c16) : memref<?xf64>
    %24 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse>
    %25 = sparse_tensor.storage_specifier.set %24  lvl_sz at 0 with %7 : !sparse_tensor.storage_specifier<#sparse>
    %26 = sparse_tensor.storage_specifier.set %25  lvl_sz at 1 with %8 : !sparse_tensor.storage_specifier<#sparse>
    %27 = sparse_tensor.storage_specifier.get %26  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %28 = arith.addi %27, %c1 : index
    %29 = arith.cmpi ugt, %28, %c16 : index
    %30 = scf.if %29 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_3(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_3 : memref<?xindex>
    }
    memref.store %c0, %30[%27] : memref<?xindex>
    %31 = sparse_tensor.storage_specifier.set %26  pos_mem_sz at 1 with %28 : !sparse_tensor.storage_specifier<#sparse>
    %32 = sparse_tensor.storage_specifier.set %31  lvl_sz at 2 with %9 : !sparse_tensor.storage_specifier<#sparse>
    %33 = sparse_tensor.storage_specifier.get %32  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %34 = arith.addi %33, %c1 : index
    %35 = arith.cmpi ugt, %34, %c16 : index
    %36 = scf.if %35 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_5(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_5 : memref<?xindex>
    }
    memref.store %c0, %36[%33] : memref<?xindex>
    %37 = sparse_tensor.storage_specifier.set %32  pos_mem_sz at 2 with %34 : !sparse_tensor.storage_specifier<#sparse>
    %dim_8 = memref.dim %30, %c0 : memref<?xindex>
    %38 = arith.addi %28, %7 : index
    %39 = arith.cmpi ugt, %38, %dim_8 : index
    %40 = scf.if %39 -> (memref<?xindex>) {
      %271 = scf.while (%arg0 = %dim_8) : (index) -> index {
        %273 = arith.muli %arg0, %c2 : index
        %274 = arith.cmpi ugt, %38, %273 : index
        scf.condition(%274) %273 : index
      } do {
      ^bb0(%arg0: index):
        scf.yield %arg0 : index
      }
      %272 = memref.realloc %30(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %30 : memref<?xindex>
    }
    %subview = memref.subview %40[%28] [%7] [%c1] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    linalg.fill ins(%c0 : index) outs(%subview : memref<?xindex, strided<[?], offset: ?>>)
    %41 = sparse_tensor.storage_specifier.set %37  pos_mem_sz at 1 with %38 : !sparse_tensor.storage_specifier<#sparse>
    %subview_9 = memref.subview %alloc_2[0] [%10] [1] : memref<?xf64> to memref<?xf64>
    %subview_10 = memref.subview %20[0] [%18] [1] : memref<?xindex> to memref<?xindex>
    %42 = arith.divui %23, %c3 : index
    %subview_11 = memref.subview %alloc_1[%c0] [%42] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %43 = arith.divui %23, %c3 : index
    %subview_12 = memref.subview %alloc_1[%c1] [%43] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %44 = arith.divui %23, %c3 : index
    %subview_13 = memref.subview %alloc_1[%c2] [%44] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %45 = memref.load %subview_10[%c0] : memref<?xindex>
    %46 = memref.load %subview_10[%c1] : memref<?xindex>
    %47 = scf.while (%arg0 = %45) : (index) -> index {
      %271 = arith.cmpi ult, %arg0, %46 : index
      %272 = scf.if %271 -> (i1) {
        %273 = memref.load %subview_11[%45] : memref<?xindex, strided<[?], offset: ?>>
        %274 = memref.load %subview_11[%arg0] : memref<?xindex, strided<[?], offset: ?>>
        %275 = arith.cmpi eq, %273, %274 : index
        scf.yield %275 : i1
      } else {
        scf.yield %false : i1
      }
      scf.condition(%272) %arg0 : index
    } do {
    ^bb0(%arg0: index):
      %271 = arith.addi %arg0, %c1 : index
      scf.yield %271 : index
    }
    %48:8 = scf.while (%arg0 = %45, %arg1 = %47, %arg2 = %40, %arg3 = %alloc_4, %arg4 = %36, %arg5 = %alloc_6, %arg6 = %alloc_7, %arg7 = %41) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
      %271 = arith.cmpi ult, %arg0, %46 : index
      scf.condition(%271) %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
    } do {
    ^bb0(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: memref<?xindex>, %arg6: memref<?xf64>, %arg7: !sparse_tensor.storage_specifier<#sparse>):
      %271 = memref.load %subview_11[%arg0] : memref<?xindex, strided<[?], offset: ?>>
      %272 = scf.while (%arg8 = %arg0) : (index) -> index {
        %275 = arith.cmpi ult, %arg8, %arg1 : index
        %276 = scf.if %275 -> (i1) {
          %277 = memref.load %subview_12[%arg0] : memref<?xindex, strided<[?], offset: ?>>
          %278 = memref.load %subview_12[%arg8] : memref<?xindex, strided<[?], offset: ?>>
          %279 = arith.cmpi eq, %277, %278 : index
          scf.yield %279 : i1
        } else {
          scf.yield %false : i1
        }
        scf.condition(%276) %arg8 : index
      } do {
      ^bb0(%arg8: index):
        %275 = arith.addi %arg8, %c1 : index
        scf.yield %275 : index
      }
      %273:8 = scf.while (%arg8 = %arg0, %arg9 = %272, %arg10 = %arg2, %arg11 = %arg3, %arg12 = %arg4, %arg13 = %arg5, %arg14 = %arg6, %arg15 = %arg7) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
        %275 = arith.cmpi ult, %arg8, %arg1 : index
        scf.condition(%275) %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
      } do {
      ^bb0(%arg8: index, %arg9: index, %arg10: memref<?xindex>, %arg11: memref<?xindex>, %arg12: memref<?xindex>, %arg13: memref<?xindex>, %arg14: memref<?xf64>, %arg15: !sparse_tensor.storage_specifier<#sparse>):
        %275 = memref.load %subview_12[%arg8] : memref<?xindex, strided<[?], offset: ?>>
        %276:6 = scf.for %arg16 = %arg8 to %arg9 step %c1 iter_args(%arg17 = %arg10, %arg18 = %arg11, %arg19 = %arg12, %arg20 = %arg13, %arg21 = %arg14, %arg22 = %arg15) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
          %278 = memref.load %subview_13[%arg16] : memref<?xindex, strided<[?], offset: ?>>
          %279 = memref.load %subview_9[%arg16] : memref<?xf64>
          %280:6 = func.call @"_insert_dense_compressed_compressed_-9223372036854775808_-9223372036854775808_-9223372036854775808_f64_0_0"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %271, %275, %278, %279) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index, index, index, f64) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>)
          scf.yield %280#0, %280#1, %280#2, %280#3, %280#4, %280#5 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
        } {"Emitted from" = "sparse_tensor.foreach"}
        %277:2 = scf.if %true -> (index, index) {
          %278 = scf.while (%arg16 = %arg9) : (index) -> index {
            %279 = arith.cmpi ult, %arg16, %arg1 : index
            %280 = scf.if %279 -> (i1) {
              %281 = memref.load %subview_12[%arg9] : memref<?xindex, strided<[?], offset: ?>>
              %282 = memref.load %subview_12[%arg16] : memref<?xindex, strided<[?], offset: ?>>
              %283 = arith.cmpi eq, %281, %282 : index
              scf.yield %283 : i1
            } else {
              scf.yield %false : i1
            }
            scf.condition(%280) %arg16 : index
          } do {
          ^bb0(%arg16: index):
            %279 = arith.addi %arg16, %c1 : index
            scf.yield %279 : index
          }
          scf.yield %arg9, %278 : index, index
        } else {
          scf.yield %arg8, %arg9 : index, index
        }
        scf.yield %277#0, %277#1, %276#0, %276#1, %276#2, %276#3, %276#4, %276#5 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
      }
      %274:2 = scf.if %true -> (index, index) {
        %275 = scf.while (%arg8 = %arg1) : (index) -> index {
          %276 = arith.cmpi ult, %arg8, %46 : index
          %277 = scf.if %276 -> (i1) {
            %278 = memref.load %subview_11[%arg1] : memref<?xindex, strided<[?], offset: ?>>
            %279 = memref.load %subview_11[%arg8] : memref<?xindex, strided<[?], offset: ?>>
            %280 = arith.cmpi eq, %278, %279 : index
            scf.yield %280 : i1
          } else {
            scf.yield %false : i1
          }
          scf.condition(%277) %arg8 : index
        } do {
        ^bb0(%arg8: index):
          %276 = arith.addi %arg8, %c1 : index
          scf.yield %276 : index
        }
        scf.yield %arg1, %275 : index, index
      } else {
        scf.yield %arg0, %arg1 : index, index
      }
      scf.yield %274#0, %274#1, %273#2, %273#3, %273#4, %273#5, %273#6, %273#7 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
    }
    %49 = sparse_tensor.storage_specifier.get %48#7  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %50 = memref.load %48#2[%c0] : memref<?xindex>
    %51 = scf.for %arg0 = %c1 to %49 step %c1 iter_args(%arg1 = %50) -> (index) {
      %271 = memref.load %48#2[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %48#2[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %52 = sparse_tensor.storage_specifier.get %48#7  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %53 = memref.load %48#4[%c0] : memref<?xindex>
    %54 = scf.for %arg0 = %c1 to %52 step %c1 iter_args(%arg1 = %53) -> (index) {
      %271 = memref.load %48#4[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %48#4[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %alloca_14 = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca_14[%c0] : memref<?xindex>
    memref.store %c0, %alloca_14[%c1] : memref<?xindex>
    memref.store %c0, %alloca_14[%c2] : memref<?xindex>
    %55 = call @createCheckedSparseTensorReader(%1, %alloca_14, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %56 = call @getSparseTensorReaderDimSizes(%55) : (!llvm.ptr) -> memref<?xindex>
    %57 = memref.load %56[%c0] : memref<?xindex>
    %58 = memref.load %56[%c1] : memref<?xindex>
    %59 = memref.load %56[%c2] : memref<?xindex>
    %60 = call @getSparseTensorReaderNSE(%55) : (!llvm.ptr) -> index
    %alloca_15 = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca_15[%c0] : memref<?xindex>
    memref.store %c1, %alloca_15[%c1] : memref<?xindex>
    memref.store %c2, %alloca_15[%c2] : memref<?xindex>
    %61 = arith.muli %60, %c3 : index
    %alloc_16 = memref.alloc(%c2) : memref<?xindex>
    %alloc_17 = memref.alloc(%61) : memref<?xindex>
    %alloc_18 = memref.alloc(%60) : memref<?xf64>
    %62 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse2>
    %63 = sparse_tensor.storage_specifier.set %62  lvl_sz at 0 with %57 : !sparse_tensor.storage_specifier<#sparse2>
    %64 = sparse_tensor.storage_specifier.get %63  pos_mem_sz at 0 : !sparse_tensor.storage_specifier<#sparse2>
    %65 = arith.addi %64, %c1 : index
    %66 = arith.cmpi ugt, %65, %c2 : index
    %67 = scf.if %66 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_16(%c4) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_16 : memref<?xindex>
    }
    memref.store %c0, %67[%64] : memref<?xindex>
    %dim_19 = memref.dim %67, %c0 : memref<?xindex>
    %68 = arith.addi %65, %c1 : index
    %69 = arith.cmpi ugt, %68, %dim_19 : index
    %70 = scf.if %69 -> (memref<?xindex>) {
      %271 = arith.muli %dim_19, %c2 : index
      %272 = memref.realloc %67(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %67 : memref<?xindex>
    }
    memref.store %c0, %70[%65] : memref<?xindex>
    %71 = call @getSparseTensorReaderReadToBuffers0F64(%55, %alloca_15, %alloca_15, %alloc_17, %alloc_18) : (!llvm.ptr, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>) -> i1
    %72 = arith.cmpi eq, %71, %false : i1
    scf.if %72 {
      %271 = arith.index_cast %60 : index to i64
      %272 = math.ctlz %271 : i64
      %273 = arith.subi %c64_i64, %272 : i64
      func.call @_sparse_hybrid_qsort_0_1_2_index_coo_0_f64(%c0, %60, %alloc_17, %alloc_18, %273) : (index, index, memref<?xindex>, memref<?xf64>, i64) -> ()
    }
    memref.store %60, %70[%c1] : memref<?xindex>
    %73 = arith.muli %60, %c3 : index
    call @delSparseTensorReader(%55) : (!llvm.ptr) -> ()
    %alloc_20 = memref.alloc(%c16) : memref<?xindex>
    %alloc_21 = memref.alloc(%c16) : memref<?xindex>
    %alloc_22 = memref.alloc(%c16) : memref<?xindex>
    %alloc_23 = memref.alloc(%c16) : memref<?xindex>
    %alloc_24 = memref.alloc(%c16) : memref<?xf64>
    %74 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse>
    %75 = sparse_tensor.storage_specifier.set %74  lvl_sz at 0 with %57 : !sparse_tensor.storage_specifier<#sparse>
    %76 = sparse_tensor.storage_specifier.set %75  lvl_sz at 1 with %58 : !sparse_tensor.storage_specifier<#sparse>
    %77 = sparse_tensor.storage_specifier.get %76  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %78 = arith.addi %77, %c1 : index
    %79 = arith.cmpi ugt, %78, %c16 : index
    %80 = scf.if %79 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_20(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_20 : memref<?xindex>
    }
    memref.store %c0, %80[%77] : memref<?xindex>
    %81 = sparse_tensor.storage_specifier.set %76  pos_mem_sz at 1 with %78 : !sparse_tensor.storage_specifier<#sparse>
    %82 = sparse_tensor.storage_specifier.set %81  lvl_sz at 2 with %59 : !sparse_tensor.storage_specifier<#sparse>
    %83 = sparse_tensor.storage_specifier.get %82  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %84 = arith.addi %83, %c1 : index
    %85 = arith.cmpi ugt, %84, %c16 : index
    %86 = scf.if %85 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_22(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_22 : memref<?xindex>
    }
    memref.store %c0, %86[%83] : memref<?xindex>
    %87 = sparse_tensor.storage_specifier.set %82  pos_mem_sz at 2 with %84 : !sparse_tensor.storage_specifier<#sparse>
    %dim_25 = memref.dim %80, %c0 : memref<?xindex>
    %88 = arith.addi %78, %57 : index
    %89 = arith.cmpi ugt, %88, %dim_25 : index
    %90 = scf.if %89 -> (memref<?xindex>) {
      %271 = scf.while (%arg0 = %dim_25) : (index) -> index {
        %273 = arith.muli %arg0, %c2 : index
        %274 = arith.cmpi ugt, %88, %273 : index
        scf.condition(%274) %273 : index
      } do {
      ^bb0(%arg0: index):
        scf.yield %arg0 : index
      }
      %272 = memref.realloc %80(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %80 : memref<?xindex>
    }
    %subview_26 = memref.subview %90[%78] [%57] [%c1] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    linalg.fill ins(%c0 : index) outs(%subview_26 : memref<?xindex, strided<[?], offset: ?>>)
    %91 = sparse_tensor.storage_specifier.set %87  pos_mem_sz at 1 with %88 : !sparse_tensor.storage_specifier<#sparse>
    %subview_27 = memref.subview %alloc_18[0] [%60] [1] : memref<?xf64> to memref<?xf64>
    %subview_28 = memref.subview %70[0] [%68] [1] : memref<?xindex> to memref<?xindex>
    %92 = arith.divui %73, %c3 : index
    %subview_29 = memref.subview %alloc_17[%c0] [%92] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %93 = arith.divui %73, %c3 : index
    %subview_30 = memref.subview %alloc_17[%c1] [%93] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %94 = arith.divui %73, %c3 : index
    %subview_31 = memref.subview %alloc_17[%c2] [%94] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %95 = memref.load %subview_28[%c0] : memref<?xindex>
    %96 = memref.load %subview_28[%c1] : memref<?xindex>
    %97 = scf.while (%arg0 = %95) : (index) -> index {
      %271 = arith.cmpi ult, %arg0, %96 : index
      %272 = scf.if %271 -> (i1) {
        %273 = memref.load %subview_29[%95] : memref<?xindex, strided<[?], offset: ?>>
        %274 = memref.load %subview_29[%arg0] : memref<?xindex, strided<[?], offset: ?>>
        %275 = arith.cmpi eq, %273, %274 : index
        scf.yield %275 : i1
      } else {
        scf.yield %false : i1
      }
      scf.condition(%272) %arg0 : index
    } do {
    ^bb0(%arg0: index):
      %271 = arith.addi %arg0, %c1 : index
      scf.yield %271 : index
    }
    %98:8 = scf.while (%arg0 = %95, %arg1 = %97, %arg2 = %90, %arg3 = %alloc_21, %arg4 = %86, %arg5 = %alloc_23, %arg6 = %alloc_24, %arg7 = %91) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
      %271 = arith.cmpi ult, %arg0, %96 : index
      scf.condition(%271) %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
    } do {
    ^bb0(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: memref<?xindex>, %arg6: memref<?xf64>, %arg7: !sparse_tensor.storage_specifier<#sparse>):
      %271 = memref.load %subview_29[%arg0] : memref<?xindex, strided<[?], offset: ?>>
      %272 = scf.while (%arg8 = %arg0) : (index) -> index {
        %275 = arith.cmpi ult, %arg8, %arg1 : index
        %276 = scf.if %275 -> (i1) {
          %277 = memref.load %subview_30[%arg0] : memref<?xindex, strided<[?], offset: ?>>
          %278 = memref.load %subview_30[%arg8] : memref<?xindex, strided<[?], offset: ?>>
          %279 = arith.cmpi eq, %277, %278 : index
          scf.yield %279 : i1
        } else {
          scf.yield %false : i1
        }
        scf.condition(%276) %arg8 : index
      } do {
      ^bb0(%arg8: index):
        %275 = arith.addi %arg8, %c1 : index
        scf.yield %275 : index
      }
      %273:8 = scf.while (%arg8 = %arg0, %arg9 = %272, %arg10 = %arg2, %arg11 = %arg3, %arg12 = %arg4, %arg13 = %arg5, %arg14 = %arg6, %arg15 = %arg7) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
        %275 = arith.cmpi ult, %arg8, %arg1 : index
        scf.condition(%275) %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
      } do {
      ^bb0(%arg8: index, %arg9: index, %arg10: memref<?xindex>, %arg11: memref<?xindex>, %arg12: memref<?xindex>, %arg13: memref<?xindex>, %arg14: memref<?xf64>, %arg15: !sparse_tensor.storage_specifier<#sparse>):
        %275 = memref.load %subview_30[%arg8] : memref<?xindex, strided<[?], offset: ?>>
        %276:6 = scf.for %arg16 = %arg8 to %arg9 step %c1 iter_args(%arg17 = %arg10, %arg18 = %arg11, %arg19 = %arg12, %arg20 = %arg13, %arg21 = %arg14, %arg22 = %arg15) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
          %278 = memref.load %subview_31[%arg16] : memref<?xindex, strided<[?], offset: ?>>
          %279 = memref.load %subview_27[%arg16] : memref<?xf64>
          %280:6 = func.call @"_insert_dense_compressed_compressed_-9223372036854775808_-9223372036854775808_-9223372036854775808_f64_0_0"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %271, %275, %278, %279) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index, index, index, f64) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>)
          scf.yield %280#0, %280#1, %280#2, %280#3, %280#4, %280#5 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
        } {"Emitted from" = "sparse_tensor.foreach"}
        %277:2 = scf.if %true -> (index, index) {
          %278 = scf.while (%arg16 = %arg9) : (index) -> index {
            %279 = arith.cmpi ult, %arg16, %arg1 : index
            %280 = scf.if %279 -> (i1) {
              %281 = memref.load %subview_30[%arg9] : memref<?xindex, strided<[?], offset: ?>>
              %282 = memref.load %subview_30[%arg16] : memref<?xindex, strided<[?], offset: ?>>
              %283 = arith.cmpi eq, %281, %282 : index
              scf.yield %283 : i1
            } else {
              scf.yield %false : i1
            }
            scf.condition(%280) %arg16 : index
          } do {
          ^bb0(%arg16: index):
            %279 = arith.addi %arg16, %c1 : index
            scf.yield %279 : index
          }
          scf.yield %arg9, %278 : index, index
        } else {
          scf.yield %arg8, %arg9 : index, index
        }
        scf.yield %277#0, %277#1, %276#0, %276#1, %276#2, %276#3, %276#4, %276#5 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
      }
      %274:2 = scf.if %true -> (index, index) {
        %275 = scf.while (%arg8 = %arg1) : (index) -> index {
          %276 = arith.cmpi ult, %arg8, %96 : index
          %277 = scf.if %276 -> (i1) {
            %278 = memref.load %subview_29[%arg1] : memref<?xindex, strided<[?], offset: ?>>
            %279 = memref.load %subview_29[%arg8] : memref<?xindex, strided<[?], offset: ?>>
            %280 = arith.cmpi eq, %278, %279 : index
            scf.yield %280 : i1
          } else {
            scf.yield %false : i1
          }
          scf.condition(%277) %arg8 : index
        } do {
        ^bb0(%arg8: index):
          %276 = arith.addi %arg8, %c1 : index
          scf.yield %276 : index
        }
        scf.yield %arg1, %275 : index, index
      } else {
        scf.yield %arg0, %arg1 : index, index
      }
      scf.yield %274#0, %274#1, %273#2, %273#3, %273#4, %273#5, %273#6, %273#7 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
    }
    %99 = sparse_tensor.storage_specifier.get %98#7  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %100 = memref.load %98#2[%c0] : memref<?xindex>
    %101 = scf.for %arg0 = %c1 to %99 step %c1 iter_args(%arg1 = %100) -> (index) {
      %271 = memref.load %98#2[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %98#2[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %102 = sparse_tensor.storage_specifier.get %98#7  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %103 = memref.load %98#4[%c0] : memref<?xindex>
    %104 = scf.for %arg0 = %c1 to %102 step %c1 iter_args(%arg1 = %103) -> (index) {
      %271 = memref.load %98#4[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %98#4[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %alloca_32 = memref.alloca(%c4) : memref<?xindex>
    memref.store %c0, %alloca_32[%c0] : memref<?xindex>
    memref.store %c0, %alloca_32[%c1] : memref<?xindex>
    memref.store %c0, %alloca_32[%c2] : memref<?xindex>
    memref.store %c0, %alloca_32[%c3] : memref<?xindex>
    %105 = call @createCheckedSparseTensorReader(%2, %alloca_32, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %106 = call @getSparseTensorReaderDimSizes(%105) : (!llvm.ptr) -> memref<?xindex>
    %107 = memref.load %106[%c0] : memref<?xindex>
    %108 = memref.load %106[%c1] : memref<?xindex>
    %109 = memref.load %106[%c2] : memref<?xindex>
    %110 = memref.load %106[%c3] : memref<?xindex>
    %111 = call @getSparseTensorReaderNSE(%105) : (!llvm.ptr) -> index
    %alloca_33 = memref.alloca(%c4) : memref<?xindex>
    memref.store %c0, %alloca_33[%c0] : memref<?xindex>
    memref.store %c1, %alloca_33[%c1] : memref<?xindex>
    memref.store %c2, %alloca_33[%c2] : memref<?xindex>
    memref.store %c3, %alloca_33[%c3] : memref<?xindex>
    %112 = arith.muli %111, %c4 : index
    %alloc_34 = memref.alloc(%c2) : memref<?xindex>
    %alloc_35 = memref.alloc(%112) : memref<?xindex>
    %alloc_36 = memref.alloc(%111) : memref<?xf64>
    %113 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse3>
    %114 = sparse_tensor.storage_specifier.set %113  lvl_sz at 0 with %107 : !sparse_tensor.storage_specifier<#sparse3>
    %115 = sparse_tensor.storage_specifier.get %114  pos_mem_sz at 0 : !sparse_tensor.storage_specifier<#sparse3>
    %116 = arith.addi %115, %c1 : index
    %117 = arith.cmpi ugt, %116, %c2 : index
    %118 = scf.if %117 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_34(%c4) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_34 : memref<?xindex>
    }
    memref.store %c0, %118[%115] : memref<?xindex>
    %dim_37 = memref.dim %118, %c0 : memref<?xindex>
    %119 = arith.addi %116, %c1 : index
    %120 = arith.cmpi ugt, %119, %dim_37 : index
    %121 = scf.if %120 -> (memref<?xindex>) {
      %271 = arith.muli %dim_37, %c2 : index
      %272 = memref.realloc %118(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %118 : memref<?xindex>
    }
    memref.store %c0, %121[%116] : memref<?xindex>
    %122 = call @getSparseTensorReaderReadToBuffers0F64(%105, %alloca_33, %alloca_33, %alloc_35, %alloc_36) : (!llvm.ptr, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>) -> i1
    %123 = arith.cmpi eq, %122, %false : i1
    scf.if %123 {
      %271 = arith.index_cast %111 : index to i64
      %272 = math.ctlz %271 : i64
      %273 = arith.subi %c64_i64, %272 : i64
      func.call @_sparse_hybrid_qsort_0_1_2_3_index_coo_0_f64(%c0, %111, %alloc_35, %alloc_36, %273) : (index, index, memref<?xindex>, memref<?xf64>, i64) -> ()
    }
    memref.store %111, %121[%c1] : memref<?xindex>
    %124 = arith.muli %111, %c4 : index
    call @delSparseTensorReader(%105) : (!llvm.ptr) -> ()
    %alloc_38 = memref.alloc(%c16) : memref<?xindex>
    %alloc_39 = memref.alloc(%c16) : memref<?xindex>
    %alloc_40 = memref.alloc(%c16) : memref<?xindex>
    %alloc_41 = memref.alloc(%c16) : memref<?xindex>
    %alloc_42 = memref.alloc(%c16) : memref<?xindex>
    %alloc_43 = memref.alloc(%c16) : memref<?xindex>
    %alloc_44 = memref.alloc(%c16) : memref<?xf64>
    %125 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse1>
    %126 = sparse_tensor.storage_specifier.set %125  lvl_sz at 0 with %107 : !sparse_tensor.storage_specifier<#sparse1>
    %127 = sparse_tensor.storage_specifier.set %126  lvl_sz at 1 with %108 : !sparse_tensor.storage_specifier<#sparse1>
    %128 = sparse_tensor.storage_specifier.get %127  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse1>
    %129 = arith.addi %128, %c1 : index
    %130 = arith.cmpi ugt, %129, %c16 : index
    %131 = scf.if %130 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_38(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_38 : memref<?xindex>
    }
    memref.store %c0, %131[%128] : memref<?xindex>
    %132 = sparse_tensor.storage_specifier.set %127  pos_mem_sz at 1 with %129 : !sparse_tensor.storage_specifier<#sparse1>
    %133 = sparse_tensor.storage_specifier.set %132  lvl_sz at 2 with %109 : !sparse_tensor.storage_specifier<#sparse1>
    %134 = sparse_tensor.storage_specifier.get %133  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse1>
    %135 = arith.addi %134, %c1 : index
    %136 = arith.cmpi ugt, %135, %c16 : index
    %137 = scf.if %136 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_40(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_40 : memref<?xindex>
    }
    memref.store %c0, %137[%134] : memref<?xindex>
    %138 = sparse_tensor.storage_specifier.set %133  pos_mem_sz at 2 with %135 : !sparse_tensor.storage_specifier<#sparse1>
    %139 = sparse_tensor.storage_specifier.set %138  lvl_sz at 3 with %110 : !sparse_tensor.storage_specifier<#sparse1>
    %140 = sparse_tensor.storage_specifier.get %139  pos_mem_sz at 3 : !sparse_tensor.storage_specifier<#sparse1>
    %141 = arith.addi %140, %c1 : index
    %142 = arith.cmpi ugt, %141, %c16 : index
    %143 = scf.if %142 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_42(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_42 : memref<?xindex>
    }
    memref.store %c0, %143[%140] : memref<?xindex>
    %144 = sparse_tensor.storage_specifier.set %139  pos_mem_sz at 3 with %141 : !sparse_tensor.storage_specifier<#sparse1>
    %dim_45 = memref.dim %131, %c0 : memref<?xindex>
    %145 = arith.addi %129, %107 : index
    %146 = arith.cmpi ugt, %145, %dim_45 : index
    %147 = scf.if %146 -> (memref<?xindex>) {
      %271 = scf.while (%arg0 = %dim_45) : (index) -> index {
        %273 = arith.muli %arg0, %c2 : index
        %274 = arith.cmpi ugt, %145, %273 : index
        scf.condition(%274) %273 : index
      } do {
      ^bb0(%arg0: index):
        scf.yield %arg0 : index
      }
      %272 = memref.realloc %131(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %131 : memref<?xindex>
    }
    %subview_46 = memref.subview %147[%129] [%107] [%c1] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    linalg.fill ins(%c0 : index) outs(%subview_46 : memref<?xindex, strided<[?], offset: ?>>)
    %148 = sparse_tensor.storage_specifier.set %144  pos_mem_sz at 1 with %145 : !sparse_tensor.storage_specifier<#sparse1>
    %subview_47 = memref.subview %alloc_36[0] [%111] [1] : memref<?xf64> to memref<?xf64>
    %subview_48 = memref.subview %121[0] [%119] [1] : memref<?xindex> to memref<?xindex>
    %149 = arith.divui %124, %c4 : index
    %subview_49 = memref.subview %alloc_35[%c0] [%149] [%c4] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %150 = arith.divui %124, %c4 : index
    %subview_50 = memref.subview %alloc_35[%c1] [%150] [%c4] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %151 = arith.divui %124, %c4 : index
    %subview_51 = memref.subview %alloc_35[%c2] [%151] [%c4] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %152 = arith.divui %124, %c4 : index
    %subview_52 = memref.subview %alloc_35[%c3] [%152] [%c4] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %153 = memref.load %subview_48[%c0] : memref<?xindex>
    %154 = memref.load %subview_48[%c1] : memref<?xindex>
    %155 = scf.while (%arg0 = %153) : (index) -> index {
      %271 = arith.cmpi ult, %arg0, %154 : index
      %272 = scf.if %271 -> (i1) {
        %273 = memref.load %subview_49[%153] : memref<?xindex, strided<[?], offset: ?>>
        %274 = memref.load %subview_49[%arg0] : memref<?xindex, strided<[?], offset: ?>>
        %275 = arith.cmpi eq, %273, %274 : index
        scf.yield %275 : i1
      } else {
        scf.yield %false : i1
      }
      scf.condition(%272) %arg0 : index
    } do {
    ^bb0(%arg0: index):
      %271 = arith.addi %arg0, %c1 : index
      scf.yield %271 : index
    }
    %156:10 = scf.while (%arg0 = %153, %arg1 = %155, %arg2 = %147, %arg3 = %alloc_39, %arg4 = %137, %arg5 = %alloc_41, %arg6 = %143, %arg7 = %alloc_43, %arg8 = %alloc_44, %arg9 = %148) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>) {
      %271 = arith.cmpi ult, %arg0, %154 : index
      scf.condition(%271) %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>
    } do {
    ^bb0(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: memref<?xindex>, %arg6: memref<?xindex>, %arg7: memref<?xindex>, %arg8: memref<?xf64>, %arg9: !sparse_tensor.storage_specifier<#sparse1>):
      %271 = memref.load %subview_49[%arg0] : memref<?xindex, strided<[?], offset: ?>>
      %272 = scf.while (%arg10 = %arg0) : (index) -> index {
        %275 = arith.cmpi ult, %arg10, %arg1 : index
        %276 = scf.if %275 -> (i1) {
          %277 = memref.load %subview_50[%arg0] : memref<?xindex, strided<[?], offset: ?>>
          %278 = memref.load %subview_50[%arg10] : memref<?xindex, strided<[?], offset: ?>>
          %279 = arith.cmpi eq, %277, %278 : index
          scf.yield %279 : i1
        } else {
          scf.yield %false : i1
        }
        scf.condition(%276) %arg10 : index
      } do {
      ^bb0(%arg10: index):
        %275 = arith.addi %arg10, %c1 : index
        scf.yield %275 : index
      }
      %273:10 = scf.while (%arg10 = %arg0, %arg11 = %272, %arg12 = %arg2, %arg13 = %arg3, %arg14 = %arg4, %arg15 = %arg5, %arg16 = %arg6, %arg17 = %arg7, %arg18 = %arg8, %arg19 = %arg9) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>) {
        %275 = arith.cmpi ult, %arg10, %arg1 : index
        scf.condition(%275) %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>
      } do {
      ^bb0(%arg10: index, %arg11: index, %arg12: memref<?xindex>, %arg13: memref<?xindex>, %arg14: memref<?xindex>, %arg15: memref<?xindex>, %arg16: memref<?xindex>, %arg17: memref<?xindex>, %arg18: memref<?xf64>, %arg19: !sparse_tensor.storage_specifier<#sparse1>):
        %275 = memref.load %subview_50[%arg10] : memref<?xindex, strided<[?], offset: ?>>
        %276 = scf.while (%arg20 = %arg10) : (index) -> index {
          %279 = arith.cmpi ult, %arg20, %arg11 : index
          %280 = scf.if %279 -> (i1) {
            %281 = memref.load %subview_51[%arg10] : memref<?xindex, strided<[?], offset: ?>>
            %282 = memref.load %subview_51[%arg20] : memref<?xindex, strided<[?], offset: ?>>
            %283 = arith.cmpi eq, %281, %282 : index
            scf.yield %283 : i1
          } else {
            scf.yield %false : i1
          }
          scf.condition(%280) %arg20 : index
        } do {
        ^bb0(%arg20: index):
          %279 = arith.addi %arg20, %c1 : index
          scf.yield %279 : index
        }
        %277:10 = scf.while (%arg20 = %arg10, %arg21 = %276, %arg22 = %arg12, %arg23 = %arg13, %arg24 = %arg14, %arg25 = %arg15, %arg26 = %arg16, %arg27 = %arg17, %arg28 = %arg18, %arg29 = %arg19) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>) {
          %279 = arith.cmpi ult, %arg20, %arg11 : index
          scf.condition(%279) %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>
        } do {
        ^bb0(%arg20: index, %arg21: index, %arg22: memref<?xindex>, %arg23: memref<?xindex>, %arg24: memref<?xindex>, %arg25: memref<?xindex>, %arg26: memref<?xindex>, %arg27: memref<?xindex>, %arg28: memref<?xf64>, %arg29: !sparse_tensor.storage_specifier<#sparse1>):
          %279 = memref.load %subview_51[%arg20] : memref<?xindex, strided<[?], offset: ?>>
          %280:8 = scf.for %arg30 = %arg20 to %arg21 step %c1 iter_args(%arg31 = %arg22, %arg32 = %arg23, %arg33 = %arg24, %arg34 = %arg25, %arg35 = %arg26, %arg36 = %arg27, %arg37 = %arg28, %arg38 = %arg29) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>) {
            %282 = memref.load %subview_52[%arg30] : memref<?xindex, strided<[?], offset: ?>>
            %283 = memref.load %subview_47[%arg30] : memref<?xf64>
            %284:8 = func.call @"_insert_dense_compressed_compressed_compressed_-9223372036854775808_-9223372036854775808_-9223372036854775808_-9223372036854775808_f64_0_0"(%arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %271, %275, %279, %282, %283) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, index, index, index, index, f64) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>)
            scf.yield %284#0, %284#1, %284#2, %284#3, %284#4, %284#5, %284#6, %284#7 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>
          } {"Emitted from" = "sparse_tensor.foreach"}
          %281:2 = scf.if %true -> (index, index) {
            %282 = scf.while (%arg30 = %arg21) : (index) -> index {
              %283 = arith.cmpi ult, %arg30, %arg11 : index
              %284 = scf.if %283 -> (i1) {
                %285 = memref.load %subview_51[%arg21] : memref<?xindex, strided<[?], offset: ?>>
                %286 = memref.load %subview_51[%arg30] : memref<?xindex, strided<[?], offset: ?>>
                %287 = arith.cmpi eq, %285, %286 : index
                scf.yield %287 : i1
              } else {
                scf.yield %false : i1
              }
              scf.condition(%284) %arg30 : index
            } do {
            ^bb0(%arg30: index):
              %283 = arith.addi %arg30, %c1 : index
              scf.yield %283 : index
            }
            scf.yield %arg21, %282 : index, index
          } else {
            scf.yield %arg20, %arg21 : index, index
          }
          scf.yield %281#0, %281#1, %280#0, %280#1, %280#2, %280#3, %280#4, %280#5, %280#6, %280#7 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>
        }
        %278:2 = scf.if %true -> (index, index) {
          %279 = scf.while (%arg20 = %arg11) : (index) -> index {
            %280 = arith.cmpi ult, %arg20, %arg1 : index
            %281 = scf.if %280 -> (i1) {
              %282 = memref.load %subview_50[%arg11] : memref<?xindex, strided<[?], offset: ?>>
              %283 = memref.load %subview_50[%arg20] : memref<?xindex, strided<[?], offset: ?>>
              %284 = arith.cmpi eq, %282, %283 : index
              scf.yield %284 : i1
            } else {
              scf.yield %false : i1
            }
            scf.condition(%281) %arg20 : index
          } do {
          ^bb0(%arg20: index):
            %280 = arith.addi %arg20, %c1 : index
            scf.yield %280 : index
          }
          scf.yield %arg11, %279 : index, index
        } else {
          scf.yield %arg10, %arg11 : index, index
        }
        scf.yield %278#0, %278#1, %277#2, %277#3, %277#4, %277#5, %277#6, %277#7, %277#8, %277#9 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>
      }
      %274:2 = scf.if %true -> (index, index) {
        %275 = scf.while (%arg10 = %arg1) : (index) -> index {
          %276 = arith.cmpi ult, %arg10, %154 : index
          %277 = scf.if %276 -> (i1) {
            %278 = memref.load %subview_49[%arg1] : memref<?xindex, strided<[?], offset: ?>>
            %279 = memref.load %subview_49[%arg10] : memref<?xindex, strided<[?], offset: ?>>
            %280 = arith.cmpi eq, %278, %279 : index
            scf.yield %280 : i1
          } else {
            scf.yield %false : i1
          }
          scf.condition(%277) %arg10 : index
        } do {
        ^bb0(%arg10: index):
          %276 = arith.addi %arg10, %c1 : index
          scf.yield %276 : index
        }
        scf.yield %arg1, %275 : index, index
      } else {
        scf.yield %arg0, %arg1 : index, index
      }
      scf.yield %274#0, %274#1, %273#2, %273#3, %273#4, %273#5, %273#6, %273#7, %273#8, %273#9 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>
    }
    %157 = sparse_tensor.storage_specifier.get %156#9  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse1>
    %158 = memref.load %156#2[%c0] : memref<?xindex>
    %159 = scf.for %arg0 = %c1 to %157 step %c1 iter_args(%arg1 = %158) -> (index) {
      %271 = memref.load %156#2[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %156#2[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %160 = sparse_tensor.storage_specifier.get %156#9  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse1>
    %161 = memref.load %156#4[%c0] : memref<?xindex>
    %162 = scf.for %arg0 = %c1 to %160 step %c1 iter_args(%arg1 = %161) -> (index) {
      %271 = memref.load %156#4[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %156#4[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %163 = sparse_tensor.storage_specifier.get %156#9  pos_mem_sz at 3 : !sparse_tensor.storage_specifier<#sparse1>
    %164 = memref.load %156#6[%c0] : memref<?xindex>
    %165 = scf.for %arg0 = %c1 to %163 step %c1 iter_args(%arg1 = %164) -> (index) {
      %271 = memref.load %156#6[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %156#6[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %alloca_53 = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca_53[%c0] : memref<?xindex>
    memref.store %c0, %alloca_53[%c1] : memref<?xindex>
    memref.store %c0, %alloca_53[%c2] : memref<?xindex>
    %166 = call @createCheckedSparseTensorReader(%3, %alloca_53, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %167 = call @getSparseTensorReaderDimSizes(%166) : (!llvm.ptr) -> memref<?xindex>
    %168 = memref.load %167[%c0] : memref<?xindex>
    %169 = memref.load %167[%c1] : memref<?xindex>
    %170 = memref.load %167[%c2] : memref<?xindex>
    %171 = call @getSparseTensorReaderNSE(%166) : (!llvm.ptr) -> index
    %alloca_54 = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca_54[%c0] : memref<?xindex>
    memref.store %c1, %alloca_54[%c1] : memref<?xindex>
    memref.store %c2, %alloca_54[%c2] : memref<?xindex>
    %172 = arith.muli %171, %c3 : index
    %alloc_55 = memref.alloc(%c2) : memref<?xindex>
    %alloc_56 = memref.alloc(%172) : memref<?xindex>
    %alloc_57 = memref.alloc(%171) : memref<?xf64>
    %173 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse2>
    %174 = sparse_tensor.storage_specifier.set %173  lvl_sz at 0 with %168 : !sparse_tensor.storage_specifier<#sparse2>
    %175 = sparse_tensor.storage_specifier.get %174  pos_mem_sz at 0 : !sparse_tensor.storage_specifier<#sparse2>
    %176 = arith.addi %175, %c1 : index
    %177 = arith.cmpi ugt, %176, %c2 : index
    %178 = scf.if %177 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_55(%c4) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_55 : memref<?xindex>
    }
    memref.store %c0, %178[%175] : memref<?xindex>
    %dim_58 = memref.dim %178, %c0 : memref<?xindex>
    %179 = arith.addi %176, %c1 : index
    %180 = arith.cmpi ugt, %179, %dim_58 : index
    %181 = scf.if %180 -> (memref<?xindex>) {
      %271 = arith.muli %dim_58, %c2 : index
      %272 = memref.realloc %178(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %178 : memref<?xindex>
    }
    memref.store %c0, %181[%176] : memref<?xindex>
    %182 = call @getSparseTensorReaderReadToBuffers0F64(%166, %alloca_54, %alloca_54, %alloc_56, %alloc_57) : (!llvm.ptr, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>) -> i1
    %183 = arith.cmpi eq, %182, %false : i1
    scf.if %183 {
      %271 = arith.index_cast %171 : index to i64
      %272 = math.ctlz %271 : i64
      %273 = arith.subi %c64_i64, %272 : i64
      func.call @_sparse_hybrid_qsort_0_1_2_index_coo_0_f64(%c0, %171, %alloc_56, %alloc_57, %273) : (index, index, memref<?xindex>, memref<?xf64>, i64) -> ()
    }
    memref.store %171, %181[%c1] : memref<?xindex>
    %184 = arith.muli %171, %c3 : index
    call @delSparseTensorReader(%166) : (!llvm.ptr) -> ()
    %alloc_59 = memref.alloc(%c16) : memref<?xindex>
    %alloc_60 = memref.alloc(%c16) : memref<?xindex>
    %alloc_61 = memref.alloc(%c16) : memref<?xindex>
    %alloc_62 = memref.alloc(%c16) : memref<?xindex>
    %alloc_63 = memref.alloc(%c16) : memref<?xf64>
    %185 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse>
    %186 = sparse_tensor.storage_specifier.set %185  lvl_sz at 0 with %168 : !sparse_tensor.storage_specifier<#sparse>
    %187 = sparse_tensor.storage_specifier.set %186  lvl_sz at 1 with %169 : !sparse_tensor.storage_specifier<#sparse>
    %188 = sparse_tensor.storage_specifier.get %187  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %189 = arith.addi %188, %c1 : index
    %190 = arith.cmpi ugt, %189, %c16 : index
    %191 = scf.if %190 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_59(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_59 : memref<?xindex>
    }
    memref.store %c0, %191[%188] : memref<?xindex>
    %192 = sparse_tensor.storage_specifier.set %187  pos_mem_sz at 1 with %189 : !sparse_tensor.storage_specifier<#sparse>
    %193 = sparse_tensor.storage_specifier.set %192  lvl_sz at 2 with %170 : !sparse_tensor.storage_specifier<#sparse>
    %194 = sparse_tensor.storage_specifier.get %193  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %195 = arith.addi %194, %c1 : index
    %196 = arith.cmpi ugt, %195, %c16 : index
    %197 = scf.if %196 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_61(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_61 : memref<?xindex>
    }
    memref.store %c0, %197[%194] : memref<?xindex>
    %198 = sparse_tensor.storage_specifier.set %193  pos_mem_sz at 2 with %195 : !sparse_tensor.storage_specifier<#sparse>
    %dim_64 = memref.dim %191, %c0 : memref<?xindex>
    %199 = arith.addi %189, %168 : index
    %200 = arith.cmpi ugt, %199, %dim_64 : index
    %201 = scf.if %200 -> (memref<?xindex>) {
      %271 = scf.while (%arg0 = %dim_64) : (index) -> index {
        %273 = arith.muli %arg0, %c2 : index
        %274 = arith.cmpi ugt, %199, %273 : index
        scf.condition(%274) %273 : index
      } do {
      ^bb0(%arg0: index):
        scf.yield %arg0 : index
      }
      %272 = memref.realloc %191(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %191 : memref<?xindex>
    }
    %subview_65 = memref.subview %201[%189] [%168] [%c1] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    linalg.fill ins(%c0 : index) outs(%subview_65 : memref<?xindex, strided<[?], offset: ?>>)
    %202 = sparse_tensor.storage_specifier.set %198  pos_mem_sz at 1 with %199 : !sparse_tensor.storage_specifier<#sparse>
    %subview_66 = memref.subview %alloc_57[0] [%171] [1] : memref<?xf64> to memref<?xf64>
    %subview_67 = memref.subview %181[0] [%179] [1] : memref<?xindex> to memref<?xindex>
    %203 = arith.divui %184, %c3 : index
    %subview_68 = memref.subview %alloc_56[%c0] [%203] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %204 = arith.divui %184, %c3 : index
    %subview_69 = memref.subview %alloc_56[%c1] [%204] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %205 = arith.divui %184, %c3 : index
    %subview_70 = memref.subview %alloc_56[%c2] [%205] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %206 = memref.load %subview_67[%c0] : memref<?xindex>
    %207 = memref.load %subview_67[%c1] : memref<?xindex>
    %208 = scf.while (%arg0 = %206) : (index) -> index {
      %271 = arith.cmpi ult, %arg0, %207 : index
      %272 = scf.if %271 -> (i1) {
        %273 = memref.load %subview_68[%206] : memref<?xindex, strided<[?], offset: ?>>
        %274 = memref.load %subview_68[%arg0] : memref<?xindex, strided<[?], offset: ?>>
        %275 = arith.cmpi eq, %273, %274 : index
        scf.yield %275 : i1
      } else {
        scf.yield %false : i1
      }
      scf.condition(%272) %arg0 : index
    } do {
    ^bb0(%arg0: index):
      %271 = arith.addi %arg0, %c1 : index
      scf.yield %271 : index
    }
    %209:8 = scf.while (%arg0 = %206, %arg1 = %208, %arg2 = %201, %arg3 = %alloc_60, %arg4 = %197, %arg5 = %alloc_62, %arg6 = %alloc_63, %arg7 = %202) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
      %271 = arith.cmpi ult, %arg0, %207 : index
      scf.condition(%271) %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
    } do {
    ^bb0(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: memref<?xindex>, %arg6: memref<?xf64>, %arg7: !sparse_tensor.storage_specifier<#sparse>):
      %271 = memref.load %subview_68[%arg0] : memref<?xindex, strided<[?], offset: ?>>
      %272 = scf.while (%arg8 = %arg0) : (index) -> index {
        %275 = arith.cmpi ult, %arg8, %arg1 : index
        %276 = scf.if %275 -> (i1) {
          %277 = memref.load %subview_69[%arg0] : memref<?xindex, strided<[?], offset: ?>>
          %278 = memref.load %subview_69[%arg8] : memref<?xindex, strided<[?], offset: ?>>
          %279 = arith.cmpi eq, %277, %278 : index
          scf.yield %279 : i1
        } else {
          scf.yield %false : i1
        }
        scf.condition(%276) %arg8 : index
      } do {
      ^bb0(%arg8: index):
        %275 = arith.addi %arg8, %c1 : index
        scf.yield %275 : index
      }
      %273:8 = scf.while (%arg8 = %arg0, %arg9 = %272, %arg10 = %arg2, %arg11 = %arg3, %arg12 = %arg4, %arg13 = %arg5, %arg14 = %arg6, %arg15 = %arg7) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
        %275 = arith.cmpi ult, %arg8, %arg1 : index
        scf.condition(%275) %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
      } do {
      ^bb0(%arg8: index, %arg9: index, %arg10: memref<?xindex>, %arg11: memref<?xindex>, %arg12: memref<?xindex>, %arg13: memref<?xindex>, %arg14: memref<?xf64>, %arg15: !sparse_tensor.storage_specifier<#sparse>):
        %275 = memref.load %subview_69[%arg8] : memref<?xindex, strided<[?], offset: ?>>
        %276:6 = scf.for %arg16 = %arg8 to %arg9 step %c1 iter_args(%arg17 = %arg10, %arg18 = %arg11, %arg19 = %arg12, %arg20 = %arg13, %arg21 = %arg14, %arg22 = %arg15) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
          %278 = memref.load %subview_70[%arg16] : memref<?xindex, strided<[?], offset: ?>>
          %279 = memref.load %subview_66[%arg16] : memref<?xf64>
          %280:6 = func.call @"_insert_dense_compressed_compressed_-9223372036854775808_-9223372036854775808_-9223372036854775808_f64_0_0"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %271, %275, %278, %279) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index, index, index, f64) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>)
          scf.yield %280#0, %280#1, %280#2, %280#3, %280#4, %280#5 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
        } {"Emitted from" = "sparse_tensor.foreach"}
        %277:2 = scf.if %true -> (index, index) {
          %278 = scf.while (%arg16 = %arg9) : (index) -> index {
            %279 = arith.cmpi ult, %arg16, %arg1 : index
            %280 = scf.if %279 -> (i1) {
              %281 = memref.load %subview_69[%arg9] : memref<?xindex, strided<[?], offset: ?>>
              %282 = memref.load %subview_69[%arg16] : memref<?xindex, strided<[?], offset: ?>>
              %283 = arith.cmpi eq, %281, %282 : index
              scf.yield %283 : i1
            } else {
              scf.yield %false : i1
            }
            scf.condition(%280) %arg16 : index
          } do {
          ^bb0(%arg16: index):
            %279 = arith.addi %arg16, %c1 : index
            scf.yield %279 : index
          }
          scf.yield %arg9, %278 : index, index
        } else {
          scf.yield %arg8, %arg9 : index, index
        }
        scf.yield %277#0, %277#1, %276#0, %276#1, %276#2, %276#3, %276#4, %276#5 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
      }
      %274:2 = scf.if %true -> (index, index) {
        %275 = scf.while (%arg8 = %arg1) : (index) -> index {
          %276 = arith.cmpi ult, %arg8, %207 : index
          %277 = scf.if %276 -> (i1) {
            %278 = memref.load %subview_68[%arg1] : memref<?xindex, strided<[?], offset: ?>>
            %279 = memref.load %subview_68[%arg8] : memref<?xindex, strided<[?], offset: ?>>
            %280 = arith.cmpi eq, %278, %279 : index
            scf.yield %280 : i1
          } else {
            scf.yield %false : i1
          }
          scf.condition(%277) %arg8 : index
        } do {
        ^bb0(%arg8: index):
          %276 = arith.addi %arg8, %c1 : index
          scf.yield %276 : index
        }
        scf.yield %arg1, %275 : index, index
      } else {
        scf.yield %arg0, %arg1 : index, index
      }
      scf.yield %274#0, %274#1, %273#2, %273#3, %273#4, %273#5, %273#6, %273#7 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
    }
    %210 = sparse_tensor.storage_specifier.get %209#7  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %211 = memref.load %209#2[%c0] : memref<?xindex>
    %212 = scf.for %arg0 = %c1 to %210 step %c1 iter_args(%arg1 = %211) -> (index) {
      %271 = memref.load %209#2[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %209#2[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %213 = sparse_tensor.storage_specifier.get %209#7  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %214 = memref.load %209#4[%c0] : memref<?xindex>
    %215 = scf.for %arg0 = %c1 to %213 step %c1 iter_args(%arg1 = %214) -> (index) {
      %271 = memref.load %209#4[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %209#4[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %alloca_71 = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca_71[%c0] : memref<?xindex>
    memref.store %c0, %alloca_71[%c1] : memref<?xindex>
    memref.store %c0, %alloca_71[%c2] : memref<?xindex>
    %216 = call @createCheckedSparseTensorReader(%4, %alloca_71, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %217 = call @getSparseTensorReaderDimSizes(%216) : (!llvm.ptr) -> memref<?xindex>
    %218 = memref.load %217[%c0] : memref<?xindex>
    %219 = memref.load %217[%c1] : memref<?xindex>
    %220 = memref.load %217[%c2] : memref<?xindex>
    %221 = call @getSparseTensorReaderNSE(%216) : (!llvm.ptr) -> index
    %alloca_72 = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca_72[%c0] : memref<?xindex>
    memref.store %c1, %alloca_72[%c1] : memref<?xindex>
    memref.store %c2, %alloca_72[%c2] : memref<?xindex>
    %222 = arith.muli %221, %c3 : index
    %alloc_73 = memref.alloc(%c2) : memref<?xindex>
    %alloc_74 = memref.alloc(%222) : memref<?xindex>
    %alloc_75 = memref.alloc(%221) : memref<?xf64>
    %223 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse2>
    %224 = sparse_tensor.storage_specifier.set %223  lvl_sz at 0 with %218 : !sparse_tensor.storage_specifier<#sparse2>
    %225 = sparse_tensor.storage_specifier.get %224  pos_mem_sz at 0 : !sparse_tensor.storage_specifier<#sparse2>
    %226 = arith.addi %225, %c1 : index
    %227 = arith.cmpi ugt, %226, %c2 : index
    %228 = scf.if %227 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_73(%c4) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_73 : memref<?xindex>
    }
    memref.store %c0, %228[%225] : memref<?xindex>
    %dim_76 = memref.dim %228, %c0 : memref<?xindex>
    %229 = arith.addi %226, %c1 : index
    %230 = arith.cmpi ugt, %229, %dim_76 : index
    %231 = scf.if %230 -> (memref<?xindex>) {
      %271 = arith.muli %dim_76, %c2 : index
      %272 = memref.realloc %228(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %228 : memref<?xindex>
    }
    memref.store %c0, %231[%226] : memref<?xindex>
    %232 = call @getSparseTensorReaderReadToBuffers0F64(%216, %alloca_72, %alloca_72, %alloc_74, %alloc_75) : (!llvm.ptr, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>) -> i1
    %233 = arith.cmpi eq, %232, %false : i1
    scf.if %233 {
      %271 = arith.index_cast %221 : index to i64
      %272 = math.ctlz %271 : i64
      %273 = arith.subi %c64_i64, %272 : i64
      func.call @_sparse_hybrid_qsort_0_1_2_index_coo_0_f64(%c0, %221, %alloc_74, %alloc_75, %273) : (index, index, memref<?xindex>, memref<?xf64>, i64) -> ()
    }
    memref.store %221, %231[%c1] : memref<?xindex>
    %234 = arith.muli %221, %c3 : index
    call @delSparseTensorReader(%216) : (!llvm.ptr) -> ()
    %alloc_77 = memref.alloc(%c16) : memref<?xindex>
    %alloc_78 = memref.alloc(%c16) : memref<?xindex>
    %alloc_79 = memref.alloc(%c16) : memref<?xindex>
    %alloc_80 = memref.alloc(%c16) : memref<?xindex>
    %alloc_81 = memref.alloc(%c16) : memref<?xf64>
    %235 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#sparse>
    %236 = sparse_tensor.storage_specifier.set %235  lvl_sz at 0 with %218 : !sparse_tensor.storage_specifier<#sparse>
    %237 = sparse_tensor.storage_specifier.set %236  lvl_sz at 1 with %219 : !sparse_tensor.storage_specifier<#sparse>
    %238 = sparse_tensor.storage_specifier.get %237  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %239 = arith.addi %238, %c1 : index
    %240 = arith.cmpi ugt, %239, %c16 : index
    %241 = scf.if %240 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_77(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_77 : memref<?xindex>
    }
    memref.store %c0, %241[%238] : memref<?xindex>
    %242 = sparse_tensor.storage_specifier.set %237  pos_mem_sz at 1 with %239 : !sparse_tensor.storage_specifier<#sparse>
    %243 = sparse_tensor.storage_specifier.set %242  lvl_sz at 2 with %220 : !sparse_tensor.storage_specifier<#sparse>
    %244 = sparse_tensor.storage_specifier.get %243  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %245 = arith.addi %244, %c1 : index
    %246 = arith.cmpi ugt, %245, %c16 : index
    %247 = scf.if %246 -> (memref<?xindex>) {
      %271 = memref.realloc %alloc_79(%c32) : memref<?xindex> to memref<?xindex>
      scf.yield %271 : memref<?xindex>
    } else {
      scf.yield %alloc_79 : memref<?xindex>
    }
    memref.store %c0, %247[%244] : memref<?xindex>
    %248 = sparse_tensor.storage_specifier.set %243  pos_mem_sz at 2 with %245 : !sparse_tensor.storage_specifier<#sparse>
    %dim_82 = memref.dim %241, %c0 : memref<?xindex>
    %249 = arith.addi %239, %218 : index
    %250 = arith.cmpi ugt, %249, %dim_82 : index
    %251 = scf.if %250 -> (memref<?xindex>) {
      %271 = scf.while (%arg0 = %dim_82) : (index) -> index {
        %273 = arith.muli %arg0, %c2 : index
        %274 = arith.cmpi ugt, %249, %273 : index
        scf.condition(%274) %273 : index
      } do {
      ^bb0(%arg0: index):
        scf.yield %arg0 : index
      }
      %272 = memref.realloc %241(%271) : memref<?xindex> to memref<?xindex>
      scf.yield %272 : memref<?xindex>
    } else {
      scf.yield %241 : memref<?xindex>
    }
    %subview_83 = memref.subview %251[%239] [%218] [%c1] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    linalg.fill ins(%c0 : index) outs(%subview_83 : memref<?xindex, strided<[?], offset: ?>>)
    %252 = sparse_tensor.storage_specifier.set %248  pos_mem_sz at 1 with %249 : !sparse_tensor.storage_specifier<#sparse>
    %subview_84 = memref.subview %alloc_75[0] [%221] [1] : memref<?xf64> to memref<?xf64>
    %subview_85 = memref.subview %231[0] [%229] [1] : memref<?xindex> to memref<?xindex>
    %253 = arith.divui %234, %c3 : index
    %subview_86 = memref.subview %alloc_74[%c0] [%253] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %254 = arith.divui %234, %c3 : index
    %subview_87 = memref.subview %alloc_74[%c1] [%254] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %255 = arith.divui %234, %c3 : index
    %subview_88 = memref.subview %alloc_74[%c2] [%255] [%c3] : memref<?xindex> to memref<?xindex, strided<[?], offset: ?>>
    %256 = memref.load %subview_85[%c0] : memref<?xindex>
    %257 = memref.load %subview_85[%c1] : memref<?xindex>
    %258 = scf.while (%arg0 = %256) : (index) -> index {
      %271 = arith.cmpi ult, %arg0, %257 : index
      %272 = scf.if %271 -> (i1) {
        %273 = memref.load %subview_86[%256] : memref<?xindex, strided<[?], offset: ?>>
        %274 = memref.load %subview_86[%arg0] : memref<?xindex, strided<[?], offset: ?>>
        %275 = arith.cmpi eq, %273, %274 : index
        scf.yield %275 : i1
      } else {
        scf.yield %false : i1
      }
      scf.condition(%272) %arg0 : index
    } do {
    ^bb0(%arg0: index):
      %271 = arith.addi %arg0, %c1 : index
      scf.yield %271 : index
    }
    %259:8 = scf.while (%arg0 = %256, %arg1 = %258, %arg2 = %251, %arg3 = %alloc_78, %arg4 = %247, %arg5 = %alloc_80, %arg6 = %alloc_81, %arg7 = %252) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
      %271 = arith.cmpi ult, %arg0, %257 : index
      scf.condition(%271) %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
    } do {
    ^bb0(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: memref<?xindex>, %arg6: memref<?xf64>, %arg7: !sparse_tensor.storage_specifier<#sparse>):
      %271 = memref.load %subview_86[%arg0] : memref<?xindex, strided<[?], offset: ?>>
      %272 = scf.while (%arg8 = %arg0) : (index) -> index {
        %275 = arith.cmpi ult, %arg8, %arg1 : index
        %276 = scf.if %275 -> (i1) {
          %277 = memref.load %subview_87[%arg0] : memref<?xindex, strided<[?], offset: ?>>
          %278 = memref.load %subview_87[%arg8] : memref<?xindex, strided<[?], offset: ?>>
          %279 = arith.cmpi eq, %277, %278 : index
          scf.yield %279 : i1
        } else {
          scf.yield %false : i1
        }
        scf.condition(%276) %arg8 : index
      } do {
      ^bb0(%arg8: index):
        %275 = arith.addi %arg8, %c1 : index
        scf.yield %275 : index
      }
      %273:8 = scf.while (%arg8 = %arg0, %arg9 = %272, %arg10 = %arg2, %arg11 = %arg3, %arg12 = %arg4, %arg13 = %arg5, %arg14 = %arg6, %arg15 = %arg7) : (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> (index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
        %275 = arith.cmpi ult, %arg8, %arg1 : index
        scf.condition(%275) %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
      } do {
      ^bb0(%arg8: index, %arg9: index, %arg10: memref<?xindex>, %arg11: memref<?xindex>, %arg12: memref<?xindex>, %arg13: memref<?xindex>, %arg14: memref<?xf64>, %arg15: !sparse_tensor.storage_specifier<#sparse>):
        %275 = memref.load %subview_87[%arg8] : memref<?xindex, strided<[?], offset: ?>>
        %276:6 = scf.for %arg16 = %arg8 to %arg9 step %c1 iter_args(%arg17 = %arg10, %arg18 = %arg11, %arg19 = %arg12, %arg20 = %arg13, %arg21 = %arg14, %arg22 = %arg15) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) {
          %278 = memref.load %subview_88[%arg16] : memref<?xindex, strided<[?], offset: ?>>
          %279 = memref.load %subview_84[%arg16] : memref<?xf64>
          %280:6 = func.call @"_insert_dense_compressed_compressed_-9223372036854775808_-9223372036854775808_-9223372036854775808_f64_0_0"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %271, %275, %278, %279) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, index, index, index, f64) -> (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>)
          scf.yield %280#0, %280#1, %280#2, %280#3, %280#4, %280#5 : memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
        } {"Emitted from" = "sparse_tensor.foreach"}
        %277:2 = scf.if %true -> (index, index) {
          %278 = scf.while (%arg16 = %arg9) : (index) -> index {
            %279 = arith.cmpi ult, %arg16, %arg1 : index
            %280 = scf.if %279 -> (i1) {
              %281 = memref.load %subview_87[%arg9] : memref<?xindex, strided<[?], offset: ?>>
              %282 = memref.load %subview_87[%arg16] : memref<?xindex, strided<[?], offset: ?>>
              %283 = arith.cmpi eq, %281, %282 : index
              scf.yield %283 : i1
            } else {
              scf.yield %false : i1
            }
            scf.condition(%280) %arg16 : index
          } do {
          ^bb0(%arg16: index):
            %279 = arith.addi %arg16, %c1 : index
            scf.yield %279 : index
          }
          scf.yield %arg9, %278 : index, index
        } else {
          scf.yield %arg8, %arg9 : index, index
        }
        scf.yield %277#0, %277#1, %276#0, %276#1, %276#2, %276#3, %276#4, %276#5 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
      }
      %274:2 = scf.if %true -> (index, index) {
        %275 = scf.while (%arg8 = %arg1) : (index) -> index {
          %276 = arith.cmpi ult, %arg8, %257 : index
          %277 = scf.if %276 -> (i1) {
            %278 = memref.load %subview_86[%arg1] : memref<?xindex, strided<[?], offset: ?>>
            %279 = memref.load %subview_86[%arg8] : memref<?xindex, strided<[?], offset: ?>>
            %280 = arith.cmpi eq, %278, %279 : index
            scf.yield %280 : i1
          } else {
            scf.yield %false : i1
          }
          scf.condition(%277) %arg8 : index
        } do {
        ^bb0(%arg8: index):
          %276 = arith.addi %arg8, %c1 : index
          scf.yield %276 : index
        }
        scf.yield %arg1, %275 : index, index
      } else {
        scf.yield %arg0, %arg1 : index, index
      }
      scf.yield %274#0, %274#1, %273#2, %273#3, %273#4, %273#5, %273#6, %273#7 : index, index, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>
    }
    %260 = sparse_tensor.storage_specifier.get %259#7  pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#sparse>
    %261 = memref.load %259#2[%c0] : memref<?xindex>
    %262 = scf.for %arg0 = %c1 to %260 step %c1 iter_args(%arg1 = %261) -> (index) {
      %271 = memref.load %259#2[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %259#2[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %263 = sparse_tensor.storage_specifier.get %259#7  pos_mem_sz at 2 : !sparse_tensor.storage_specifier<#sparse>
    %264 = memref.load %259#4[%c0] : memref<?xindex>
    %265 = scf.for %arg0 = %c1 to %263 step %c1 iter_args(%arg1 = %264) -> (index) {
      %271 = memref.load %259#4[%arg0] : memref<?xindex>
      %272 = arith.cmpi eq, %271, %c0 : index
      %273 = scf.if %272 -> (index) {
        memref.store %arg1, %259#4[%arg0] : memref<?xindex>
        scf.yield %arg1 : index
      } else {
        scf.yield %271 : index
      }
      scf.yield %273 : index
    }
    %266 = call @dmrg_expectation_value(%48#2, %48#3, %48#4, %48#5, %48#6, %48#7, %98#2, %98#3, %98#4, %98#5, %98#6, %98#7, %156#2, %156#3, %156#4, %156#5, %156#6, %156#7, %156#8, %156#9, %209#2, %209#3, %209#4, %209#5, %209#6, %209#7, %259#2, %259#3, %259#4, %259#5, %259#6, %259#7) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> memref<f64>
    %267 = memref.load %266[] : memref<f64>
    call @printF64(%267) : (f64) -> ()
    memref.dealloc %266 : memref<f64>
    vector.print str "\n"
    %268 = call @rtclock() : () -> f64
    scf.for %arg0 = %c0 to %c100 step %c1 {
      %271 = func.call @dmrg_expectation_value(%48#2, %48#3, %48#4, %48#5, %48#6, %48#7, %98#2, %98#3, %98#4, %98#5, %98#6, %98#7, %156#2, %156#3, %156#4, %156#5, %156#6, %156#7, %156#8, %156#9, %209#2, %209#3, %209#4, %209#5, %209#6, %209#7, %259#2, %259#3, %259#4, %259#5, %259#6, %259#7) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse1>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier<#sparse>) -> memref<f64>
      memref.dealloc %271 : memref<f64>
    }
    %269 = call @rtclock() : () -> f64
    %270 = arith.subf %269, %268 : f64
    call @printF64(%270) : (f64) -> ()
    return
  }
}

