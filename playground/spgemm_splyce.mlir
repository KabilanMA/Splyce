#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func private @delSparseTensor(!llvm.ptr)
  func.func private @delSparseTensorReader(!llvm.ptr)
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @getSparseTensorReaderDimSizes(!llvm.ptr) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @createCheckedSparseTensorReader(!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseLvlSize(!llvm.ptr, index) -> index
  func.func private @rtclock() -> f64
  func.func private @printNewline()
  func.func private @printF64(f64)
  llvm.func @fopen(!llvm.ptr, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32 attributes {llvm.emit_c_interface}
  llvm.func @fclose(!llvm.ptr) -> i32 attributes {llvm.emit_c_interface}
  llvm.mlir.global internal constant @fileA("tensor_A.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fileB("tensor_B.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @benchmarkFile("benchmark\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @mode_w("w\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_time("%f\0A\00") {addr_space = 0 : i32}
  func.func @spgemm(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> tensor<?x?xf64> {
    %cst = arith.constant dense<0.000000e+00> : vector<4xf64>
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = call @sparseLvlSize(%arg0, %c0) : (!llvm.ptr, index) -> index
    %1 = call @sparseLvlSize(%arg1, %c0) : (!llvm.ptr, index) -> index
    %2 = bufferization.alloc_tensor(%0, %1) : tensor<?x?xf64>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?xf64>
    %4 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %5 = call @sparseValuesF64(%arg1) : (!llvm.ptr) -> memref<?xf64>
    %6 = bufferization.to_buffer %3 : tensor<?x?xf64> to memref<?x?xf64>
    %7 = call @sparseLvlSize(%arg0, %c0) : (!llvm.ptr, index) -> index
    %8 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %9 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %10 = call @sparseLvlSize(%arg1, %c0) : (!llvm.ptr, index) -> index
    %11 = call @sparsePositions0(%arg1, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %12 = call @sparseCoordinates0(%arg1, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    scf.for %arg2 = %c0 to %7 step %c1 {
      %14 = arith.addi %arg2, %c1 : index
      scf.for %arg3 = %c0 to %10 step %c1 {
        %15 = memref.load %6[%arg2, %arg3] : memref<?x?xf64>
        %16 = memref.load %8[%arg2] : memref<?xindex>
        %17 = memref.load %8[%14] : memref<?xindex>
        %18 = memref.load %11[%arg3] : memref<?xindex>
        %19 = arith.addi %arg3, %c1 : index
        %20 = memref.load %11[%19] : memref<?xindex>
        %21:3 = scf.while (%arg4 = %16, %arg5 = %18, %arg6 = %15) : (index, index, f64) -> (index, index, f64) {
          %23 = arith.addi %arg4, %c4 : index
          %24 = arith.cmpi ule, %23, %17 : index
          %25 = arith.addi %arg5, %c4 : index
          %26 = arith.cmpi ule, %25, %20 : index
          %27 = arith.andi %24, %26 : i1
          scf.condition(%27) %arg4, %arg5, %arg6 : index, index, f64
        } do {
        ^bb0(%arg4: index, %arg5: index, %arg6: f64):
          %23 = vector.load %9[%arg4] : memref<?xindex>, vector<4xindex>
          %24 = vector.load %4[%arg4] : memref<?xf64>, vector<4xf64>
          %25 = vector.load %12[%arg5] : memref<?xindex>, vector<4xindex>
          %26 = vector.load %5[%arg5] : memref<?xf64>, vector<4xf64>
          %27 = vector.extract %23[0] : index from vector<4xindex>
          %28 = vector.extract %23[1] : index from vector<4xindex>
          %29 = vector.extract %23[2] : index from vector<4xindex>
          %30 = vector.extract %23[3] : index from vector<4xindex>
          %31 = vector.extract %25[0] : index from vector<4xindex>
          %32 = vector.extract %25[1] : index from vector<4xindex>
          %33 = vector.extract %25[2] : index from vector<4xindex>
          %34 = vector.extract %25[3] : index from vector<4xindex>
          %35 = vector.broadcast %27 : index to vector<4xindex>
          %36 = arith.cmpi eq, %35, %25 : vector<4xindex>
          %37 = vector.broadcast %28 : index to vector<4xindex>
          %38 = arith.cmpi eq, %37, %25 : vector<4xindex>
          %39 = vector.broadcast %29 : index to vector<4xindex>
          %40 = arith.cmpi eq, %39, %25 : vector<4xindex>
          %41 = vector.broadcast %30 : index to vector<4xindex>
          %42 = arith.cmpi eq, %41, %25 : vector<4xindex>
          %43 = arith.select %36, %26, %cst : vector<4xi1>, vector<4xf64>
          %44 = vector.reduction <add>, %43 : vector<4xf64> into f64
          %45 = arith.select %38, %26, %cst : vector<4xi1>, vector<4xf64>
          %46 = vector.reduction <add>, %45 : vector<4xf64> into f64
          %47 = arith.select %40, %26, %cst : vector<4xi1>, vector<4xf64>
          %48 = vector.reduction <add>, %47 : vector<4xf64> into f64
          %49 = arith.select %42, %26, %cst : vector<4xi1>, vector<4xf64>
          %50 = vector.reduction <add>, %49 : vector<4xf64> into f64
          %51 = vector.extract %24[0] : f64 from vector<4xf64>
          %52 = vector.extract %24[1] : f64 from vector<4xf64>
          %53 = vector.extract %24[2] : f64 from vector<4xf64>
          %54 = vector.extract %24[3] : f64 from vector<4xf64>
          %55 = arith.mulf %51, %44 : f64
          %56 = arith.mulf %52, %46 : f64
          %57 = arith.mulf %53, %48 : f64
          %58 = arith.mulf %54, %50 : f64
          %59 = arith.addf %55, %56 : f64
          %60 = arith.addf %57, %58 : f64
          %61 = arith.addf %59, %60 : f64
          %62 = arith.addf %arg6, %61 : f64
          %63 = arith.minui %30, %34 : index
          %64 = arith.cmpi ule, %27, %63 : index
          %65 = arith.cmpi ule, %28, %63 : index
          %66 = arith.cmpi ule, %29, %63 : index
          %67 = arith.cmpi ule, %30, %63 : index
          %68 = arith.select %64, %c1, %c0 : index
          %69 = arith.select %65, %c1, %c0 : index
          %70 = arith.select %66, %c1, %c0 : index
          %71 = arith.select %67, %c1, %c0 : index
          %72 = arith.addi %68, %69 : index
          %73 = arith.addi %70, %71 : index
          %74 = arith.addi %72, %73 : index
          %75 = arith.addi %arg4, %74 : index
          %76 = arith.cmpi ule, %31, %63 : index
          %77 = arith.cmpi ule, %32, %63 : index
          %78 = arith.cmpi ule, %33, %63 : index
          %79 = arith.cmpi ule, %34, %63 : index
          %80 = arith.select %76, %c1, %c0 : index
          %81 = arith.select %77, %c1, %c0 : index
          %82 = arith.select %78, %c1, %c0 : index
          %83 = arith.select %79, %c1, %c0 : index
          %84 = arith.addi %80, %81 : index
          %85 = arith.addi %82, %83 : index
          %86 = arith.addi %84, %85 : index
          %87 = arith.addi %arg5, %86 : index
          scf.yield %75, %87, %62 : index, index, f64
        }
        %22:3 = scf.while (%arg4 = %21#0, %arg5 = %21#1, %arg6 = %21#2) : (index, index, f64) -> (index, index, f64) {
          %23 = arith.cmpi ult, %arg4, %17 : index
          %24 = arith.cmpi ult, %arg5, %20 : index
          %25 = arith.andi %23, %24 : i1
          scf.condition(%25) %arg4, %arg5, %arg6 : index, index, f64
        } do {
        ^bb0(%arg4: index, %arg5: index, %arg6: f64):
          %23 = memref.load %9[%arg4] : memref<?xindex>
          %24 = memref.load %12[%arg5] : memref<?xindex>
          %25 = arith.cmpi ult, %24, %23 : index
          %26 = arith.select %25, %24, %23 : index
          %27 = arith.cmpi eq, %23, %26 : index
          %28 = arith.cmpi eq, %24, %26 : index
          %29 = arith.andi %27, %28 : i1
          %30 = scf.if %29 -> (f64) {
            %37 = memref.load %4[%arg4] : memref<?xf64>
            %38 = memref.load %5[%arg5] : memref<?xf64>
            %39 = arith.mulf %37, %38 : f64
            %40 = arith.addf %arg6, %39 : f64
            scf.yield %40 : f64
          } else {
            scf.yield %arg6 : f64
          }
          %31 = arith.cmpi eq, %23, %26 : index
          %32 = arith.addi %arg4, %c1 : index
          %33 = arith.select %31, %32, %arg4 : index
          %34 = arith.cmpi eq, %24, %26 : index
          %35 = arith.addi %arg5, %c1 : index
          %36 = arith.select %34, %35, %arg5 : index
          scf.yield %33, %36, %30 : index, index, f64
        } attributes {"Emitted from" = "linalg.generic", splyce.scalar_fallback}
        memref.store %22#2, %6[%arg2, %arg3] : memref<?x?xf64>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %13 = bufferization.to_tensor %6 restrict : memref<?x?xf64> to tensor<?x?xf64>
    return %13 : tensor<?x?xf64>
  }
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c262144_i64 = arith.constant 262144 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @fmt_time : !llvm.ptr
    %1 = llvm.mlir.addressof @mode_w : !llvm.ptr
    %2 = llvm.mlir.addressof @benchmarkFile : !llvm.ptr
    %c25 = arith.constant 25 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %3 = llvm.mlir.addressof @fileA : !llvm.ptr
    %4 = llvm.mlir.addressof @fileB : !llvm.ptr
    %c2 = arith.constant 2 : index
    %alloca = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca[%c0] : memref<?xindex>
    memref.store %c0, %alloca[%c1] : memref<?xindex>
    %5 = call @createCheckedSparseTensorReader(%3, %alloca, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %6 = call @getSparseTensorReaderDimSizes(%5) : (!llvm.ptr) -> memref<?xindex>
    %alloca_0 = memref.alloca(%c2) : memref<?xi64>
    memref.store %c65536_i64, %alloca_0[%c0] : memref<?xi64>
    memref.store %c262144_i64, %alloca_0[%c1] : memref<?xi64>
    %alloca_1 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_1[%c0] : memref<?xindex>
    memref.store %c1, %alloca_1[%c1] : memref<?xindex>
    %7 = call @newSparseTensor(%6, %6, %alloca_0, %alloca_1, %alloca_1, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %5) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%5) : (!llvm.ptr) -> ()
    %alloca_2 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_2[%c0] : memref<?xindex>
    memref.store %c0, %alloca_2[%c1] : memref<?xindex>
    %8 = call @createCheckedSparseTensorReader(%4, %alloca_2, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %9 = call @getSparseTensorReaderDimSizes(%8) : (!llvm.ptr) -> memref<?xindex>
    %10 = memref.load %9[%c0] : memref<?xindex>
    %11 = memref.load %9[%c1] : memref<?xindex>
    %alloca_3 = memref.alloca(%c2) : memref<?xi64>
    memref.store %c65536_i64, %alloca_3[%c0] : memref<?xi64>
    memref.store %c262144_i64, %alloca_3[%c1] : memref<?xi64>
    %alloca_4 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c1, %alloca_4[%c0] : memref<?xindex>
    memref.store %c0, %alloca_4[%c1] : memref<?xindex>
    %alloca_5 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c1, %alloca_5[%c0] : memref<?xindex>
    memref.store %c0, %alloca_5[%c1] : memref<?xindex>
    %alloca_6 = memref.alloca(%c2) : memref<?xindex>
    memref.store %11, %alloca_6[%c0] : memref<?xindex>
    memref.store %10, %alloca_6[%c1] : memref<?xindex>
    %12 = call @newSparseTensor(%9, %alloca_6, %alloca_3, %alloca_4, %alloca_5, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %8) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%8) : (!llvm.ptr) -> ()
    %13 = call @spgemm(%7, %12) : (!llvm.ptr, !llvm.ptr) -> tensor<?x?xf64>
    bufferization.dealloc_tensor %13 : tensor<?x?xf64>
    %alloc = memref.alloc() : memref<25xf64>
    scf.for %arg0 = %c0 to %c25 step %c1 {
      %16 = func.call @rtclock() : () -> f64
      %17 = func.call @spgemm(%7, %12) : (!llvm.ptr, !llvm.ptr) -> tensor<?x?xf64>
      %18 = func.call @rtclock() : () -> f64
      %19 = arith.subf %18, %16 : f64
      func.call @printF64(%19) : (f64) -> ()
      func.call @printNewline() : () -> ()
      memref.store %19, %alloc[%arg0] : memref<25xf64>
      bufferization.dealloc_tensor %17 : tensor<?x?xf64>
    }
    %14 = llvm.call @fopen(%2, %1) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    scf.for %arg0 = %c0 to %c25 step %c1 {
      %16 = memref.load %alloc[%arg0] : memref<25xf64>
      %17 = llvm.call @fprintf(%14, %0, %16) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
    }
    %15 = llvm.call @fclose(%14) : (!llvm.ptr) -> i32
    memref.dealloc %alloc : memref<25xf64>
    call @delSparseTensor(%7) : (!llvm.ptr) -> ()
    call @delSparseTensor(%12) : (!llvm.ptr) -> ()
    return
  }
}

