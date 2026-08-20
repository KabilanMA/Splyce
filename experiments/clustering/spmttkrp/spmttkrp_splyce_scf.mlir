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
  llvm.func @fopen(!llvm.ptr, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32 attributes {llvm.emit_c_interface}
  llvm.func @fclose(!llvm.ptr) -> i32 attributes {llvm.emit_c_interface}
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {llvm.emit_c_interface}
  llvm.mlir.global internal constant @fileB("tensor_B.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fileC("tensor_C.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fileD("tensor_D.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @benchmarkFile("benchmark\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @mode_w("w\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_time("%f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_exec_time("Execution time: %f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_scalar_count("Scalar elements processed: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_simd_count("SIMD elements processed: %ld\0A\00") {addr_space = 0 : i32}
  memref.global "private" @scalar_epilogue_count : memref<i64> = dense<0>
  memref.global "private" @simd_fastlane_count : memref<i64> = dense<0>
  func.func @spmttkrp(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> tensor<?x?xf64> {
    %cst = arith.constant dense<0.000000e+00> : vector<4xf64>
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // Scalar-epilogue and SIMD-fastlane element counters: both reset at
    // the start of every call to @spmttkrp, incremented once per
    // coiteration step in their respective "do" regions below (the
    // scalar_fallback one, and the vectorized one right above it), and
    // printed separately just before returning.
    %scalar_counter_ref = memref.get_global @scalar_epilogue_count : memref<i64>
    %simd_counter_ref = memref.get_global @simd_fastlane_count : memref<i64>
    %scalar_zero_i64 = arith.constant 0 : i64
    %scalar_one_i64 = arith.constant 1 : i64
    memref.store %scalar_zero_i64, %scalar_counter_ref[] : memref<i64>
    memref.store %scalar_zero_i64, %simd_counter_ref[] : memref<i64>
    %0 = call @sparseLvlSize(%arg0, %c0) : (!llvm.ptr, index) -> index
    %1 = call @sparseLvlSize(%arg0, %c1) : (!llvm.ptr, index) -> index
    %2 = bufferization.alloc_tensor(%0, %1) : tensor<?x?xf64>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?xf64>
    %4 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %5 = call @sparseValuesF64(%arg1) : (!llvm.ptr) -> memref<?xf64>
    %6 = call @sparseValuesF64(%arg2) : (!llvm.ptr) -> memref<?xf64>
    %7 = bufferization.to_buffer %3 : tensor<?x?xf64> to memref<?x?xf64>
    %8 = call @sparseLvlSize(%arg0, %c0) : (!llvm.ptr, index) -> index
    %9 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %10 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %11 = call @sparsePositions0(%arg0, %c2) : (!llvm.ptr, index) -> memref<?xindex>
    %12 = call @sparseCoordinates0(%arg0, %c2) : (!llvm.ptr, index) -> memref<?xindex>
    %13 = call @sparsePositions0(%arg1, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %14 = call @sparseCoordinates0(%arg1, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %15 = call @sparsePositions0(%arg2, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %16 = call @sparseCoordinates0(%arg2, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    scf.for %arg3 = %c0 to %8 step %c1 {
      %18 = memref.load %9[%arg3] : memref<?xindex>
      %19 = arith.addi %arg3, %c1 : index
      %20 = memref.load %9[%19] : memref<?xindex>
      scf.for %arg4 = %18 to %20 step %c1 {
        %21 = memref.load %10[%arg4] : memref<?xindex>
        %22 = memref.load %7[%arg3, %21] : memref<?x?xf64>
        %23 = memref.load %11[%arg4] : memref<?xindex>
        %24 = arith.addi %arg4, %c1 : index
        %25 = memref.load %11[%24] : memref<?xindex>
        %26 = arith.addi %21, %c1 : index
        %27 = scf.for %arg5 = %23 to %25 step %c1 iter_args(%arg6 = %22) -> (f64) {
          %28 = memref.load %12[%arg5] : memref<?xindex>
          %29 = memref.load %4[%arg5] : memref<?xf64>
          %30 = memref.load %13[%28] : memref<?xindex>
          %31 = arith.addi %28, %c1 : index
          %32 = memref.load %13[%31] : memref<?xindex>
          %33 = memref.load %15[%21] : memref<?xindex>
          %34 = memref.load %15[%26] : memref<?xindex>
          %35:3 = scf.while (%arg7 = %30, %arg8 = %33, %arg9 = %arg6) : (index, index, f64) -> (index, index, f64) {
            %37 = arith.addi %arg7, %c4 : index
            %38 = arith.cmpi ule, %37, %32 : index
            %39 = arith.addi %arg8, %c4 : index
            %40 = arith.cmpi ule, %39, %34 : index
            %41 = arith.andi %38, %40 : i1
            scf.condition(%41) %arg7, %arg8, %arg9 : index, index, f64
          } do {
          ^bb0(%arg7: index, %arg8: index, %arg9: f64):
            %37 = vector.load %14[%arg7] : memref<?xindex>, vector<4xindex>
            %38 = vector.load %5[%arg7] : memref<?xf64>, vector<4xf64>
            %39 = vector.load %16[%arg8] : memref<?xindex>, vector<4xindex>
            %40 = vector.load %6[%arg8] : memref<?xf64>, vector<4xf64>
            %41 = vector.extract %37[0] : index from vector<4xindex>
            %42 = vector.extract %37[1] : index from vector<4xindex>
            %43 = vector.extract %37[2] : index from vector<4xindex>
            %44 = vector.extract %37[3] : index from vector<4xindex>
            %45 = vector.extract %39[0] : index from vector<4xindex>
            %46 = vector.extract %39[1] : index from vector<4xindex>
            %47 = vector.extract %39[2] : index from vector<4xindex>
            %48 = vector.extract %39[3] : index from vector<4xindex>
            %49 = vector.broadcast %41 : index to vector<4xindex>
            %50 = arith.cmpi eq, %49, %39 : vector<4xindex>
            %51 = vector.broadcast %42 : index to vector<4xindex>
            %52 = arith.cmpi eq, %51, %39 : vector<4xindex>
            %53 = vector.broadcast %43 : index to vector<4xindex>
            %54 = arith.cmpi eq, %53, %39 : vector<4xindex>
            %55 = vector.broadcast %44 : index to vector<4xindex>
            %56 = arith.cmpi eq, %55, %39 : vector<4xindex>
            %57 = arith.select %50, %40, %cst : vector<4xi1>, vector<4xf64>
            %58 = vector.reduction <add>, %57 : vector<4xf64> into f64
            %59 = arith.select %52, %40, %cst : vector<4xi1>, vector<4xf64>
            %60 = vector.reduction <add>, %59 : vector<4xf64> into f64
            %61 = arith.select %54, %40, %cst : vector<4xi1>, vector<4xf64>
            %62 = vector.reduction <add>, %61 : vector<4xf64> into f64
            %63 = arith.select %56, %40, %cst : vector<4xi1>, vector<4xf64>
            %64 = vector.reduction <add>, %63 : vector<4xf64> into f64
            %65 = vector.extract %38[0] : f64 from vector<4xf64>
            %66 = vector.extract %38[1] : f64 from vector<4xf64>
            %67 = vector.extract %38[2] : f64 from vector<4xf64>
            %68 = vector.extract %38[3] : f64 from vector<4xf64>
            %69 = arith.mulf %65, %58 : f64
            %70 = arith.mulf %69, %29 : f64
            %71 = arith.mulf %66, %60 : f64
            %72 = arith.mulf %71, %29 : f64
            %73 = arith.mulf %67, %62 : f64
            %74 = arith.mulf %73, %29 : f64
            %75 = arith.mulf %68, %64 : f64
            %76 = arith.mulf %75, %29 : f64
            %77 = arith.addf %70, %72 : f64
            %78 = arith.addf %74, %76 : f64
            %79 = arith.addf %77, %78 : f64
            %80 = arith.addf %arg9, %79 : f64
            %81 = arith.minui %44, %48 : index
            %82 = arith.cmpi ule, %41, %81 : index
            %83 = arith.cmpi ule, %42, %81 : index
            %84 = arith.cmpi ule, %43, %81 : index
            %85 = arith.cmpi ule, %44, %81 : index
            %86 = arith.select %82, %c1, %c0 : index
            %87 = arith.select %83, %c1, %c0 : index
            %88 = arith.select %84, %c1, %c0 : index
            %89 = arith.select %85, %c1, %c0 : index
            %90 = arith.addi %86, %87 : index
            %91 = arith.addi %88, %89 : index
            %92 = arith.addi %90, %91 : index
            %93 = arith.addi %arg7, %92 : index
            %94 = arith.cmpi ule, %45, %81 : index
            %95 = arith.cmpi ule, %46, %81 : index
            %96 = arith.cmpi ule, %47, %81 : index
            %97 = arith.cmpi ule, %48, %81 : index
            %98 = arith.select %94, %c1, %c0 : index
            %99 = arith.select %95, %c1, %c0 : index
            %100 = arith.select %96, %c1, %c0 : index
            %101 = arith.select %97, %c1, %c0 : index
            %102 = arith.addi %98, %99 : index
            %103 = arith.addi %100, %101 : index
            %104 = arith.addi %102, %103 : index
            %105 = arith.addi %arg8, %104 : index
            // %92 and %104 are this iteration's arg7-side/arg8-side
            // (C-side/D-side) advance counts (0-4 each, from the
            // popcount-style select+add above) — their sum is exactly how
            // many index-stream elements this SIMD iteration consumed,
            // the same unit the scalar counter uses.
            %simd_elems_idx = arith.addi %92, %104 : index
            %simd_elems_i64 = arith.index_cast %simd_elems_idx : index to i64
            %simd_count_val = memref.load %simd_counter_ref[] : memref<i64>
            %simd_count_next = arith.addi %simd_count_val, %simd_elems_i64 : i64
            memref.store %simd_count_next, %simd_counter_ref[] : memref<i64>
            scf.yield %93, %105, %80 : index, index, f64
          }
          %36:3 = scf.while (%arg7 = %35#0, %arg8 = %35#1, %arg9 = %35#2) : (index, index, f64) -> (index, index, f64) {
            %37 = arith.cmpi ult, %arg7, %32 : index
            %38 = arith.cmpi ult, %arg8, %34 : index
            %39 = arith.andi %37, %38 : i1
            scf.condition(%39) %arg7, %arg8, %arg9 : index, index, f64
          } do {
          ^bb0(%arg7: index, %arg8: index, %arg9: f64):
            %37 = memref.load %14[%arg7] : memref<?xindex>
            %38 = memref.load %16[%arg8] : memref<?xindex>
            %39 = arith.cmpi ult, %38, %37 : index
            %40 = arith.select %39, %38, %37 : index
            %41 = arith.cmpi eq, %37, %40 : index
            %42 = arith.cmpi eq, %38, %40 : index
            // This iteration consumes 1 element from C (%14/%5, arg7
            // side) if %41, and 1 from D (%16/%6, arg8 side) if %42 —
            // never zero (at least one side always equals the min), and
            // 2 on an actual index match (both true), same asymmetric-
            // consumption logic the SIMD side's %92/%104 already track —
            // so the two counters stay apples-to-apples instead of
            // scalar just counting loop steps.
            %scalar_c_adv = arith.select %41, %scalar_one_i64, %scalar_zero_i64 : i64
            %scalar_d_adv = arith.select %42, %scalar_one_i64, %scalar_zero_i64 : i64
            %scalar_elems_this_iter = arith.addi %scalar_c_adv, %scalar_d_adv : i64
            %scalar_count_val = memref.load %scalar_counter_ref[] : memref<i64>
            %scalar_count_next = arith.addi %scalar_count_val, %scalar_elems_this_iter : i64
            memref.store %scalar_count_next, %scalar_counter_ref[] : memref<i64>
            %43 = arith.andi %41, %42 : i1
            %44 = scf.if %43 -> (f64) {
              %51 = memref.load %5[%arg7] : memref<?xf64>
              %52 = arith.mulf %29, %51 : f64
              %53 = memref.load %6[%arg8] : memref<?xf64>
              %54 = arith.mulf %52, %53 : f64
              %55 = arith.addf %arg9, %54 : f64
              scf.yield %55 : f64
            } else {
              scf.yield %arg9 : f64
            }
            %45 = arith.cmpi eq, %37, %40 : index
            %46 = arith.addi %arg7, %c1 : index
            %47 = arith.select %45, %46, %arg7 : index
            %48 = arith.cmpi eq, %38, %40 : index
            %49 = arith.addi %arg8, %c1 : index
            %50 = arith.select %48, %49, %arg8 : index
            scf.yield %47, %50, %44 : index, index, f64
          } attributes {"Emitted from" = "linalg.generic", splyce.scalar_fallback}
          scf.yield %36#2 : f64
        } {"Emitted from" = "linalg.generic"}
        memref.store %27, %7[%arg3, %21] : memref<?x?xf64>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %17 = bufferization.to_tensor %7 restrict : memref<?x?xf64> to tensor<?x?xf64>
    %scalar_final_count = memref.load %scalar_counter_ref[] : memref<i64>
    %simd_final_count = memref.load %simd_counter_ref[] : memref<i64>
    %fmt_scalar_count_ptr = llvm.mlir.addressof @fmt_scalar_count : !llvm.ptr
    %fmt_simd_count_ptr = llvm.mlir.addressof @fmt_simd_count : !llvm.ptr
    %print_scalar_count = llvm.call @printf(%fmt_scalar_count_ptr, %scalar_final_count) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
    %print_simd_count = llvm.call @printf(%fmt_simd_count_ptr, %simd_final_count) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
    return %17 : tensor<?x?xf64>
  }
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c262144_i64 = arith.constant 262144 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %0 = llvm.mlir.addressof @fmt_time : !llvm.ptr
    %1 = llvm.mlir.addressof @mode_w : !llvm.ptr
    %2 = llvm.mlir.addressof @benchmarkFile : !llvm.ptr
    %3 = llvm.mlir.addressof @fmt_exec_time : !llvm.ptr
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %4 = llvm.mlir.addressof @fileB : !llvm.ptr
    %5 = llvm.mlir.addressof @fileC : !llvm.ptr
    %6 = llvm.mlir.addressof @fileD : !llvm.ptr
    %c3 = arith.constant 3 : index
    %alloca = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca[%c0] : memref<?xindex>
    memref.store %c0, %alloca[%c1] : memref<?xindex>
    memref.store %c0, %alloca[%c2] : memref<?xindex>
    %7 = call @createCheckedSparseTensorReader(%4, %alloca, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %8 = call @getSparseTensorReaderDimSizes(%7) : (!llvm.ptr) -> memref<?xindex>
    %alloca_0 = memref.alloca(%c3) : memref<?xi64>
    memref.store %c65536_i64, %alloca_0[%c0] : memref<?xi64>
    memref.store %c262144_i64, %alloca_0[%c1] : memref<?xi64>
    memref.store %c262144_i64, %alloca_0[%c2] : memref<?xi64>
    %alloca_1 = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca_1[%c0] : memref<?xindex>
    memref.store %c1, %alloca_1[%c1] : memref<?xindex>
    memref.store %c2, %alloca_1[%c2] : memref<?xindex>
    %9 = call @newSparseTensor(%8, %8, %alloca_0, %alloca_1, %alloca_1, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %7) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%7) : (!llvm.ptr) -> ()
    %alloca_2 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_2[%c0] : memref<?xindex>
    memref.store %c0, %alloca_2[%c1] : memref<?xindex>
    %10 = call @createCheckedSparseTensorReader(%5, %alloca_2, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %11 = call @getSparseTensorReaderDimSizes(%10) : (!llvm.ptr) -> memref<?xindex>
    %alloca_3 = memref.alloca(%c2) : memref<?xi64>
    memref.store %c65536_i64, %alloca_3[%c0] : memref<?xi64>
    memref.store %c262144_i64, %alloca_3[%c1] : memref<?xi64>
    %alloca_4 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_4[%c0] : memref<?xindex>
    memref.store %c1, %alloca_4[%c1] : memref<?xindex>
    %12 = call @newSparseTensor(%11, %11, %alloca_3, %alloca_4, %alloca_4, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %10) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%10) : (!llvm.ptr) -> ()
    %alloca_5 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_5[%c0] : memref<?xindex>
    memref.store %c0, %alloca_5[%c1] : memref<?xindex>
    %13 = call @createCheckedSparseTensorReader(%6, %alloca_5, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %14 = call @getSparseTensorReaderDimSizes(%13) : (!llvm.ptr) -> memref<?xindex>
    %alloca_6 = memref.alloca(%c2) : memref<?xi64>
    memref.store %c65536_i64, %alloca_6[%c0] : memref<?xi64>
    memref.store %c262144_i64, %alloca_6[%c1] : memref<?xi64>
    %alloca_7 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_7[%c0] : memref<?xindex>
    memref.store %c1, %alloca_7[%c1] : memref<?xindex>
    %15 = call @newSparseTensor(%14, %14, %alloca_6, %alloca_7, %alloca_7, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %13) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%13) : (!llvm.ptr) -> ()
    %16 = call @rtclock() : () -> f64
    %17 = call @spmttkrp(%9, %12, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> tensor<?x?xf64>
    %18 = call @rtclock() : () -> f64
    %19 = arith.subf %18, %16 : f64
    bufferization.dealloc_tensor %17 : tensor<?x?xf64>
    %20 = llvm.call @printf(%3, %19) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    %21 = llvm.call @fopen(%2, %1) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %22 = llvm.call @fprintf(%21, %0, %19) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
    %23 = llvm.call @fclose(%21) : (!llvm.ptr) -> i32
    call @delSparseTensor(%9) : (!llvm.ptr) -> ()
    call @delSparseTensor(%12) : (!llvm.ptr) -> ()
    call @delSparseTensor(%15) : (!llvm.ptr) -> ()
    return
  }
}
