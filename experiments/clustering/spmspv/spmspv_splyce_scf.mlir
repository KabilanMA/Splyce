#map = affine_map<(d0) -> (d0)>
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
  llvm.mlir.global internal constant @fileX("tensor_x.tns\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @benchmarkFile("benchmark\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @mode_w("w\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_time("%f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_exec_time("Execution time: %f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_scalar_count("Scalar elements processed: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_simd_count("SIMD elements processed: %ld\0A\00") {addr_space = 0 : i32}
  memref.global "private" @scalar_epilogue_count : memref<i64> = dense<0>
  memref.global "private" @simd_fastlane_count : memref<i64> = dense<0>
  func.func @spmspv(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> tensor<?xf64> {
    %cst = arith.constant dense<0.000000e+00> : vector<4xf64>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    // Scalar-epilogue and SIMD-fastlane element counters: both reset at
    // the start of every call to @spmspv, incremented once per coiteration
    // step in their respective "do" regions below (the scalar_fallback
    // one, and the vectorized one right above it), and printed separately
    // just before returning.
    %scalar_counter_ref = memref.get_global @scalar_epilogue_count : memref<i64>
    %simd_counter_ref = memref.get_global @simd_fastlane_count : memref<i64>
    %scalar_zero_i64 = arith.constant 0 : i64
    %scalar_one_i64 = arith.constant 1 : i64
    memref.store %scalar_zero_i64, %scalar_counter_ref[] : memref<i64>
    memref.store %scalar_zero_i64, %simd_counter_ref[] : memref<i64>
    %0 = call @sparseLvlSize(%arg0, %c0) : (!llvm.ptr, index) -> index
    %1 = bufferization.alloc_tensor(%0) : tensor<?xf64>
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?xf64>
    %3 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %4 = call @sparseValuesF64(%arg1) : (!llvm.ptr) -> memref<?xf64>
    %5 = bufferization.to_buffer %2 : tensor<?xf64> to memref<?xf64>
    %6 = call @sparseLvlSize(%arg0, %c0) : (!llvm.ptr, index) -> index
    %7 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %8 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %9 = call @sparsePositions0(%arg1, %c0) : (!llvm.ptr, index) -> memref<?xindex>
    %10 = call @sparseCoordinates0(%arg1, %c0) : (!llvm.ptr, index) -> memref<?xindex>
    scf.for %arg2 = %c0 to %6 step %c1 {
      %12 = memref.load %5[%arg2] : memref<?xf64>
      %13 = memref.load %7[%arg2] : memref<?xindex>
      %14 = arith.addi %arg2, %c1 : index
      %15 = memref.load %7[%14] : memref<?xindex>
      %16 = memref.load %9[%c0] : memref<?xindex>
      %17 = memref.load %9[%c1] : memref<?xindex>
      %18:3 = scf.while (%arg3 = %13, %arg4 = %16, %arg5 = %12) : (index, index, f64) -> (index, index, f64) {
        %20 = arith.addi %arg3, %c4 : index
        %21 = arith.cmpi ule, %20, %15 : index
        %22 = arith.addi %arg4, %c4 : index
        %23 = arith.cmpi ule, %22, %17 : index
        %24 = arith.andi %21, %23 : i1
        scf.condition(%24) %arg3, %arg4, %arg5 : index, index, f64
      } do {
      ^bb0(%arg3: index, %arg4: index, %arg5: f64):
        %20 = vector.load %8[%arg3] : memref<?xindex>, vector<4xindex>
        %21 = vector.load %3[%arg3] : memref<?xf64>, vector<4xf64>
        %22 = vector.load %10[%arg4] : memref<?xindex>, vector<4xindex>
        %23 = vector.load %4[%arg4] : memref<?xf64>, vector<4xf64>
        %24 = vector.extract %20[0] : index from vector<4xindex>
        %25 = vector.extract %20[1] : index from vector<4xindex>
        %26 = vector.extract %20[2] : index from vector<4xindex>
        %27 = vector.extract %20[3] : index from vector<4xindex>
        %28 = vector.extract %22[0] : index from vector<4xindex>
        %29 = vector.extract %22[1] : index from vector<4xindex>
        %30 = vector.extract %22[2] : index from vector<4xindex>
        %31 = vector.extract %22[3] : index from vector<4xindex>
        %32 = vector.broadcast %24 : index to vector<4xindex>
        %33 = arith.cmpi eq, %32, %22 : vector<4xindex>
        %34 = vector.broadcast %25 : index to vector<4xindex>
        %35 = arith.cmpi eq, %34, %22 : vector<4xindex>
        %36 = vector.broadcast %26 : index to vector<4xindex>
        %37 = arith.cmpi eq, %36, %22 : vector<4xindex>
        %38 = vector.broadcast %27 : index to vector<4xindex>
        %39 = arith.cmpi eq, %38, %22 : vector<4xindex>
        %40 = arith.select %33, %23, %cst : vector<4xi1>, vector<4xf64>
        %41 = vector.reduction <add>, %40 : vector<4xf64> into f64
        %42 = arith.select %35, %23, %cst : vector<4xi1>, vector<4xf64>
        %43 = vector.reduction <add>, %42 : vector<4xf64> into f64
        %44 = arith.select %37, %23, %cst : vector<4xi1>, vector<4xf64>
        %45 = vector.reduction <add>, %44 : vector<4xf64> into f64
        %46 = arith.select %39, %23, %cst : vector<4xi1>, vector<4xf64>
        %47 = vector.reduction <add>, %46 : vector<4xf64> into f64
        %48 = vector.extract %21[0] : f64 from vector<4xf64>
        %49 = vector.extract %21[1] : f64 from vector<4xf64>
        %50 = vector.extract %21[2] : f64 from vector<4xf64>
        %51 = vector.extract %21[3] : f64 from vector<4xf64>
        %52 = arith.mulf %48, %41 : f64
        %53 = arith.mulf %49, %43 : f64
        %54 = arith.mulf %50, %45 : f64
        %55 = arith.mulf %51, %47 : f64
        %56 = arith.addf %52, %53 : f64
        %57 = arith.addf %54, %55 : f64
        %58 = arith.addf %56, %57 : f64
        %59 = arith.addf %arg5, %58 : f64
        %60 = arith.minui %27, %31 : index
        %61 = arith.cmpi ule, %24, %60 : index
        %62 = arith.cmpi ule, %25, %60 : index
        %63 = arith.cmpi ule, %26, %60 : index
        %64 = arith.cmpi ule, %27, %60 : index
        %65 = arith.select %61, %c1, %c0 : index
        %66 = arith.select %62, %c1, %c0 : index
        %67 = arith.select %63, %c1, %c0 : index
        %68 = arith.select %64, %c1, %c0 : index
        %69 = arith.addi %65, %66 : index
        %70 = arith.addi %67, %68 : index
        %71 = arith.addi %69, %70 : index
        %72 = arith.addi %arg3, %71 : index
        %73 = arith.cmpi ule, %28, %60 : index
        %74 = arith.cmpi ule, %29, %60 : index
        %75 = arith.cmpi ule, %30, %60 : index
        %76 = arith.cmpi ule, %31, %60 : index
        %77 = arith.select %73, %c1, %c0 : index
        %78 = arith.select %74, %c1, %c0 : index
        %79 = arith.select %75, %c1, %c0 : index
        %80 = arith.select %76, %c1, %c0 : index
        %81 = arith.addi %77, %78 : index
        %82 = arith.addi %79, %80 : index
        %83 = arith.addi %81, %82 : index
        %84 = arith.addi %arg4, %83 : index
        // %71 and %83 are this iteration's arg3-side/arg4-side advance
        // counts (0-4 each, from the popcount-style select+add above) —
        // their sum is exactly how many index-stream elements this SIMD
        // iteration consumed, the same unit the scalar counter uses.
        %simd_elems_idx = arith.addi %71, %83 : index
        %simd_elems_i64 = arith.index_cast %simd_elems_idx : index to i64
        %simd_count_val = memref.load %simd_counter_ref[] : memref<i64>
        %simd_count_next = arith.addi %simd_count_val, %simd_elems_i64 : i64
        memref.store %simd_count_next, %simd_counter_ref[] : memref<i64>
        scf.yield %72, %84, %59 : index, index, f64
      }
      %19:3 = scf.while (%arg3 = %18#0, %arg4 = %18#1, %arg5 = %18#2) : (index, index, f64) -> (index, index, f64) {
        %20 = arith.cmpi ult, %arg3, %15 : index
        %21 = arith.cmpi ult, %arg4, %17 : index
        %22 = arith.andi %20, %21 : i1
        scf.condition(%22) %arg3, %arg4, %arg5 : index, index, f64
      } do {
      ^bb0(%arg3: index, %arg4: index, %arg5: f64):
        %20 = memref.load %8[%arg3] : memref<?xindex>
        %21 = memref.load %10[%arg4] : memref<?xindex>
        %22 = arith.cmpi ult, %21, %20 : index
        %23 = arith.select %22, %21, %20 : index
        %24 = arith.cmpi eq, %20, %23 : index
        %25 = arith.cmpi eq, %21, %23 : index
        // This iteration consumes 1 element from B (%8/%3, arg3 side) if
        // %24, and 1 from x (%10/%4, arg4 side) if %25 — never zero (at
        // least one side always equals the min), and 2 on an actual index
        // match (both true), same asymmetric-consumption logic the SIMD
        // side's %71/%83 already track — so the two counters stay
        // apples-to-apples instead of scalar just counting loop steps.
        %scalar_b_adv = arith.select %24, %scalar_one_i64, %scalar_zero_i64 : i64
        %scalar_x_adv = arith.select %25, %scalar_one_i64, %scalar_zero_i64 : i64
        %scalar_elems_this_iter = arith.addi %scalar_b_adv, %scalar_x_adv : i64
        %scalar_count_val = memref.load %scalar_counter_ref[] : memref<i64>
        %scalar_count_next = arith.addi %scalar_count_val, %scalar_elems_this_iter : i64
        memref.store %scalar_count_next, %scalar_counter_ref[] : memref<i64>
        %26 = arith.andi %24, %25 : i1
        %27 = scf.if %26 -> (f64) {
          %34 = memref.load %3[%arg3] : memref<?xf64>
          %35 = memref.load %4[%arg4] : memref<?xf64>
          %36 = arith.mulf %34, %35 : f64
          %37 = arith.addf %arg5, %36 : f64
          scf.yield %37 : f64
        } else {
          scf.yield %arg5 : f64
        }
        %28 = arith.cmpi eq, %20, %23 : index
        %29 = arith.addi %arg3, %c1 : index
        %30 = arith.select %28, %29, %arg3 : index
        %31 = arith.cmpi eq, %21, %23 : index
        %32 = arith.addi %arg4, %c1 : index
        %33 = arith.select %31, %32, %arg4 : index
        scf.yield %30, %33, %27 : index, index, f64
      } attributes {"Emitted from" = "linalg.generic", splyce.scalar_fallback}
      memref.store %19#2, %5[%arg2] : memref<?xf64>
    } {"Emitted from" = "linalg.generic"}
    %11 = bufferization.to_tensor %5 restrict : memref<?xf64> to tensor<?xf64>
    %scalar_final_count = memref.load %scalar_counter_ref[] : memref<i64>
    %simd_final_count = memref.load %simd_counter_ref[] : memref<i64>
    %fmt_scalar_count_ptr = llvm.mlir.addressof @fmt_scalar_count : !llvm.ptr
    %fmt_simd_count_ptr = llvm.mlir.addressof @fmt_simd_count : !llvm.ptr
    %print_scalar_count = llvm.call @printf(%fmt_scalar_count_ptr, %scalar_final_count) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
    %print_simd_count = llvm.call @printf(%fmt_simd_count_ptr, %simd_final_count) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
    return %11 : tensor<?xf64>
  }
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c262144_i64 = arith.constant 262144 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %0 = llvm.mlir.addressof @fmt_time : !llvm.ptr
    %1 = llvm.mlir.addressof @mode_w : !llvm.ptr
    %2 = llvm.mlir.addressof @benchmarkFile : !llvm.ptr
    %3 = llvm.mlir.addressof @fmt_exec_time : !llvm.ptr
    %4 = llvm.mlir.addressof @fileB : !llvm.ptr
    %5 = llvm.mlir.addressof @fileX : !llvm.ptr
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %alloca = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca[%c0] : memref<?xindex>
    memref.store %c0, %alloca[%c1] : memref<?xindex>
    %6 = call @createCheckedSparseTensorReader(%4, %alloca, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %7 = call @getSparseTensorReaderDimSizes(%6) : (!llvm.ptr) -> memref<?xindex>
    %alloca_0 = memref.alloca(%c2) : memref<?xi64>
    memref.store %c65536_i64, %alloca_0[%c0] : memref<?xi64>
    memref.store %c262144_i64, %alloca_0[%c1] : memref<?xi64>
    %alloca_1 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_1[%c0] : memref<?xindex>
    memref.store %c1, %alloca_1[%c1] : memref<?xindex>
    %8 = call @newSparseTensor(%7, %7, %alloca_0, %alloca_1, %alloca_1, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %6) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%6) : (!llvm.ptr) -> ()
    %alloca_2 = memref.alloca(%c1) : memref<?xindex>
    memref.store %c0, %alloca_2[%c0] : memref<?xindex>
    %9 = call @createCheckedSparseTensorReader(%5, %alloca_2, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %10 = call @getSparseTensorReaderDimSizes(%9) : (!llvm.ptr) -> memref<?xindex>
    %alloca_3 = memref.alloca(%c1) : memref<?xi64>
    memref.store %c262144_i64, %alloca_3[%c0] : memref<?xi64>
    %alloca_4 = memref.alloca(%c1) : memref<?xindex>
    memref.store %c0, %alloca_4[%c0] : memref<?xindex>
    %11 = call @newSparseTensor(%10, %10, %alloca_3, %alloca_4, %alloca_4, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %9) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%9) : (!llvm.ptr) -> ()
    %12 = call @rtclock() : () -> f64
    %13 = call @spmspv(%8, %11) : (!llvm.ptr, !llvm.ptr) -> tensor<?xf64>
    %14 = call @rtclock() : () -> f64
    %15 = arith.subf %14, %12 : f64
    bufferization.dealloc_tensor %13 : tensor<?xf64>
    %16 = llvm.call @printf(%3, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    %17 = llvm.call @fopen(%2, %1) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %18 = llvm.call @fprintf(%17, %0, %15) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
    %19 = llvm.call @fclose(%17) : (!llvm.ptr) -> i32
    call @delSparseTensor(%8) : (!llvm.ptr) -> ()
    call @delSparseTensor(%11) : (!llvm.ptr) -> ()
    return
  }
}
