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
  func.func @spmmh(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> tensor<?x?xf64> {
    %cst = arith.constant dense<0.000000e+00> : vector<4xf64>
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // Scalar-epilogue and SIMD-fastlane element counters: both reset at
    // the start of every call to @spmmh, incremented once per coiteration
    // step in their respective "do" regions below (the scalar_fallback
    // one, and the vectorized one right above it), and printed separately
    // just before returning.
    %scalar_counter_ref = memref.get_global @scalar_epilogue_count : memref<i64>
    %simd_counter_ref = memref.get_global @simd_fastlane_count : memref<i64>
    %scalar_zero_i64 = arith.constant 0 : i64
    %scalar_one_i64 = arith.constant 1 : i64
    memref.store %scalar_zero_i64, %scalar_counter_ref[] : memref<i64>
    memref.store %scalar_zero_i64, %simd_counter_ref[] : memref<i64>
    %0 = call @sparseLvlSize(%arg0, %c1) : (!llvm.ptr, index) -> index
    %1 = call @sparseLvlSize(%arg2, %c0) : (!llvm.ptr, index) -> index
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
    %11 = call @sparseLvlSize(%arg1, %c1) : (!llvm.ptr, index) -> index
    %12 = call @sparsePositions0(%arg2, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %13 = call @sparseCoordinates0(%arg2, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    scf.for %arg3 = %c0 to %8 step %c1 {
      %15 = arith.muli %arg3, %11 : index
      %16 = arith.addi %arg3, %c1 : index
      scf.for %arg4 = %c0 to %11 step %c1 {
        %17 = arith.addi %arg4, %15 : index
        %18 = memref.load %5[%17] : memref<?xf64>
        %19 = memref.load %9[%arg3] : memref<?xindex>
        %20 = memref.load %9[%16] : memref<?xindex>
        %21 = memref.load %12[%arg4] : memref<?xindex>
        %22 = arith.addi %arg4, %c1 : index
        %23 = memref.load %12[%22] : memref<?xindex>
        %24:2 = scf.while (%arg5 = %19, %arg6 = %21) : (index, index) -> (index, index) {
          %26 = arith.addi %arg5, %c4 : index
          %27 = arith.cmpi ule, %26, %20 : index
          %28 = arith.addi %arg6, %c4 : index
          %29 = arith.cmpi ule, %28, %23 : index
          %30 = arith.andi %27, %29 : i1
          scf.condition(%30) %arg5, %arg6 : index, index
        } do {
        ^bb0(%arg5: index, %arg6: index):
          %26 = vector.load %10[%arg5] : memref<?xindex>, vector<4xindex>
          %27 = vector.load %4[%arg5] : memref<?xf64>, vector<4xf64>
          %28 = vector.load %13[%arg6] : memref<?xindex>, vector<4xindex>
          %29 = vector.load %6[%arg6] : memref<?xf64>, vector<4xf64>
          %30 = vector.extract %26[0] : index from vector<4xindex>
          %31 = vector.extract %26[1] : index from vector<4xindex>
          %32 = vector.extract %26[2] : index from vector<4xindex>
          %33 = vector.extract %26[3] : index from vector<4xindex>
          %34 = vector.extract %28[0] : index from vector<4xindex>
          %35 = vector.extract %28[1] : index from vector<4xindex>
          %36 = vector.extract %28[2] : index from vector<4xindex>
          %37 = vector.extract %28[3] : index from vector<4xindex>
          %38 = vector.broadcast %30 : index to vector<4xindex>
          %39 = arith.cmpi eq, %38, %28 : vector<4xindex>
          %40 = vector.broadcast %31 : index to vector<4xindex>
          %41 = arith.cmpi eq, %40, %28 : vector<4xindex>
          %42 = vector.broadcast %32 : index to vector<4xindex>
          %43 = arith.cmpi eq, %42, %28 : vector<4xindex>
          %44 = vector.broadcast %33 : index to vector<4xindex>
          %45 = arith.cmpi eq, %44, %28 : vector<4xindex>
          %46 = arith.select %39, %29, %cst : vector<4xi1>, vector<4xf64>
          %47 = vector.reduction <add>, %46 : vector<4xf64> into f64
          %48 = arith.select %41, %29, %cst : vector<4xi1>, vector<4xf64>
          %49 = vector.reduction <add>, %48 : vector<4xf64> into f64
          %50 = arith.select %43, %29, %cst : vector<4xi1>, vector<4xf64>
          %51 = vector.reduction <add>, %50 : vector<4xf64> into f64
          %52 = arith.select %45, %29, %cst : vector<4xi1>, vector<4xf64>
          %53 = vector.reduction <add>, %52 : vector<4xf64> into f64
          %54 = vector.extract %27[0] : f64 from vector<4xf64>
          %55 = vector.extract %27[1] : f64 from vector<4xf64>
          %56 = vector.extract %27[2] : f64 from vector<4xf64>
          %57 = vector.extract %27[3] : f64 from vector<4xf64>
          %58 = arith.mulf %54, %47 : f64
          %59 = arith.mulf %58, %18 : f64
          %60 = arith.mulf %55, %49 : f64
          %61 = arith.mulf %60, %18 : f64
          %62 = arith.mulf %56, %51 : f64
          %63 = arith.mulf %62, %18 : f64
          %64 = arith.mulf %57, %53 : f64
          %65 = arith.mulf %64, %18 : f64
          %66 = memref.load %7[%30, %arg4] : memref<?x?xf64>
          %67 = arith.addf %66, %59 : f64
          memref.store %67, %7[%30, %arg4] : memref<?x?xf64>
          %68 = memref.load %7[%31, %arg4] : memref<?x?xf64>
          %69 = arith.addf %68, %61 : f64
          memref.store %69, %7[%31, %arg4] : memref<?x?xf64>
          %70 = memref.load %7[%32, %arg4] : memref<?x?xf64>
          %71 = arith.addf %70, %63 : f64
          memref.store %71, %7[%32, %arg4] : memref<?x?xf64>
          %72 = memref.load %7[%33, %arg4] : memref<?x?xf64>
          %73 = arith.addf %72, %65 : f64
          memref.store %73, %7[%33, %arg4] : memref<?x?xf64>
          %74 = arith.minui %33, %37 : index
          %75 = arith.cmpi ule, %30, %74 : index
          %76 = arith.cmpi ule, %31, %74 : index
          %77 = arith.cmpi ule, %32, %74 : index
          %78 = arith.cmpi ule, %33, %74 : index
          %79 = arith.select %75, %c1, %c0 : index
          %80 = arith.select %76, %c1, %c0 : index
          %81 = arith.select %77, %c1, %c0 : index
          %82 = arith.select %78, %c1, %c0 : index
          %83 = arith.addi %79, %80 : index
          %84 = arith.addi %81, %82 : index
          %85 = arith.addi %83, %84 : index
          %86 = arith.addi %arg5, %85 : index
          %87 = arith.cmpi ule, %34, %74 : index
          %88 = arith.cmpi ule, %35, %74 : index
          %89 = arith.cmpi ule, %36, %74 : index
          %90 = arith.cmpi ule, %37, %74 : index
          %91 = arith.select %87, %c1, %c0 : index
          %92 = arith.select %88, %c1, %c0 : index
          %93 = arith.select %89, %c1, %c0 : index
          %94 = arith.select %90, %c1, %c0 : index
          %95 = arith.addi %91, %92 : index
          %96 = arith.addi %93, %94 : index
          %97 = arith.addi %95, %96 : index
          %98 = arith.addi %arg6, %97 : index
          // %85 and %97 are this iteration's arg5-side/arg6-side (B-side/
          // D-side) advance counts (0-4 each, from the popcount-style
          // select+add above) — their sum is exactly how many
          // index-stream elements this SIMD iteration consumed, the same
          // unit the scalar counter uses.
          %simd_elems_idx = arith.addi %85, %97 : index
          %simd_elems_i64 = arith.index_cast %simd_elems_idx : index to i64
          %simd_count_val = memref.load %simd_counter_ref[] : memref<i64>
          %simd_count_next = arith.addi %simd_count_val, %simd_elems_i64 : i64
          memref.store %simd_count_next, %simd_counter_ref[] : memref<i64>
          scf.yield %86, %98 : index, index
        }
        %25:2 = scf.while (%arg5 = %24#0, %arg6 = %24#1) : (index, index) -> (index, index) {
          %26 = arith.cmpi ult, %arg5, %20 : index
          %27 = arith.cmpi ult, %arg6, %23 : index
          %28 = arith.andi %26, %27 : i1
          scf.condition(%28) %arg5, %arg6 : index, index
        } do {
        ^bb0(%arg5: index, %arg6: index):
          %26 = memref.load %10[%arg5] : memref<?xindex>
          %27 = memref.load %13[%arg6] : memref<?xindex>
          %28 = arith.cmpi ult, %27, %26 : index
          %29 = arith.select %28, %27, %26 : index
          %30 = arith.cmpi eq, %26, %29 : index
          %31 = arith.cmpi eq, %27, %29 : index
          // This iteration consumes 1 element from B (%10/%4, arg5 side)
          // if %30, and 1 from D (%13/%6, arg6 side) if %31 — never zero
          // (at least one side always equals the min), and 2 on an actual
          // index match (both true), same asymmetric-consumption logic
          // the SIMD side's %85/%97 already track — so the two counters
          // stay apples-to-apples instead of scalar just counting loop
          // steps.
          %scalar_b_adv = arith.select %30, %scalar_one_i64, %scalar_zero_i64 : i64
          %scalar_d_adv = arith.select %31, %scalar_one_i64, %scalar_zero_i64 : i64
          %scalar_elems_this_iter = arith.addi %scalar_b_adv, %scalar_d_adv : i64
          %scalar_count_val = memref.load %scalar_counter_ref[] : memref<i64>
          %scalar_count_next = arith.addi %scalar_count_val, %scalar_elems_this_iter : i64
          memref.store %scalar_count_next, %scalar_counter_ref[] : memref<i64>
          %32 = arith.andi %30, %31 : i1
          scf.if %32 {
            %39 = memref.load %7[%29, %arg4] : memref<?x?xf64>
            %40 = memref.load %4[%arg5] : memref<?xf64>
            %41 = arith.mulf %40, %18 : f64
            %42 = memref.load %6[%arg6] : memref<?xf64>
            %43 = arith.mulf %41, %42 : f64
            %44 = arith.addf %39, %43 : f64
            memref.store %44, %7[%29, %arg4] : memref<?x?xf64>
          } else {
          }
          %33 = arith.cmpi eq, %26, %29 : index
          %34 = arith.addi %arg5, %c1 : index
          %35 = arith.select %33, %34, %arg5 : index
          %36 = arith.cmpi eq, %27, %29 : index
          %37 = arith.addi %arg6, %c1 : index
          %38 = arith.select %36, %37, %arg6 : index
          scf.yield %35, %38 : index, index
        } attributes {"Emitted from" = "linalg.generic", splyce.scalar_fallback}
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %14 = bufferization.to_tensor %7 restrict : memref<?x?xf64> to tensor<?x?xf64>
    %scalar_final_count = memref.load %scalar_counter_ref[] : memref<i64>
    %simd_final_count = memref.load %simd_counter_ref[] : memref<i64>
    %fmt_scalar_count_ptr = llvm.mlir.addressof @fmt_scalar_count : !llvm.ptr
    %fmt_simd_count_ptr = llvm.mlir.addressof @fmt_simd_count : !llvm.ptr
    %print_scalar_count = llvm.call @printf(%fmt_scalar_count_ptr, %scalar_final_count) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
    %print_simd_count = llvm.call @printf(%fmt_simd_count_ptr, %simd_final_count) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
    return %14 : tensor<?x?xf64>
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
    %5 = llvm.mlir.addressof @fileC : !llvm.ptr
    %6 = llvm.mlir.addressof @fileD : !llvm.ptr
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %alloca = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca[%c0] : memref<?xindex>
    memref.store %c0, %alloca[%c1] : memref<?xindex>
    %7 = call @createCheckedSparseTensorReader(%4, %alloca, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %8 = call @getSparseTensorReaderDimSizes(%7) : (!llvm.ptr) -> memref<?xindex>
    %9 = memref.load %8[%c0] : memref<?xindex>
    %10 = memref.load %8[%c1] : memref<?xindex>
    %alloca_0 = memref.alloca(%c2) : memref<?xi64>
    memref.store %c65536_i64, %alloca_0[%c0] : memref<?xi64>
    memref.store %c262144_i64, %alloca_0[%c1] : memref<?xi64>
    %alloca_1 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c1, %alloca_1[%c0] : memref<?xindex>
    memref.store %c0, %alloca_1[%c1] : memref<?xindex>
    %alloca_2 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c1, %alloca_2[%c0] : memref<?xindex>
    memref.store %c0, %alloca_2[%c1] : memref<?xindex>
    %alloca_3 = memref.alloca(%c2) : memref<?xindex>
    memref.store %10, %alloca_3[%c0] : memref<?xindex>
    memref.store %9, %alloca_3[%c1] : memref<?xindex>
    %11 = call @newSparseTensor(%8, %alloca_3, %alloca_0, %alloca_1, %alloca_2, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %7) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%7) : (!llvm.ptr) -> ()
    %alloca_4 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_4[%c0] : memref<?xindex>
    memref.store %c0, %alloca_4[%c1] : memref<?xindex>
    %12 = call @createCheckedSparseTensorReader(%5, %alloca_4, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %13 = call @getSparseTensorReaderDimSizes(%12) : (!llvm.ptr) -> memref<?xindex>
    %alloca_5 = memref.alloca(%c2) : memref<?xi64>
    memref.store %c65536_i64, %alloca_5[%c0] : memref<?xi64>
    memref.store %c65536_i64, %alloca_5[%c1] : memref<?xi64>
    %alloca_6 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_6[%c0] : memref<?xindex>
    memref.store %c1, %alloca_6[%c1] : memref<?xindex>
    %14 = call @newSparseTensor(%13, %13, %alloca_5, %alloca_6, %alloca_6, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %12) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%12) : (!llvm.ptr) -> ()
    %alloca_7 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_7[%c0] : memref<?xindex>
    memref.store %c0, %alloca_7[%c1] : memref<?xindex>
    %15 = call @createCheckedSparseTensorReader(%6, %alloca_7, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %16 = call @getSparseTensorReaderDimSizes(%15) : (!llvm.ptr) -> memref<?xindex>
    %17 = memref.load %16[%c0] : memref<?xindex>
    %18 = memref.load %16[%c1] : memref<?xindex>
    %alloca_8 = memref.alloca(%c2) : memref<?xi64>
    memref.store %c65536_i64, %alloca_8[%c0] : memref<?xi64>
    memref.store %c262144_i64, %alloca_8[%c1] : memref<?xi64>
    %alloca_9 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c1, %alloca_9[%c0] : memref<?xindex>
    memref.store %c0, %alloca_9[%c1] : memref<?xindex>
    %alloca_10 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c1, %alloca_10[%c0] : memref<?xindex>
    memref.store %c0, %alloca_10[%c1] : memref<?xindex>
    %alloca_11 = memref.alloca(%c2) : memref<?xindex>
    memref.store %18, %alloca_11[%c0] : memref<?xindex>
    memref.store %17, %alloca_11[%c1] : memref<?xindex>
    %19 = call @newSparseTensor(%16, %alloca_11, %alloca_8, %alloca_9, %alloca_10, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %15) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%15) : (!llvm.ptr) -> ()
    %20 = call @rtclock() : () -> f64
    %21 = call @spmmh(%11, %14, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> tensor<?x?xf64>
    %22 = call @rtclock() : () -> f64
    %23 = arith.subf %22, %20 : f64
    bufferization.dealloc_tensor %21 : tensor<?x?xf64>
    %24 = llvm.call @printf(%3, %23) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    %25 = llvm.call @fopen(%2, %1) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %26 = llvm.call @fprintf(%25, %0, %23) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
    %27 = llvm.call @fclose(%25) : (!llvm.ptr) -> i32
    call @delSparseTensor(%11) : (!llvm.ptr) -> ()
    call @delSparseTensor(%14) : (!llvm.ptr) -> ()
    call @delSparseTensor(%19) : (!llvm.ptr) -> ()
    return
  }
}

