#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
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
  llvm.mlir.global internal constant @benchmarkFile("benchmark\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @mode_w("w\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_time("%f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_exec_time("Execution time: %f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_scalar_count("Scalar elements processed: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fmt_simd_count("SIMD elements processed: %ld\0A\00") {addr_space = 0 : i32}
  memref.global "private" @scalar_epilogue_count : memref<i64> = dense<0>
  memref.global "private" @simd_fastlane_count : memref<i64> = dense<0>
  func.func @spttspm(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> tensor<?x?x?xf64> {
    %cst = arith.constant dense<0.000000e+00> : vector<4xf64>
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // Scalar-epilogue and SIMD-fastlane element counters: both reset at
    // the start of every call to @spttspm, incremented once per
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
    %2 = call @sparseLvlSize(%arg1, %c0) : (!llvm.ptr, index) -> index
    %3 = bufferization.alloc_tensor(%0, %1, %2) : tensor<?x?x?xf64>
    %4 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%3 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst_0 : f64
    } -> tensor<?x?x?xf64>
    %5 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %6 = call @sparseValuesF64(%arg1) : (!llvm.ptr) -> memref<?xf64>
    %7 = bufferization.to_buffer %4 : tensor<?x?x?xf64> to memref<?x?x?xf64>
    %8 = call @sparseLvlSize(%arg0, %c0) : (!llvm.ptr, index) -> index
    %9 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %10 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %11 = call @sparsePositions0(%arg0, %c2) : (!llvm.ptr, index) -> memref<?xindex>
    %12 = call @sparseCoordinates0(%arg0, %c2) : (!llvm.ptr, index) -> memref<?xindex>
    %13 = call @sparseLvlSize(%arg1, %c0) : (!llvm.ptr, index) -> index
    %14 = call @sparsePositions0(%arg1, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %15 = call @sparseCoordinates0(%arg1, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    scf.for %arg2 = %c0 to %8 step %c1 {
      %17 = memref.load %9[%arg2] : memref<?xindex>
      %18 = arith.addi %arg2, %c1 : index
      %19 = memref.load %9[%18] : memref<?xindex>
      scf.for %arg3 = %17 to %19 step %c1 {
        %20 = memref.load %10[%arg3] : memref<?xindex>
        %21 = arith.addi %arg3, %c1 : index
        scf.for %arg4 = %c0 to %13 step %c1 {
          %22 = memref.load %7[%arg2, %20, %arg4] : memref<?x?x?xf64>
          %23 = memref.load %11[%arg3] : memref<?xindex>
          %24 = memref.load %11[%21] : memref<?xindex>
          %25 = memref.load %14[%arg4] : memref<?xindex>
          %26 = arith.addi %arg4, %c1 : index
          %27 = memref.load %14[%26] : memref<?xindex>
          %28:3 = scf.while (%arg5 = %23, %arg6 = %25, %arg7 = %22) : (index, index, f64) -> (index, index, f64) {
            %30 = arith.addi %arg5, %c4 : index
            %31 = arith.cmpi ule, %30, %24 : index
            %32 = arith.addi %arg6, %c4 : index
            %33 = arith.cmpi ule, %32, %27 : index
            %34 = arith.andi %31, %33 : i1
            scf.condition(%34) %arg5, %arg6, %arg7 : index, index, f64
          } do {
          ^bb0(%arg5: index, %arg6: index, %arg7: f64):
            %30 = vector.load %12[%arg5] : memref<?xindex>, vector<4xindex>
            %31 = vector.load %5[%arg5] : memref<?xf64>, vector<4xf64>
            %32 = vector.load %15[%arg6] : memref<?xindex>, vector<4xindex>
            %33 = vector.load %6[%arg6] : memref<?xf64>, vector<4xf64>
            %34 = vector.extract %30[0] : index from vector<4xindex>
            %35 = vector.extract %30[1] : index from vector<4xindex>
            %36 = vector.extract %30[2] : index from vector<4xindex>
            %37 = vector.extract %30[3] : index from vector<4xindex>
            %38 = vector.extract %32[0] : index from vector<4xindex>
            %39 = vector.extract %32[1] : index from vector<4xindex>
            %40 = vector.extract %32[2] : index from vector<4xindex>
            %41 = vector.extract %32[3] : index from vector<4xindex>
            %42 = vector.broadcast %34 : index to vector<4xindex>
            %43 = arith.cmpi eq, %42, %32 : vector<4xindex>
            %44 = vector.broadcast %35 : index to vector<4xindex>
            %45 = arith.cmpi eq, %44, %32 : vector<4xindex>
            %46 = vector.broadcast %36 : index to vector<4xindex>
            %47 = arith.cmpi eq, %46, %32 : vector<4xindex>
            %48 = vector.broadcast %37 : index to vector<4xindex>
            %49 = arith.cmpi eq, %48, %32 : vector<4xindex>
            %50 = arith.select %43, %33, %cst : vector<4xi1>, vector<4xf64>
            %51 = vector.reduction <add>, %50 : vector<4xf64> into f64
            %52 = arith.select %45, %33, %cst : vector<4xi1>, vector<4xf64>
            %53 = vector.reduction <add>, %52 : vector<4xf64> into f64
            %54 = arith.select %47, %33, %cst : vector<4xi1>, vector<4xf64>
            %55 = vector.reduction <add>, %54 : vector<4xf64> into f64
            %56 = arith.select %49, %33, %cst : vector<4xi1>, vector<4xf64>
            %57 = vector.reduction <add>, %56 : vector<4xf64> into f64
            %58 = vector.extract %31[0] : f64 from vector<4xf64>
            %59 = vector.extract %31[1] : f64 from vector<4xf64>
            %60 = vector.extract %31[2] : f64 from vector<4xf64>
            %61 = vector.extract %31[3] : f64 from vector<4xf64>
            %62 = arith.mulf %58, %51 : f64
            %63 = arith.mulf %59, %53 : f64
            %64 = arith.mulf %60, %55 : f64
            %65 = arith.mulf %61, %57 : f64
            %66 = arith.addf %62, %63 : f64
            %67 = arith.addf %64, %65 : f64
            %68 = arith.addf %66, %67 : f64
            %69 = arith.addf %arg7, %68 : f64
            %70 = arith.minui %37, %41 : index
            %71 = arith.cmpi ule, %34, %70 : index
            %72 = arith.cmpi ule, %35, %70 : index
            %73 = arith.cmpi ule, %36, %70 : index
            %74 = arith.cmpi ule, %37, %70 : index
            %75 = arith.select %71, %c1, %c0 : index
            %76 = arith.select %72, %c1, %c0 : index
            %77 = arith.select %73, %c1, %c0 : index
            %78 = arith.select %74, %c1, %c0 : index
            %79 = arith.addi %75, %76 : index
            %80 = arith.addi %77, %78 : index
            %81 = arith.addi %79, %80 : index
            %82 = arith.addi %arg5, %81 : index
            %83 = arith.cmpi ule, %38, %70 : index
            %84 = arith.cmpi ule, %39, %70 : index
            %85 = arith.cmpi ule, %40, %70 : index
            %86 = arith.cmpi ule, %41, %70 : index
            %87 = arith.select %83, %c1, %c0 : index
            %88 = arith.select %84, %c1, %c0 : index
            %89 = arith.select %85, %c1, %c0 : index
            %90 = arith.select %86, %c1, %c0 : index
            %91 = arith.addi %87, %88 : index
            %92 = arith.addi %89, %90 : index
            %93 = arith.addi %91, %92 : index
            %94 = arith.addi %arg6, %93 : index
            // %81 and %93 are this iteration's arg5-side/arg6-side (B-
            // side/C-side) advance counts (0-4 each, from the
            // popcount-style select+add above) — their sum is exactly how
            // many index-stream elements this SIMD iteration consumed,
            // the same unit the scalar counter uses.
            %simd_elems_idx = arith.addi %81, %93 : index
            %simd_elems_i64 = arith.index_cast %simd_elems_idx : index to i64
            %simd_count_val = memref.load %simd_counter_ref[] : memref<i64>
            %simd_count_next = arith.addi %simd_count_val, %simd_elems_i64 : i64
            memref.store %simd_count_next, %simd_counter_ref[] : memref<i64>
            scf.yield %82, %94, %69 : index, index, f64
          }
          %29:3 = scf.while (%arg5 = %28#0, %arg6 = %28#1, %arg7 = %28#2) : (index, index, f64) -> (index, index, f64) {
            %30 = arith.cmpi ult, %arg5, %24 : index
            %31 = arith.cmpi ult, %arg6, %27 : index
            %32 = arith.andi %30, %31 : i1
            scf.condition(%32) %arg5, %arg6, %arg7 : index, index, f64
          } do {
          ^bb0(%arg5: index, %arg6: index, %arg7: f64):
            %30 = memref.load %12[%arg5] : memref<?xindex>
            %31 = memref.load %15[%arg6] : memref<?xindex>
            %32 = arith.cmpi ult, %31, %30 : index
            %33 = arith.select %32, %31, %30 : index
            %34 = arith.cmpi eq, %30, %33 : index
            %35 = arith.cmpi eq, %31, %33 : index
            // This iteration consumes 1 element from B (%12/%5, arg5
            // side) if %34, and 1 from C (%15/%6, arg6 side) if %35 —
            // never zero (at least one side always equals the min), and
            // 2 on an actual index match (both true), same asymmetric-
            // consumption logic the SIMD side's %81/%93 already track —
            // so the two counters stay apples-to-apples instead of
            // scalar just counting loop steps.
            %scalar_b_adv = arith.select %34, %scalar_one_i64, %scalar_zero_i64 : i64
            %scalar_c_adv = arith.select %35, %scalar_one_i64, %scalar_zero_i64 : i64
            %scalar_elems_this_iter = arith.addi %scalar_b_adv, %scalar_c_adv : i64
            %scalar_count_val = memref.load %scalar_counter_ref[] : memref<i64>
            %scalar_count_next = arith.addi %scalar_count_val, %scalar_elems_this_iter : i64
            memref.store %scalar_count_next, %scalar_counter_ref[] : memref<i64>
            %36 = arith.andi %34, %35 : i1
            %37 = scf.if %36 -> (f64) {
              %44 = memref.load %5[%arg5] : memref<?xf64>
              %45 = memref.load %6[%arg6] : memref<?xf64>
              %46 = arith.mulf %44, %45 : f64
              %47 = arith.addf %arg7, %46 : f64
              scf.yield %47 : f64
            } else {
              scf.yield %arg7 : f64
            }
            %38 = arith.cmpi eq, %30, %33 : index
            %39 = arith.addi %arg5, %c1 : index
            %40 = arith.select %38, %39, %arg5 : index
            %41 = arith.cmpi eq, %31, %33 : index
            %42 = arith.addi %arg6, %c1 : index
            %43 = arith.select %41, %42, %arg6 : index
            scf.yield %40, %43, %37 : index, index, f64
          } attributes {"Emitted from" = "linalg.generic", splyce.scalar_fallback}
          memref.store %29#2, %7[%arg2, %20, %arg4] : memref<?x?x?xf64>
        } {"Emitted from" = "linalg.generic"}
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %16 = bufferization.to_tensor %7 restrict : memref<?x?x?xf64> to tensor<?x?x?xf64>
    %scalar_final_count = memref.load %scalar_counter_ref[] : memref<i64>
    %simd_final_count = memref.load %simd_counter_ref[] : memref<i64>
    %fmt_scalar_count_ptr = llvm.mlir.addressof @fmt_scalar_count : !llvm.ptr
    %fmt_simd_count_ptr = llvm.mlir.addressof @fmt_simd_count : !llvm.ptr
    %print_scalar_count = llvm.call @printf(%fmt_scalar_count_ptr, %scalar_final_count) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
    %print_simd_count = llvm.call @printf(%fmt_simd_count_ptr, %simd_final_count) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
    return %16 : tensor<?x?x?xf64>
  }
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c262144_i64 = arith.constant 262144 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = llvm.mlir.addressof @fmt_time : !llvm.ptr
    %1 = llvm.mlir.addressof @mode_w : !llvm.ptr
    %2 = llvm.mlir.addressof @benchmarkFile : !llvm.ptr
    %3 = llvm.mlir.addressof @fmt_exec_time : !llvm.ptr
    %4 = llvm.mlir.addressof @fileB : !llvm.ptr
    %5 = llvm.mlir.addressof @fileC : !llvm.ptr
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %alloca = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca[%c0] : memref<?xindex>
    memref.store %c0, %alloca[%c1] : memref<?xindex>
    memref.store %c0, %alloca[%c2] : memref<?xindex>
    %6 = call @createCheckedSparseTensorReader(%4, %alloca, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %7 = call @getSparseTensorReaderDimSizes(%6) : (!llvm.ptr) -> memref<?xindex>
    %alloca_0 = memref.alloca(%c3) : memref<?xi64>
    memref.store %c65536_i64, %alloca_0[%c0] : memref<?xi64>
    memref.store %c262144_i64, %alloca_0[%c1] : memref<?xi64>
    memref.store %c262144_i64, %alloca_0[%c2] : memref<?xi64>
    %alloca_1 = memref.alloca(%c3) : memref<?xindex>
    memref.store %c0, %alloca_1[%c0] : memref<?xindex>
    memref.store %c1, %alloca_1[%c1] : memref<?xindex>
    memref.store %c2, %alloca_1[%c2] : memref<?xindex>
    %8 = call @newSparseTensor(%7, %7, %alloca_0, %alloca_1, %alloca_1, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %6) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%6) : (!llvm.ptr) -> ()
    %alloca_2 = memref.alloca(%c2) : memref<?xindex>
    memref.store %c0, %alloca_2[%c0] : memref<?xindex>
    memref.store %c0, %alloca_2[%c1] : memref<?xindex>
    %9 = call @createCheckedSparseTensorReader(%5, %alloca_2, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %10 = call @getSparseTensorReaderDimSizes(%9) : (!llvm.ptr) -> memref<?xindex>
    %11 = memref.load %10[%c0] : memref<?xindex>
    %12 = memref.load %10[%c1] : memref<?xindex>
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
    memref.store %12, %alloca_6[%c0] : memref<?xindex>
    memref.store %11, %alloca_6[%c1] : memref<?xindex>
    %13 = call @newSparseTensor(%10, %alloca_6, %alloca_3, %alloca_4, %alloca_5, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %9) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%9) : (!llvm.ptr) -> ()
    %14 = call @rtclock() : () -> f64
    %15 = call @spttspm(%8, %13) : (!llvm.ptr, !llvm.ptr) -> tensor<?x?x?xf64>
    %16 = call @rtclock() : () -> f64
    %17 = arith.subf %16, %14 : f64
    bufferization.dealloc_tensor %15 : tensor<?x?x?xf64>
    %18 = llvm.call @printf(%3, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    %19 = llvm.call @fopen(%2, %1) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %20 = llvm.call @fprintf(%19, %0, %17) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
    %21 = llvm.call @fclose(%19) : (!llvm.ptr) -> i32
    call @delSparseTensor(%8) : (!llvm.ptr) -> ()
    call @delSparseTensor(%13) : (!llvm.ptr) -> ()
    return
  }
}

