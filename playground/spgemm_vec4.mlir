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
    %cst = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %0 = call @sparseLvlSize(%arg0, %c0_0) : (!llvm.ptr, index) -> index
    %c0_1 = arith.constant 0 : index
    %1 = call @sparseLvlSize(%arg1, %c0_1) : (!llvm.ptr, index) -> index
    %2 = bufferization.alloc_tensor(%0, %1) : tensor<?x?xf64>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    %4 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %5 = call @sparseValuesF64(%arg1) : (!llvm.ptr) -> memref<?xf64>
    %6 = bufferization.to_buffer %3 : tensor<?x?xf64> to memref<?x?xf64>
    %c0_2 = arith.constant 0 : index
    %7 = call @sparseLvlSize(%arg0, %c0_2) : (!llvm.ptr, index) -> index
    %c1_3 = arith.constant 1 : index
    %8 = call @sparsePositions0(%arg0, %c1_3) : (!llvm.ptr, index) -> memref<?xindex>
    %c1_4 = arith.constant 1 : index
    %9 = call @sparseCoordinates0(%arg0, %c1_4) : (!llvm.ptr, index) -> memref<?xindex>
    %c0_5 = arith.constant 0 : index
    %10 = call @sparseLvlSize(%arg1, %c0_5) : (!llvm.ptr, index) -> index
    %c1_6 = arith.constant 1 : index
    %11 = call @sparsePositions0(%arg1, %c1_6) : (!llvm.ptr, index) -> memref<?xindex>
    %c1_7 = arith.constant 1 : index
    %12 = call @sparseCoordinates0(%arg1, %c1_7) : (!llvm.ptr, index) -> memref<?xindex>
    scf.for %arg2 = %c0 to %7 step %c1 {
      %14 = arith.addi %arg2, %c1 : index
      scf.for %arg3 = %c0 to %10 step %c1 {
        %15 = memref.load %6[%arg2, %arg3] : memref<?x?xf64>
        %16 = memref.load %8[%arg2] : memref<?xindex>
        %17 = memref.load %8[%14] : memref<?xindex>
        %18 = memref.load %11[%arg3] : memref<?xindex>
        %19 = arith.addi %arg3, %c1 : index
        %20 = memref.load %11[%19] : memref<?xindex>
        // SIMD fast lane: runs while BOTH rows have >= 4 elements remaining
        %vr:3 = scf.while (%arg4 = %16, %arg5 = %18, %arg6 = %15) : (index, index, f64) -> (index, index, f64) {
          %c4    = arith.constant 4 : index
          %a4p4  = arith.addi %arg4, %c4 : index
          %cond_a = arith.cmpi ule, %a4p4, %17 : index
          %b4p4  = arith.addi %arg5, %c4 : index
          %cond_b = arith.cmpi ule, %b4p4, %20 : index
          %cond   = arith.andi %cond_a, %cond_b : i1
          scf.condition(%cond) %arg4, %arg5, %arg6 : index, index, f64
        } do {
        ^bb0(%arg4: index, %arg5: index, %arg6: f64):
          %zero_vec = arith.constant dense<0.000000e+00> : vector<4xf64>

          // Load 4 coords and 4 values from A and B (safe: >= 4 guaranteed by condition)
          %a_cv = vector.load %9[%arg4]  : memref<?xindex>, vector<4xindex>
          %a_vv = vector.load %4[%arg4]  : memref<?xf64>,   vector<4xf64>
          %b_cv = vector.load %12[%arg5] : memref<?xindex>, vector<4xindex>
          %b_vv = vector.load %5[%arg5]  : memref<?xf64>,   vector<4xf64>

          // Extract individual coords for pointer advancement and cross-compare
          %a0 = vector.extract %a_cv[0] : index from vector<4xindex>
          %a1 = vector.extract %a_cv[1] : index from vector<4xindex>
          %a2 = vector.extract %a_cv[2] : index from vector<4xindex>
          %a3 = vector.extract %a_cv[3] : index from vector<4xindex>
          %b0 = vector.extract %b_cv[0] : index from vector<4xindex>
          %b1 = vector.extract %b_cv[1] : index from vector<4xindex>
          %b2 = vector.extract %b_cv[2] : index from vector<4xindex>
          %b3 = vector.extract %b_cv[3] : index from vector<4xindex>

          // 4×4 cross-comparison: row_i[j] = true iff A[i] == B[j]
          %a0_b = vector.broadcast %a0 : index to vector<4xindex>
          %row0 = arith.cmpi eq, %a0_b, %b_cv : vector<4xindex>
          %a1_b = vector.broadcast %a1 : index to vector<4xindex>
          %row1 = arith.cmpi eq, %a1_b, %b_cv : vector<4xindex>
          %a2_b = vector.broadcast %a2 : index to vector<4xindex>
          %row2 = arith.cmpi eq, %a2_b, %b_cv : vector<4xindex>
          %a3_b = vector.broadcast %a3 : index to vector<4xindex>
          %row3 = arith.cmpi eq, %a3_b, %b_cv : vector<4xindex>

          // Masked B-value extraction per A element (0.0 if no match)
          %bm0    = arith.select %row0, %b_vv, %zero_vec : vector<4xi1>, vector<4xf64>
          %b_val0 = vector.reduction <add>, %bm0 : vector<4xf64> into f64
          %bm1    = arith.select %row1, %b_vv, %zero_vec : vector<4xi1>, vector<4xf64>
          %b_val1 = vector.reduction <add>, %bm1 : vector<4xf64> into f64
          %bm2    = arith.select %row2, %b_vv, %zero_vec : vector<4xi1>, vector<4xf64>
          %b_val2 = vector.reduction <add>, %bm2 : vector<4xf64> into f64
          %bm3    = arith.select %row3, %b_vv, %zero_vec : vector<4xi1>, vector<4xf64>
          %b_val3 = vector.reduction <add>, %bm3 : vector<4xf64> into f64

          // Per-lane contribution: a_val_i * b_val_i  (0 when no match)
          %a0_val   = vector.extract %a_vv[0] : f64 from vector<4xf64>
          %a1_val   = vector.extract %a_vv[1] : f64 from vector<4xf64>
          %a2_val   = vector.extract %a_vv[2] : f64 from vector<4xf64>
          %a3_val   = vector.extract %a_vv[3] : f64 from vector<4xf64>
          %contrib0 = arith.mulf %a0_val, %b_val0 : f64
          %contrib1 = arith.mulf %a1_val, %b_val1 : f64
          %contrib2 = arith.mulf %a2_val, %b_val2 : f64
          %contrib3 = arith.mulf %a3_val, %b_val3 : f64

          // Horizontal sum and accumulate
          %s01     = arith.addf %contrib0, %contrib1 : f64
          %s23     = arith.addf %contrib2, %contrib3 : f64
          %s_total = arith.addf %s01, %s23 : f64
          %new_acc = arith.addf %arg6, %s_total : f64

          // Branchless pointer advancement via pivot = min(A_max, B_max)
          %pivot = arith.minui %a3, %b3 : index

          %a0_le  = arith.cmpi ule, %a0, %pivot : index
          %a1_le  = arith.cmpi ule, %a1, %pivot : index
          %a2_le  = arith.cmpi ule, %a2, %pivot : index
          %a3_le  = arith.cmpi ule, %a3, %pivot : index
          %a0_adv = arith.select %a0_le, %c1, %c0 : index
          %a1_adv = arith.select %a1_le, %c1, %c0 : index
          %a2_adv = arith.select %a2_le, %c1, %c0 : index
          %a3_adv = arith.select %a3_le, %c1, %c0 : index
          %a_s01  = arith.addi %a0_adv, %a1_adv : index
          %a_s23  = arith.addi %a2_adv, %a3_adv : index
          %a_step = arith.addi %a_s01, %a_s23 : index
          %new_arg4 = arith.addi %arg4, %a_step : index

          %b0_le  = arith.cmpi ule, %b0, %pivot : index
          %b1_le  = arith.cmpi ule, %b1, %pivot : index
          %b2_le  = arith.cmpi ule, %b2, %pivot : index
          %b3_le  = arith.cmpi ule, %b3, %pivot : index
          %b0_adv = arith.select %b0_le, %c1, %c0 : index
          %b1_adv = arith.select %b1_le, %c1, %c0 : index
          %b2_adv = arith.select %b2_le, %c1, %c0 : index
          %b3_adv = arith.select %b3_le, %c1, %c0 : index
          %b_s01  = arith.addi %b0_adv, %b1_adv : index
          %b_s23  = arith.addi %b2_adv, %b3_adv : index
          %b_step = arith.addi %b_s01, %b_s23 : index
          %new_arg5 = arith.addi %arg5, %b_step : index

          scf.yield %new_arg4, %new_arg5, %new_acc : index, index, f64
        } attributes {"Emitted from" = "linalg.generic"}

        // Scalar epilogue: handles remaining 0–3 elements from where the SIMD lane stopped
        %21:3 = scf.while (%arg4 = %vr#0, %arg5 = %vr#1, %arg6 = %vr#2) : (index, index, f64) -> (index, index, f64) {
          %22 = arith.cmpi ult, %arg4, %17 : index
          %23 = arith.cmpi ult, %arg5, %20 : index
          %24 = arith.andi %22, %23 : i1
          scf.condition(%24) %arg4, %arg5, %arg6 : index, index, f64
        } do {
        ^bb0(%arg4: index, %arg5: index, %arg6: f64):
          %22 = memref.load %9[%arg4] : memref<?xindex>
          %23 = memref.load %12[%arg5] : memref<?xindex>
          %24 = arith.cmpi ult, %23, %22 : index
          %25 = arith.select %24, %23, %22 : index
          %26 = arith.cmpi eq, %22, %25 : index
          %27 = arith.cmpi eq, %23, %25 : index
          %28 = arith.andi %26, %27 : i1
          %29 = scf.if %28 -> (f64) {
            %36 = memref.load %4[%arg4] : memref<?xf64>
            %37 = memref.load %5[%arg5] : memref<?xf64>
            %38 = arith.mulf %36, %37 : f64
            %39 = arith.addf %arg6, %38 : f64
            scf.yield %39 : f64
          } else {
            scf.yield %arg6 : f64
          }
          %30 = arith.cmpi eq, %22, %25 : index
          %31 = arith.addi %arg4, %c1 : index
          %32 = arith.select %30, %31, %arg4 : index
          %33 = arith.cmpi eq, %23, %25 : index
          %34 = arith.addi %arg5, %c1 : index
          %35 = arith.select %33, %34, %arg5 : index
          scf.yield %32, %35, %29 : index, index, f64
        } attributes {"Emitted from" = "linalg.generic"}
        memref.store %21#2, %6[%arg2, %arg3] : memref<?x?xf64>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %13 = bufferization.to_tensor %6 restrict : memref<?x?xf64> to tensor<?x?xf64>
    return %13 : tensor<?x?xf64>
  }
  func.func @main() {
    %0 = llvm.mlir.addressof @fmt_time : !llvm.ptr
    %1 = llvm.mlir.addressof @mode_w : !llvm.ptr
    %2 = llvm.mlir.addressof @benchmarkFile : !llvm.ptr
    %c25 = arith.constant 25 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %3 = llvm.mlir.addressof @fileA : !llvm.ptr
    %4 = llvm.mlir.addressof @fileB : !llvm.ptr
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %alloca = memref.alloca(%c2) : memref<?xindex>
    %c0_2 = arith.constant 0 : index
    memref.store %c0_0, %alloca[%c0_2] : memref<?xindex>
    %c1_3 = arith.constant 1 : index
    memref.store %c0_1, %alloca[%c1_3] : memref<?xindex>
    %c1_i32 = arith.constant 1 : i32
    %5 = call @createCheckedSparseTensorReader(%3, %alloca, %c1_i32) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %6 = call @getSparseTensorReaderDimSizes(%5) : (!llvm.ptr) -> memref<?xindex>
    %c0_4 = arith.constant 0 : index
    %7 = memref.load %6[%c0_4] : memref<?xindex>
    %c1_5 = arith.constant 1 : index
    %8 = memref.load %6[%c1_5] : memref<?xindex>
    %c65536_i64 = arith.constant 65536 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c2_6 = arith.constant 2 : index
    %alloca_7 = memref.alloca(%c2_6) : memref<?xi64>
    %c0_8 = arith.constant 0 : index
    memref.store %c65536_i64, %alloca_7[%c0_8] : memref<?xi64>
    %c1_9 = arith.constant 1 : index
    memref.store %c262144_i64, %alloca_7[%c1_9] : memref<?xi64>
    %c0_10 = arith.constant 0 : index
    %c1_11 = arith.constant 1 : index
    %c2_12 = arith.constant 2 : index
    %alloca_13 = memref.alloca(%c2_12) : memref<?xindex>
    %c0_14 = arith.constant 0 : index
    memref.store %c0_10, %alloca_13[%c0_14] : memref<?xindex>
    %c1_15 = arith.constant 1 : index
    memref.store %c1_11, %alloca_13[%c1_15] : memref<?xindex>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_16 = arith.constant 0 : i32
    %c1_i32_17 = arith.constant 1 : i32
    %c1_i32_18 = arith.constant 1 : i32
    %9 = call @newSparseTensor(%6, %6, %alloca_7, %alloca_13, %alloca_13, %c0_i32, %c0_i32_16, %c1_i32_17, %c1_i32_18, %5) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%5) : (!llvm.ptr) -> ()
    %c0_19 = arith.constant 0 : index
    %c0_20 = arith.constant 0 : index
    %c2_21 = arith.constant 2 : index
    %alloca_22 = memref.alloca(%c2_21) : memref<?xindex>
    %c0_23 = arith.constant 0 : index
    memref.store %c0_19, %alloca_22[%c0_23] : memref<?xindex>
    %c1_24 = arith.constant 1 : index
    memref.store %c0_20, %alloca_22[%c1_24] : memref<?xindex>
    %c1_i32_25 = arith.constant 1 : i32
    %10 = call @createCheckedSparseTensorReader(%4, %alloca_22, %c1_i32_25) : (!llvm.ptr, memref<?xindex>, i32) -> !llvm.ptr
    %11 = call @getSparseTensorReaderDimSizes(%10) : (!llvm.ptr) -> memref<?xindex>
    %c0_26 = arith.constant 0 : index
    %12 = memref.load %11[%c0_26] : memref<?xindex>
    %c1_27 = arith.constant 1 : index
    %13 = memref.load %11[%c1_27] : memref<?xindex>
    %c65536_i64_28 = arith.constant 65536 : i64
    %c262144_i64_29 = arith.constant 262144 : i64
    %c2_30 = arith.constant 2 : index
    %alloca_31 = memref.alloca(%c2_30) : memref<?xi64>
    %c0_32 = arith.constant 0 : index
    memref.store %c65536_i64_28, %alloca_31[%c0_32] : memref<?xi64>
    %c1_33 = arith.constant 1 : index
    memref.store %c262144_i64_29, %alloca_31[%c1_33] : memref<?xi64>
    %c1_34 = arith.constant 1 : index
    %c0_35 = arith.constant 0 : index
    %c1_36 = arith.constant 1 : index
    %c0_37 = arith.constant 0 : index
    %c2_38 = arith.constant 2 : index
    %alloca_39 = memref.alloca(%c2_38) : memref<?xindex>
    %c0_40 = arith.constant 0 : index
    memref.store %c1_34, %alloca_39[%c0_40] : memref<?xindex>
    %c1_41 = arith.constant 1 : index
    memref.store %c0_35, %alloca_39[%c1_41] : memref<?xindex>
    %c2_42 = arith.constant 2 : index
    %alloca_43 = memref.alloca(%c2_42) : memref<?xindex>
    %c0_44 = arith.constant 0 : index
    memref.store %c1_36, %alloca_43[%c0_44] : memref<?xindex>
    %c1_45 = arith.constant 1 : index
    memref.store %c0_37, %alloca_43[%c1_45] : memref<?xindex>
    %c2_46 = arith.constant 2 : index
    %alloca_47 = memref.alloca(%c2_46) : memref<?xindex>
    %c0_48 = arith.constant 0 : index
    memref.store %13, %alloca_47[%c0_48] : memref<?xindex>
    %c1_49 = arith.constant 1 : index
    memref.store %12, %alloca_47[%c1_49] : memref<?xindex>
    %c0_i32_50 = arith.constant 0 : i32
    %c0_i32_51 = arith.constant 0 : i32
    %c1_i32_52 = arith.constant 1 : i32
    %c1_i32_53 = arith.constant 1 : i32
    %14 = call @newSparseTensor(%11, %alloca_47, %alloca_31, %alloca_39, %alloca_43, %c0_i32_50, %c0_i32_51, %c1_i32_52, %c1_i32_53, %10) : (memref<?xindex>, memref<?xindex>, memref<?xi64>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    call @delSparseTensorReader(%10) : (!llvm.ptr) -> ()
    %15 = call @spgemm(%9, %14) : (!llvm.ptr, !llvm.ptr) -> tensor<?x?xf64>
    bufferization.dealloc_tensor %15 : tensor<?x?xf64>
    %alloc = memref.alloc() : memref<25xf64>
    scf.for %arg0 = %c0 to %c25 step %c1 {
      %18 = func.call @rtclock() : () -> f64
      %19 = func.call @spgemm(%9, %14) : (!llvm.ptr, !llvm.ptr) -> tensor<?x?xf64>
      %20 = func.call @rtclock() : () -> f64
      %21 = arith.subf %20, %18 : f64
      func.call @printF64(%21) : (f64) -> ()
      func.call @printNewline() : () -> ()
      memref.store %21, %alloc[%arg0] : memref<25xf64>
      bufferization.dealloc_tensor %19 : tensor<?x?xf64>
    }
    %16 = llvm.call @fopen(%2, %1) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    scf.for %arg0 = %c0 to %c25 step %c1 {
      %18 = memref.load %alloc[%arg0] : memref<25xf64>
      %19 = llvm.call @fprintf(%16, %0, %18) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
    }
    %17 = llvm.call @fclose(%16) : (!llvm.ptr) -> i32
    memref.dealloc %alloc : memref<25xf64>
    call @delSparseTensor(%9) : (!llvm.ptr) -> ()
    call @delSparseTensor(%14) : (!llvm.ptr) -> ()
    return
  }
}

