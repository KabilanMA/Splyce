module {
  func.func @coiter_basic(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<1024xf32>, %arg5: index, %arg6: index) {
    %cst = arith.constant dense<false> : vector<8xi1>
    %cst_0 = arith.constant dense<0> : vector<8xindex>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<8xf32>
    %c8 = arith.constant 8 : index
    %c51200 = arith.constant 51200 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %0 = arith.muli %arg5, %c100 : index
    %1 = arith.cmpi uge, %0, %c51200 : index
    %2:2 = scf.if %1 -> (index, index) {
      %3:2 = scf.while (%arg7 = %c0, %arg8 = %c0) : (index, index) -> (index, index) {
        %4 = arith.cmpi ult, %arg7, %arg5 : index
        %5 = arith.cmpi ult, %arg8, %arg6 : index
        %6 = arith.andi %4, %5 : i1
        scf.condition(%6) %arg7, %arg8 : index, index
      } do {
      ^bb0(%arg7: index, %arg8: index):
        %4 = arith.subi %arg5, %arg7 : index
        %5 = arith.minui %4, %c8 : index
        %6 = vector.create_mask %5 : vector<8xi1>
        %7 = arith.subi %arg6, %arg8 : index
        %8 = arith.minui %7, %c8 : index
        %9 = vector.create_mask %8 : vector<8xi1>
        %10 = vector.maskedload %arg0[%arg7], %6, %cst_0 : memref<?xindex>, vector<8xi1>, vector<8xindex> into vector<8xindex>
        %11 = vector.maskedload %arg2[%arg7], %6, %cst_1 : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
        %12 = vector.maskedload %arg1[%arg8], %9, %cst_0 : memref<?xindex>, vector<8xi1>, vector<8xindex> into vector<8xindex>
        %13 = vector.maskedload %arg3[%arg8], %9, %cst_1 : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
        %14 = vector.extract %10[0] : index from vector<8xindex>
        %15 = vector.broadcast %14 : index to vector<8xindex>
        %16 = arith.cmpi eq, %15, %12 : vector<8xindex>
        %17 = arith.andi %16, %9 : vector<8xi1>
        %18 = vector.reduction <or>, %17 : vector<8xi1> into i1
        %19 = arith.select %17, %13, %cst_1 : vector<8xi1>, vector<8xf32>
        %20 = vector.reduction <add>, %19 : vector<8xf32> into f32
        %21 = vector.insert %20, %cst_1 [0] : f32 into vector<8xf32>
        %22 = vector.insert %18, %cst [0] : i1 into vector<8xi1>
        %23 = vector.extract %10[1] : index from vector<8xindex>
        %24 = vector.broadcast %23 : index to vector<8xindex>
        %25 = arith.cmpi eq, %24, %12 : vector<8xindex>
        %26 = arith.andi %25, %9 : vector<8xi1>
        %27 = vector.reduction <or>, %26 : vector<8xi1> into i1
        %28 = arith.select %26, %13, %cst_1 : vector<8xi1>, vector<8xf32>
        %29 = vector.reduction <add>, %28 : vector<8xf32> into f32
        %30 = vector.insert %29, %21 [1] : f32 into vector<8xf32>
        %31 = vector.insert %27, %22 [1] : i1 into vector<8xi1>
        %32 = vector.extract %10[2] : index from vector<8xindex>
        %33 = vector.broadcast %32 : index to vector<8xindex>
        %34 = arith.cmpi eq, %33, %12 : vector<8xindex>
        %35 = arith.andi %34, %9 : vector<8xi1>
        %36 = vector.reduction <or>, %35 : vector<8xi1> into i1
        %37 = arith.select %35, %13, %cst_1 : vector<8xi1>, vector<8xf32>
        %38 = vector.reduction <add>, %37 : vector<8xf32> into f32
        %39 = vector.insert %38, %30 [2] : f32 into vector<8xf32>
        %40 = vector.insert %36, %31 [2] : i1 into vector<8xi1>
        %41 = vector.extract %10[3] : index from vector<8xindex>
        %42 = vector.broadcast %41 : index to vector<8xindex>
        %43 = arith.cmpi eq, %42, %12 : vector<8xindex>
        %44 = arith.andi %43, %9 : vector<8xi1>
        %45 = vector.reduction <or>, %44 : vector<8xi1> into i1
        %46 = arith.select %44, %13, %cst_1 : vector<8xi1>, vector<8xf32>
        %47 = vector.reduction <add>, %46 : vector<8xf32> into f32
        %48 = vector.insert %47, %39 [3] : f32 into vector<8xf32>
        %49 = vector.insert %45, %40 [3] : i1 into vector<8xi1>
        %50 = vector.extract %10[4] : index from vector<8xindex>
        %51 = vector.broadcast %50 : index to vector<8xindex>
        %52 = arith.cmpi eq, %51, %12 : vector<8xindex>
        %53 = arith.andi %52, %9 : vector<8xi1>
        %54 = vector.reduction <or>, %53 : vector<8xi1> into i1
        %55 = arith.select %53, %13, %cst_1 : vector<8xi1>, vector<8xf32>
        %56 = vector.reduction <add>, %55 : vector<8xf32> into f32
        %57 = vector.insert %56, %48 [4] : f32 into vector<8xf32>
        %58 = vector.insert %54, %49 [4] : i1 into vector<8xi1>
        %59 = vector.extract %10[5] : index from vector<8xindex>
        %60 = vector.broadcast %59 : index to vector<8xindex>
        %61 = arith.cmpi eq, %60, %12 : vector<8xindex>
        %62 = arith.andi %61, %9 : vector<8xi1>
        %63 = vector.reduction <or>, %62 : vector<8xi1> into i1
        %64 = arith.select %62, %13, %cst_1 : vector<8xi1>, vector<8xf32>
        %65 = vector.reduction <add>, %64 : vector<8xf32> into f32
        %66 = vector.insert %65, %57 [5] : f32 into vector<8xf32>
        %67 = vector.insert %63, %58 [5] : i1 into vector<8xi1>
        %68 = vector.extract %10[6] : index from vector<8xindex>
        %69 = vector.broadcast %68 : index to vector<8xindex>
        %70 = arith.cmpi eq, %69, %12 : vector<8xindex>
        %71 = arith.andi %70, %9 : vector<8xi1>
        %72 = vector.reduction <or>, %71 : vector<8xi1> into i1
        %73 = arith.select %71, %13, %cst_1 : vector<8xi1>, vector<8xf32>
        %74 = vector.reduction <add>, %73 : vector<8xf32> into f32
        %75 = vector.insert %74, %66 [6] : f32 into vector<8xf32>
        %76 = vector.insert %72, %67 [6] : i1 into vector<8xi1>
        %77 = vector.extract %10[7] : index from vector<8xindex>
        %78 = vector.broadcast %77 : index to vector<8xindex>
        %79 = arith.cmpi eq, %78, %12 : vector<8xindex>
        %80 = arith.andi %79, %9 : vector<8xi1>
        %81 = vector.reduction <or>, %80 : vector<8xi1> into i1
        %82 = arith.select %80, %13, %cst_1 : vector<8xi1>, vector<8xf32>
        %83 = vector.reduction <add>, %82 : vector<8xf32> into f32
        %84 = vector.insert %83, %75 [7] : f32 into vector<8xf32>
        %85 = vector.insert %81, %76 [7] : i1 into vector<8xi1>
        %86 = arith.andi %85, %6 : vector<8xi1>
        %87 = arith.mulf %11, %84 : vector<8xf32>
        %88 = arith.addf %87, %11 : vector<8xf32>
        %89 = arith.mulf %88, %84 : vector<8xf32>
        %90 = arith.addf %89, %11 : vector<8xf32>
        %91 = arith.mulf %90, %84 : vector<8xf32>
        vector.scatter %arg4[%c0] [%10], %86, %91 : memref<1024xf32>, vector<8xindex>, vector<8xi1>, vector<8xf32>
        %92 = vector.reduction <maxui>, %10 : vector<8xindex> into index
        %93 = vector.reduction <maxui>, %12 : vector<8xindex> into index
        %94 = arith.minui %92, %93 : index
        %95 = vector.broadcast %94 : index to vector<8xindex>
        %96 = arith.cmpi ule, %10, %95 : vector<8xindex>
        %97 = arith.andi %96, %6 : vector<8xi1>
        %98 = arith.extui %97 : vector<8xi1> to vector<8xi64>
        %99 = vector.reduction <add>, %98 : vector<8xi64> into i64
        %100 = arith.index_cast %99 : i64 to index
        %101 = arith.addi %arg7, %100 : index
        %102 = arith.cmpi ule, %12, %95 : vector<8xindex>
        %103 = arith.andi %102, %9 : vector<8xi1>
        %104 = arith.extui %103 : vector<8xi1> to vector<8xi64>
        %105 = vector.reduction <add>, %104 : vector<8xi64> into i64
        %106 = arith.index_cast %105 : i64 to index
        %107 = arith.addi %arg8, %106 : index
        scf.yield %101, %107 : index, index
      }
      scf.yield %3#0, %3#1 : index, index
    } else {
      %3:2 = scf.while (%arg7 = %c0, %arg8 = %c0) : (index, index) -> (index, index) {
        %4 = arith.cmpi ult, %arg7, %arg5 : index
        %5 = arith.cmpi ult, %arg8, %arg6 : index
        %6 = arith.andi %4, %5 : i1
        scf.condition(%6) %arg7, %arg8 : index, index
      } do {
      ^bb0(%arg7: index, %arg8: index):
        %4 = memref.load %arg0[%arg7] : memref<?xindex>
        %5 = memref.load %arg1[%arg8] : memref<?xindex>
        %6 = arith.cmpi ult, %5, %4 : index
        %7 = arith.select %6, %5, %4 : index
        %8 = arith.cmpi eq, %4, %7 : index
        %9 = arith.cmpi eq, %5, %7 : index
        %10 = arith.andi %8, %9 : i1
        scf.if %10 {
          %15 = memref.load %arg2[%arg7] : memref<?xf32>
          %16 = memref.load %arg3[%arg8] : memref<?xf32>
          %17 = arith.mulf %15, %16 : f32
          %18 = arith.addf %17, %15 : f32
          %19 = arith.mulf %18, %16 : f32
          %20 = arith.addf %19, %15 : f32
          %21 = arith.mulf %20, %16 : f32
          memref.store %21, %arg4[%7] : memref<1024xf32>
        }
        %11 = arith.addi %arg7, %c1 : index
        %12 = arith.addi %arg8, %c1 : index
        %13 = arith.select %8, %11, %arg7 : index
        %14 = arith.select %9, %12, %arg8 : index
        scf.yield %13, %14 : index, index
      } attributes {splyce.scalar_fallback}
      scf.yield %3#0, %3#1 : index, index
    }
    return
  }
}

