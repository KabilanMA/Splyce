add_mlir_library(SplyceTransforms
  VectorizeSpCoIt.cpp

  DEPENDS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
  MLIRSCFDialect
  MLIRSparseTensorDialect
)