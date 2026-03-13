add_mlir_library(SplyceTransforms
  CoIterPattern.cpp

  DEPENDS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
  MLIRSCFDialect
  MLIRSparseTensorDialect
)