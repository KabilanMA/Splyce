// tools/splyce-opt.cpp

// Usage:
//   splyce-opt --splyce="vector-width=8" input.mlir
//   splyce-opt --splyce="vector-width=8 min-density=0.1" input.mlir


#include "Transforms/CoIterVectorizePass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register only what the pass needs.
  registry.insert<mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect>();

  // Register the pass so --splyce is visible on the CLI.
  mlir::registerCoIterVectorizePass();

  // Standard utility passes (CSE, canonicalize) useful for pre/post inspection.
  mlir::registerTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Splyce Vectorize Tool\n", registry));
}