#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register all standard MLIR dialects (Vector, SCF, Arith, SparseTensor, etc.)
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  // TODO: Register your custom Splyce passes here later

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Splyce Optimizer Driver\n", registry));
}