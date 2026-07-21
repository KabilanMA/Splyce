// tools/splyce-translate.cpp

// A minimal mlir-translate clone that registers only the translation this
// project needs: --mlir-to-llvmir
//
// Usage:
//   splyce-translate in.mlir --mlir-to-llvmir -o out.ll

#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::registerToLLVMIRTranslation();
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "Splyce Translate Tool\n"));
}
