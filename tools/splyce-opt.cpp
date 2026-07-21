#include "Transforms/BufferizationPrepPass.h"
#include "Transforms/CoIterVectorizePass.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ComplexToLibm/ComplexToLibm.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"

// Selects between the sequential and parallel (OpenMP-ready) variants of
// --sparsify-to-scf. Off by default, matching the README's Example 1
// (single-threaded) pipeline; pass --parallelization to switch to the
// Example 2 (multi-threaded) pipeline instead.
static llvm::cl::opt<bool> clParallelization(
    "parallelization",
    llvm::cl::desc(
        "Use the dense-outer-loop parallelization strategy in "
        "--sparsify-to-scf: passes "
        "parallelization-strategy=dense-outer-loop to --sparsification and "
        "re-runs --sparse-reinterpret-map after --lower-sparse-ops-to-foreach, "
        "matching the README's multi-threaded (OpenMP) pipeline instead of "
        "the default single-threaded one"),
    llvm::cl::init(false));

// Bundles the standard linalg/sparse_tensor -> scf lowering pipeline into a
// single flag, so it can run in the same splyce-opt invocation as --splyce:
//
//   splyce-opt in.mlir --sparsify-to-scf --splyce="..." -o out.mlir
//
// With --parallelization, follows the README's Example 2 pipeline instead
// (parallelization-strategy=dense-outer-loop, plus the extra
// --sparse-reinterpret-map re-run before lowering foreach to scf).
static void buildSparsifyToSCFPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createPreSparsificationRewritePass());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::createSparseReinterpretMapPass());

  if (clParallelization) {
    mlir::SparsificationOptions options;
    options.parallelizationStrategy =
        mlir::SparseParallelizationStrategy::kDenseOuterLoop;
    pm.addPass(mlir::createSparsificationPass(options));
  } else {
    pm.addPass(mlir::createSparsificationPass());
  }

  pm.addPass(mlir::createStageSparseOperationsPass());
  pm.addPass(mlir::createLowerSparseOpsToForeachPass());

  if (clParallelization)
    pm.addPass(mlir::createSparseReinterpretMapPass());

  pm.addPass(mlir::createLowerForeachToSCFPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSparseTensorConversionPass());
}

// Bundles the scf -> llvm lowering pipeline into a single flag, so the whole
// README pipeline (sparsify, optionally vectorize, lower to LLVM) can run in
// one splyce-opt invocation:
//
//   splyce-opt in.mlir --sparsify-to-scf --splyce="..." \
//       --lower-to-llvm -o out.mlir
//
// With --parallelization, follows the README's Example 2.3 pipeline instead
// (sparsification-and-bufferization, convert-scf-to-openmp,
// convert-openmp-to-llvm, ...) matching the OpenMP-lowered IR that
// --sparsify-to-scf --parallelization produces upstream.
static void buildLowerToLLVMPipeline(mlir::OpPassManager &pm) {
  if (clParallelization) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSparsificationAndBufferizationPass());
    pm.addPass(mlir::createStorageSpecifierToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createConvertVectorToSCFPass());
    pm.addPass(mlir::memref::createExpandReallocPass());
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    pm.addPass(mlir::createConvertOpenMPToLLVMPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createConvertVectorToLLVMPass());
    pm.addPass(mlir::createConvertComplexToStandardPass());
    pm.addPass(mlir::arith::createArithExpandOpsPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertComplexToLibm());
    pm.addPass(mlir::createConvertVectorToLLVMPass());
    pm.addPass(mlir::createConvertToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  } else {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());

    mlir::bufferization::OneShotBufferizePassOptions bufferizeOptions;
    bufferizeOptions.bufferizeFunctionBoundaries = true;
    bufferizeOptions.allowReturnAllocsFromLoops = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));

    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
    pm.addPass(mlir::vector::createLowerVectorMaskPass());
    pm.addPass(mlir::createConvertVectorToSCFPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::memref::createExpandReallocPass());
    pm.addPass(mlir::createStorageSpecifierToLLVMPass());
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertVectorToLLVMPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Core dialects needed by the pass.
  // The broader set (linalg, llvm, sparse_tensor) allows the tool to parse
  // real sparsification output (e.g. mttkrp_scf.mlir) without --allow-unregistered-dialect.
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::complex::ComplexDialect,
                  mlir::func::FuncDialect,
                  mlir::index::IndexDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect,
                  mlir::omp::OpenMPDialect,
                  mlir::scf::SCFDialect,
                  mlir::sparse_tensor::SparseTensorDialect,
                  mlir::tensor::TensorDialect,
                  mlir::ub::UBDialect,
                  mlir::vector::VectorDialect>();

  // The generic --convert-to-llvm pass (--lower-to-llvm --parallelization)
  // resolves per-dialect lowering via ConvertToLLVMPatternInterface: every
  // dialect present in the module at that point must have this registered,
  // not just the ones with ops still needing conversion.
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertOpenMPToLLVMInterface(registry);

  // --lower-to-llvm's one-shot-bufferize step needs BufferizableOpInterface
  // external models for every dialect with tensor-typed ops appearing before
  // bufferization (func, tensor, linalg, scf, arith, vector, sparse_tensor).
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);

  // Register passes so they are visible on the CLI.
  mlir::registerAddRestrictToTensorPass();
  mlir::registerCoIterVectorizePass();

  // Standard utility passes (CSE, canonicalize) useful for pre/post inspection.
  mlir::registerTransformsPasses();

  // One-flag linalg/sparse_tensor -> scf lowering pipeline
  mlir::PassPipelineRegistration<>(
      "sparsify-to-scf",
      "Lower linalg + sparse_tensor to scf (the standard sparsification "
      "pipeline, bundled into one flag)",
      buildSparsifyToSCFPipeline);

  // One-flag scf -> llvm lowering pipeline
  mlir::PassPipelineRegistration<>(
      "lower-to-llvm",
      "Lower scf to the LLVM dialect (the standard post-Splyce lowering "
      "pipeline, bundled into one flag). Combine with --parallelization for "
      "the OpenMP variant.",
      buildLowerToLLVMPipeline);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Splyce Vectorize Tool\n", registry));
}