# tools/CMakeLists.txt
#
# Builds splyce-opt: a minimal mlir-opt clone that registers only the
# dialects and passes needed for this project.
#
# Keeping the tool definition here (rather than in the root) means:
#   - Adding a second tool (e.g. splyce-translate) is a one-file change.
#   - The tool's link deps are listed once, close to the source.
#   - CI can build just the library without the tool by excluding this
#     subdirectory from the root add_subdirectory call.

add_llvm_executable(splyce-opt splyce-opt.cpp PARTIAL_SOURCES_INTENDED)

target_link_libraries(splyce-opt
  PRIVATE
    # Our pass library (transitively pulls in all dialect deps via PUBLIC).
    CoIterVectorizeTransforms

    # mlir-opt infrastructure.
    MLIROptLib       # MlirOptMain, asMainReturnCode
    MLIRParser       # parseSourceFile
    MLIRSupport      # llvm::raw_ostream, StringRef, ...

    # Standard utility passes (--canonicalize, --cse, etc.) useful when
    # inspecting pre/post transformation IR by hand.
    MLIRTransforms

    # Extra dialects needed to parse real sparsification output
    # (e.g. mttkrp_scf.mlir produced by --sparsification-and-bufferization).
    MLIRLLVMDialect
    MLIRLinalgDialect
    MLIRSparseTensorDialect

    # Dialects introduced transiently by the --lower-to-llvm pipeline
    # (complex-to-standard/libm, scf-to-openmp/openmp-to-llvm).
    MLIRComplexDialect
    MLIROpenMPDialect

    # Transform passes bundled by --sparsify-to-scf (linalg/sparse_tensor ->
    # scf lowering), so the whole README pipeline can run in one invocation.
    MLIRLinalgTransforms
    MLIRSparseTensorTransforms
    MLIRBufferizationTransforms

    # BufferizableOpInterface external models needed by one-shot-bufferize
    # (part of --lower-to-llvm) for tensor-typed ops from these dialects.
    MLIRTensorTransforms
    MLIRSCFTransforms

    # Transform + conversion passes bundled by --lower-to-llvm (scf -> llvm
    # dialect), single-threaded variant.
    MLIRVectorTransforms
    MLIRMemRefTransforms
    MLIRBufferizationToMemRef
    MLIRVectorToSCF
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRMemRefToLLVM
    MLIRVectorToLLVMPass
    MLIRMathToLLVM
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRControlFlowToLLVM
    MLIRReconcileUnrealizedCasts

    # Extra transform + conversion passes for --lower-to-llvm --parallelization
    # (OpenMP variant).
    MLIRArithTransforms
    MLIRSCFToOpenMP
    MLIROpenMPToLLVM
    MLIRComplexToStandard
    MLIRComplexToLibm
    MLIRConvertToLLVMPass
    MLIRUBDialect
    MLIRUBToLLVM

    # ConvertToLLVMPatternInterface external models, needed by the generic
    # --convert-to-llvm pass for every dialect present at that point in the
    # OpenMP pipeline (not just the ones with ops still needing conversion).
    MLIRComplexToLLVM
    MLIRIndexToLLVM
    MLIRControlFlowDialect
    MLIRIndexDialect
)

# Install the tool alongside mlir-opt so it can be used in CI pipelines
# that already have an MLIR install on PATH.
install(TARGETS splyce-opt DESTINATION bin)

# splyce-translate: a minimal mlir-translate clone registering only the
# --mlir-to-llvmir translation, the final step of the README pipeline.
add_llvm_executable(splyce-translate splyce-translate.cpp PARTIAL_SOURCES_INTENDED)

target_link_libraries(splyce-translate
  PRIVATE
    MLIRTranslateLib                    # mlirTranslateMain
    MLIRToLLVMIRTranslationRegistration # registerToLLVMIRTranslation
)

install(TARGETS splyce-translate DESTINATION bin)