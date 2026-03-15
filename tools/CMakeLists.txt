# tools/CMakeLists.txt
#
# Builds coiter-opt: a minimal mlir-opt clone that registers only the
# dialects and passes needed for this project.
#
# Keeping the tool definition here (rather than in the root) means:
#   • Adding a second tool (e.g. coiter-translate) is a one-file change.
#   • The tool's link deps are listed once, close to the source.
#   • CI can build just the library without the tool by excluding this
#     subdirectory from the root add_subdirectory call.

add_llvm_executable(splyce-opt splyce-opt.cpp)

target_link_libraries(splyce-opt
  PRIVATE
    # Our pass library (transitively pulls in all dialect deps via PUBLIC).
    CoIterVectorizeTransforms

    # mlir-opt infrastructure.
    MLIROptLib       # MlirOptMain, asMainReturnCode
    MLIRParser       # parseSourceFile
    MLIRSupport      # llvm::raw_ostream, StringRef, …

    # Standard utility passes (--canonicalize, --cse, etc.) useful when
    # inspecting pre/post transformation IR by hand.
    MLIRTransforms
)

# Install the tool alongside mlir-opt so it can be used in CI pipelines
# that already have an MLIR install on PATH.
install(TARGETS splyce-opt DESTINATION bin)