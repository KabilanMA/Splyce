get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_tool(splyce-opt
  splyce-opt.cpp

  DEPENDS
  ${dialect_libs}
  ${extension_libs}
  SplyceTransforms
  MLIROptLib
  MLIRPass
)