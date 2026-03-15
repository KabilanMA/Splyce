# include/CMakeLists.txt
#
# Header-only subdirectory.  The only build artifact produced here is the
# TableGen-generated Passes.h.inc; everything else is a plain .h file that
# does not need to be registered with CMake.
#
# Subdirectories mirror the include/ layout so each dialect/transform
# group can own its own TableGen invocations independently.
 
add_subdirectory(Transforms)