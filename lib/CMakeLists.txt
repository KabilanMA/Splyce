# lib/CMakeLists.txt
#
# Top of the library source tree.  Each transform group lives in its own
# subdirectory so that new passes can be added without touching this file —
# just create lib/NewPass/ with its own CMakeLists.txt and add it here.

add_subdirectory(Transforms)