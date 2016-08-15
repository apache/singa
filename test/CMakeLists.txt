INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)

IF(ENABLE_DIST)
  ADD_EXECUTABLE(test_ep "singa/test_ep.cc")
  ADD_DEPENDENCIES(test_ep singa_io)
  TARGET_LINK_LIBRARIES(test_ep singa_utils singa_io protobuf ${SINGA_LINKER_LIBS})
ENDIF()

ADD_LIBRARY(gtest STATIC EXCLUDE_FROM_ALL "gtest/gtest.h" "gtest/gtest-all.cc")

AUX_SOURCE_DIRECTORY(singa singa_test_source)
LIST(REMOVE_ITEM singa_test_source "singa/test_ep.cc")

IF(NOT USE_OPENCL)
    MESSAGE(STATUS "Skipping OpenCL tests")
    LIST(REMOVE_ITEM singa_test_source "singa/test_opencl.cc")
ENDIF()


ADD_EXECUTABLE(test_singa "gtest/gtest_main.cc" ${singa_test_source})
ADD_DEPENDENCIES(test_singa singa_core singa_utils)
#MESSAGE(STATUS "link libs" ${singa_linker_libs})
TARGET_LINK_LIBRARIES(test_singa gtest singa_core singa_utils singa_model
    singa_io proto protobuf ${SINGA_LINKER_LIBS})
IF(UNIX AND (NOT APPLE))
    LIST(APPEND LINK_FLAGS "-pthread")
ENDIF()
SET_TARGET_PROPERTIES(test_singa PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

