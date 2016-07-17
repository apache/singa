INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
ADD_LIBRARY(gtest STATIC EXCLUDE_FROM_ALL "gtest/gtest.h" "gtest/gtest-all.cc")

AUX_SOURCE_DIRECTORY(singa singa_test_source)

IF(NOT BUILD_OPENCL_TESTS)
    MESSAGE(STATUS "Skipping OpenCL tests")
    LIST(REMOVE_ITEM singa_test_source "singa/test_opencl.cc")
ENDIF()

ADD_EXECUTABLE(test_singa "gtest/gtest_main.cc" ${singa_test_source}) 
ADD_DEPENDENCIES(test_singa singa_core singa_utils)
MESSAGE(STATUS "link libs" ${singa_linker_libs})
TARGET_LINK_LIBRARIES(test_singa gtest singa_core singa_utils singa_model
    singa_io proto protobuf ${SINGA_LINKER_LIBS})
SET_TARGET_PROPERTIES(test_singa PROPERTIES LINK_FLAGS "${LINK_FLAGS} -pthread")
