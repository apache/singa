INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
ADD_LIBRARY(gtest STATIC EXCLUDE_FROM_ALL "gtest/gtest.h" "gtest/gtest-all.cc")

AUX_SOURCE_DIRECTORY(singa singa_test_source)

ADD_EXECUTABLE(test_singa "gtest/gtest_main.cc" ${singa_test_source})
ADD_DEPENDENCIES(test_singa singa_core singa_utils)
MESSAGE(STATUS "link libs" ${singa_linker_libs})
TARGET_LINK_LIBRARIES(test_singa gtest singa_core singa_utils proto protobuf glog)
SET_TARGET_PROPERTIES(test_singa PROPERTIES LINK_FLAGS "${LINK_FLAGS} -pthread")
