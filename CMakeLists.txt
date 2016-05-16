CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(singa)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -DUSE_CUDA -DUSE_CUDNN")

# Flags
IF(UNIX OR APPLE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
ENDIF()

# Includes
SET(singa_include_dir ${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${singa_include_dir} ${PROJECT_BINARY_DIR})
INCLUDE_DIRECTORIES("/home/wangwei/local/cudnn5/include" "/usr/local/cuda/include")


SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(singa_linker_lib cudnn)
LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH} "/home/wangwei/local/cudnn5/lib64/")

INCLUDE(cmake/ProtoBuf.cmake)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
