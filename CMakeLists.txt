CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(singa)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")
#message(STATUS "${CMAKE_CXX_FLAGS}")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Thirdparty)
#message(STATUS "module path: ${CMAKE_MODULE_PATH}")

# Flags
IF(UNIX OR APPLE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
ENDIF()

INCLUDE(cmake/ProtoBuf.cmake)
INCLUDE(cmake/Cuda.cmake)
# Includes
SET(SINGA_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${SINGA_INCLUDE_DIR})
#INCLUDE_DIRECTORIES(${singa_include_dir} ${PROJECT_BINARY_DIR})
#INCLUDE_DIRECTORIES("/home/wangwei/local/cudnn5/include" "/usr/local/cuda/include")


SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#SET(singa_linker_lib cudnn)
#LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH} "/home/wangwei/local/cudnn5/lib64/")

#message(STATUS "include_dir: ${SINGA_INCLUDE_DIR}")


ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
