#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(singa)
SET(PACKAGE_VERSION "1.0.0")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2 ")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Thirdparty)
#message(STATUS "module path: ${CMAKE_MODULE_PATH}")

# Flags
IF(UNIX OR APPLE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
ENDIF()
IF(CMAKE_BUILD_TYPE=Debug)
  SET(NVCC_FLAG "${NVCC_FLAG} -g -G ")
ENDIF()
#message(STATUS "${CMAKE_CXX_FLAGS}")
SET(SINGA_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/include;${CMAKE_SOURCE_DIR}/lib/cnmem/include;${PROJECT_BINARY_DIR}")
INCLUDE_DIRECTORIES(${SINGA_INCLUDE_DIR})

OPTION(USE_CBLAS "Use CBlas libs" ON)
OPTION(USE_CUDA "Use Cuda libs" OFF)
OPTION(USE_CUDNN "Use Cudnn libs" ON)
OPTION(USE_OPENCV "Use opencv" OFF)
OPTION(USE_LMDB "Use LMDB libs" OFF)
OPTION(USE_PYTHON "Generate py wrappers" OFF)
#OPTION(USE_OPENCL "Use OpenCL" OFF)
OPTION(ENABLE_DIST "enable distributed training" OFF)
#OPTION(BUILD_OPENCL_TESTS "Build OpenCL tests" OFF)

INCLUDE("cmake/Dependencies.cmake")
INCLUDE("cmake/Utils.cmake")
ADD_DEFINITIONS(-DUSE_CMAKE)
#message(STATUS "${SINGA_INCLUDE_DIR}")

CONFIGURE_FILE (
    "${PROJECT_SOURCE_DIR}/cmake/Templates/singa_config.h.in"
    "${PROJECT_BINARY_DIR}/include/singa/singa_config.h")

#set(SINGA_CONFIGURE_SRC "${PROJECT_BINARY_DIR}/singa_config.h")
#LIST(APPEND SRCS ${SINGA_CONFIGURE_SRCS} ${PROJECT_BINARY_DIR}/singa_config.h)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

IF (USE_CUDA)
    ADD_SUBDIRECTORY(lib/cnmem)
    LIST(APPEND SINGA_LINKER_LIBS cnmem)
ENDIF()

# TODO(wangwei) detect the ev lib
IF (ENABLE_DIST)
  LIST(APPEND SINGA_LINKER_LIBS ev)
ENDIF()

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(examples)

INSTALL(DIRECTORY include/singa DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
INSTALL(FILES ${CMAKE_BINARY_DIR}/include/singa/singa_config.h DESTINATION
  ${CMAKE_INSTALL_PREFIX}/include/singa)
INSTALL (DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION
  ${CMAKE_INSTALL_PREFIX}
  PATTERN "*libgtest.a" EXCLUDE )
