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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(singa)
SET(PACKAGE_VERSION "1.1.0")
SET(SINGA_MAJOR_VERSION 1)  # 0 -
SET(SINGA_MINOR_VERSION 1)  # 0 - 9
SET(SINGA_PATCH_VERSION 0)  # 0 - 99
MATH(EXPR SINGA_VERSION "${SINGA_MAJOR_VERSION} * 1000 + ${SINGA_MINOR_VERSION} * 100 + ${SINGA_PATCH_VERSION}")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Thirdparty)
#message(STATUS "module path: ${CMAKE_MODULE_PATH}")

# Flags
IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2 -fPIC -Wall -pthread")
ENDIF()
IF(APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++ -g -O2 -fPIC -Wall -pthread")
    SET(CMAKE_MACOSX_RPATH 1)
ENDIF()

IF(WIN32)
    IF (MSVC)
        MESSAGE(STATUS "Using msvc compiler")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SCL_SECURE_NO_WARNINGS")
        #MESSAGE(STATUS "Using msvc compiler ${CMAKE_CXX_FLAGS}")
    ENDIF()
ENDIF()
IF(CMAKE_BUILD_TYPE=Debug)
    SET(NVCC_FLAG "${NVCC_FLAG} -g -G ")
ENDIF()
#message(STATUS "${CMAKE_CXX_FLAGS}")
SET(SINGA_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/include;${PROJECT_BINARY_DIR}")
INCLUDE_DIRECTORIES(${SINGA_INCLUDE_DIR})

#OPTION(USE_CBLAS "Use CBlas libs" ON)
OPTION(USE_CUDA "Use Cuda libs" OFF)
OPTION(USE_CUDNN "Use Cudnn libs" ON)
OPTION(USE_OPENCV "Use opencv" OFF)
OPTION(USE_LMDB "Use LMDB libs" OFF)
OPTION(USE_PYTHON "Generate py wrappers" ON)
OPTION(USE_JAVA "Generate java wrappers" OFF)
OPTION(USE_OPENCL "Use OpenCL" OFF)
OPTION(ENABLE_DIST "Enable distributed training" OFF)
OPTION(ENABLE_TEST "Enable unit test" OFF)
OPTION(DISABLE_WARNINGS "Disable warnings under windows" ON)
OPTION(USE_MODULES "Compile dependent libs as submodules together with singa" OFF)
#OPTION(USE_SHARED_LIBS "Use shared library" OFF)

# TODO: remove all USE_CBLAS in codes
SET(USE_CBLAS ON)

INCLUDE("cmake/Dependencies.cmake")
#INCLUDE("cmake/Utils.cmake")
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
    include(ExternalProject)
    ExternalProject_Add(cnmem
        GIT_REPOSITORY "https://github.com/nusdbsystem/cnmem.git"
        GIT_TAG "master"
        SOURCE_DIR "cnmem/"
        CONFIGURE_COMMAND "${CMAKE_COMMAND}"
        "-H${CMAKE_BINARY_DIR}/cnmem" "-B."
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}"
        )
    #ADD_SUBDIRECTORY(lib/cnmem)
    #LIST(APPEND SINGA_LINKER_LIBS cnmem)
    SET(global_cuda_objs "")
ENDIF()

# TODO(wangwei) detect the ev lib
IF (ENABLE_DIST)
    LIST(APPEND SINGA_LINKER_LIBS ev)
ENDIF()

IF (USE_MODULES)
    include(ExternalProject)
    #IF(USE_SHARED_LIBS)
    #    ExternalProject_Add(protobuf
    #        GIT_REPOSITORY "https://github.com/google/protobuf.git"
    #        GIT_TAG "2.7.0"
    #        SOURCE_DIR "protobuf/"
    #        CONFIGURE_COMMAND "${CMAKE_COMMAND}"
    #        "-H${CMAKE_BINARY_DIR}/protobuf/cmake" "-B."
    #        "-Dprotobuf_BUILD_TESTS=OFF" "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}"
    #        "-Dprotobuf_BUILD_SHARED_LIBS=ON" "-DCMAKE_C_FLAGS=-fPIC"
    #        "-DCMAKE_CXX_FLAGS=-fPIC"
    #        BUILD_COMMAND "${CMAKE_COMMAND}" --build "."
    #        )
    #ELSE()
    #ENDIF()
    ExternalProject_Add(protobuf
        DOWNLOAD_COMMAND "wget"
        "http://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz"
        "-O" "protobuf-2.6.1.tar.gz"
        UPDATE_COMMAND "tar" "zxvf"
        "${CMAKE_BINARY_DIR}/protobuf-prefix/src/protobuf-2.6.1.tar.gz" "-C" ".."
        SOURCE_DIR "protobuf-2.6.1/"
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND "./configure" "--disable-shared"
        "--prefix=${CMAKE_BINARY_DIR}/" "CXXFLAGS=-fPIC"
        INSTALL_COMMAND "make" "install"
        )
    SET(Protobuf_PROTOC_EXECUTABLE "${CMAKE_BINARY_DIR}/bin/protoc")
    # message(STATUS "${PROTOBUF_PROTOC_EXECUTABLE}")

    ExternalProject_Add(openblas
        GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
        #GIT_TAG "develop"
        #GIT_TAG "f3419e6"
        GIT_TAG "optimized_for_deeplearning"
        SOURCE_DIR "openblas/"
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND "make" "ONLY_CBLAS=1"
        INSTALL_COMMAND "make" "PREFIX=${CMAKE_BINARY_DIR}/" "install"
        )
ENDIF()

INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/include")
ADD_SUBDIRECTORY(src)

IF (ENABLE_TEST)
  ADD_SUBDIRECTORY(test)
  ADD_SUBDIRECTORY(examples)
ENDIF()

IF (USE_PYTHON)
    ADD_SUBDIRECTORY(python)
ENDIF()

IF (USE_JAVA)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing ")
    ADD_SUBDIRECTORY(java)
ENDIF()

INSTALL(DIRECTORY ${CMAKE_BINARY_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
INSTALL(DIRECTORY include/singa DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
INSTALL(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION ${CMAKE_INSTALL_PREFIX} PATTERN "*libgtest.a" EXCLUDE )

INSTALL(CODE "execute_process(COMMAND python setup.py install --prefix=${CMAKE_INSTALL_PREFIX} WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/python)")

# CPack
IF (USE_MODULES)
	SET(CORE_DEPENDENCIES "libgoogle-glog-dev, libstdc++6, libc6")
ELSE()
	SET(CORE_DEPENDENCIES "libgoogle-glog-dev, libprotobuf-dev, libopenblas-dev, libstdc++6, libc6")
ENDIF()

SET(PYTHON_DEPENDENCIES "${CORE_DEPENDENCIES}, python-dev, libpython2.7, python-pip, python-numpy, python-pillow")

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Apache Incubator <dev@singa.incubator.apache.org>")
SET(CPACK_DEBIAN_PACKAGE_VERSION ${PACKAGE_VERSION})
SET(CPACK_DEBIAN_PACKAGE_SECTION "devel")
SET(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
SET(CPACK_DEBIAN_PACKAGE_HOMEPAGE "http://singa.apache.org")
SET(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Singa is an Apache Incubating project for developing an open source deep learning library.
 The Singa project was initiated by the DB System Group at National University of Singapore in 2014. It focused on distributed deep learning by partitioning the model and data onto nodes in a cluster and parallelize the training. The prototype was accepted by Apache Incubator in March 2015.")
IF (USE_PYTHON)
	SET(CPACK_DEBIAN_PACKAGE_DEPENDS ${PYTHON_DEPENDENCIES})
        #SET(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_SOURCE_DIR}/tool/debian/postinst;${PROJECT_SOURCE_DIR}/tool/debian/postrm" )
	IF (USE_CUDA)
		SET(CPACK_DEBIAN_PACKAGE_NAME "python-singa-cuda")
		SET(CPACK_PACKAGE_FILE_NAME "python-singa-cuda-${PACKAGE_VERSION}")
		SET(CPACK_DEBIAN_PACKAGE_PREDEPENDS "ca-certificates, cuda-cudart-8-0, cuda-curand-8-0")
	ELSE()
		SET(CPACK_DEBIAN_PACKAGE_NAME "python-singa")
		SET(CPACK_PACKAGE_FILE_NAME "python-singa-${PACKAGE_VERSION}")
		SET(CPACK_DEBIAN_PACKAGE_PREDEPENDS "ca-certificates")
	ENDIF()
ELSE()
	SET(CPACK_DEBIAN_PACKAGE_NAME "singa")
	SET(CPACK_PACKAGE_FILE_NAME "singa-${PACKAGE_VERSION}")
        SET(CPACK_DEBIAN_PACKAGE_DEPENDS ${CORE_DEPENDENCIES})
ENDIF()
INCLUDE(CPack)
