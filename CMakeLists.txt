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

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Thirdparty)

#include(GetGitRevisionDescription)
#git_describe(VERSION --tags --dirty=-d)
#string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" VERSION_MAJOR "${VERSION}")
#string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${VERSION}")
#string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH "${VERSION}")


SET(PACKAGE_VERSION 3.3.0) # ${VERSION})
SET(VERSION 3.3.0)
SET(SINGA_MAJOR_VERSION 3)
SET(SINGA_MINOR_VERSION 3)
SET(SINGA_PATCH_VERSION 0)
#SET(SINGA_MAJOR_VERSION ${VERSION_MAJOR})  # 0 -
#SET(SINGA_MINOR_VERSION ${VERSION_MINOR})  # 0 - 9
#SET(SINGA_PATCH_VERSION ${VERSION_PATCH})  # 0 - 99

MATH(EXPR SINGA_VERSION "${SINGA_MAJOR_VERSION} * 1000 + ${SINGA_MINOR_VERSION} * 100 + ${SINGA_PATCH_VERSION}")

message(STATUS "SINGA git Version ${VERSION}")
message(STATUS "SINGA Version ${SINGA_VERSION}")

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


OPTION(USE_CUDA "Use Cuda libs" OFF)
OPTION(ENABLE_TEST "Enable unit test" OFF)
option(CODE_COVERAGE "Enable coverage reporting" OFF)
OPTION(USE_PYTHON "Generate py wrappers" ON)
OPTION(USE_PYTHON3 "Python 3x" OFF)

OPTION(USE_CUDNN "Use Cudnn libs" ON)
OPTION(USE_OPENCV "Use opencv" OFF)
OPTION(USE_LMDB "Use LMDB libs" OFF)
OPTION(USE_JAVA "Generate java wrappers" OFF)
OPTION(USE_OPENCL "Use OpenCL" OFF)
OPTION(ENABLE_DIST "Enable distributed training" OFF)
OPTION(DISABLE_WARNINGS "Disable warnings under windows" ON)
OPTION(USE_MODULES "Compile dependent libs as submodules together with singa" OFF)
OPTION(USE_DNNL "Use dnnl libs" OFF)
OPTION(USE_DIST "Use nccl distributed module" OFF)

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
    # add support cuda fp16
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --gpu-architecture=compute_75")
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
        "https://github.com/google/protobuf/archive/v3.3.0.tar.gz"
        "-O" "protobuf-3.3.0.tar.gz"
        UPDATE_COMMAND "tar" "zxvf"
        "${CMAKE_BINARY_DIR}/protobuf-prefix/src/protobuf-3.3.0.tar.gz" "-C" ".."
        SOURCE_DIR "protobuf-3.3.0/"
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND "./autogen.sh" COMMAND "./configure" "--disable-shared"
        "--prefix=${CMAKE_BINARY_DIR}/" "CXXFLAGS=-fPIC"
        INSTALL_COMMAND "make" "install"
        )
    SET(Protobuf_PROTOC_EXECUTABLE "${CMAKE_BINARY_DIR}/bin/protoc")
    # message(STATUS "${PROTOBUF_PROTOC_EXECUTABLE}")

    ExternalProject_Add(openblas
        GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
        #GIT_TAG "develop"
        #GIT_TAG "f3419e6"
        GIT_TAG "v0.2.20"
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

INSTALL(DIRECTORY ${CMAKE_BINARY_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX}/)
INSTALL(DIRECTORY include/singa DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
INSTALL(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION ${CMAKE_INSTALL_PREFIX}/ PATTERN "*libgtest.a" EXCLUDE )

INSTALL(CODE "execute_process(COMMAND python setup.py install --prefix=${CMAKE_INSTALL_PREFIX} WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/python)")

# CPack
IF(PACKAGE)
	IF(USE_PYTHON)
		INSTALL(FILES ${CMAKE_BINARY_DIR}/python/setup.py
			DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/singa/python)
		INSTALL(FILES ${CMAKE_BINARY_DIR}/python/singa/singa_wrap.py
			DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/singa/python/singa)
		INSTALL(FILES ${CMAKE_BINARY_DIR}/python/singa/_singa_wrap.so
			DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/singa/python/singa)
		INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/python
			DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/singa)
		INSTALL(DIRECTORY ${CMAKE_BINARY_DIR}/python/singa/proto
			DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/singa/python/singa)
	ENDIF()

	IF (USE_MODULES)
		SET(CORE_DEPENDENCIES "libgoogle-glog-dev, libstdc++6, libc6")
	ELSE()
		SET(CORE_DEPENDENCIES "libgoogle-glog-dev, libprotobuf-dev, libopenblas-dev, libstdc++6, libc6")
	ENDIF()

	IF(USE_PYTHON3)
		SET(PYTHON_DEPENDENCIES "${CORE_DEPENDENCIES}, python3, python3-dev, python3-pip, python3-numpy, python3-pillow, python3-matplotlib")
	ELSE()
		SET(PYTHON_DEPENDENCIES "${CORE_DEPENDENCIES}, python-dev, libpython2.7, python-pip, python-numpy, python-pillow")
	ENDIF()

	SET(CPACK_GENERATOR "DEB")
	SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Apache Incubator <dev@singa.incubator.apache.org>")
	SET(CPACK_DEBIAN_PACKAGE_VERSION ${PACKAGE_VERSION})
	SET(CPACK_DEBIAN_PACKAGE_SECTION "devel")
	SET(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
	SET(CPACK_DEBIAN_PACKAGE_HOMEPAGE "http://singa.apache.org")
	SET(CPACK_DEBIAN_PACKAGE_DESCRIPTION "SINGA is an Apache Incubating project for developing an open source deep learning library.
	 The SINGA project was initiated by the DB System Group at National University of Singapore in 2014. It focused on distributed deep learning by partitioning the model and data onto nodes in a cluster and parallelize the training. The prototype was accepted by Apache Incubator in March 2015.")
	IF (USE_PYTHON)
		SET(CPACK_DEBIAN_PACKAGE_DEPENDS ${PYTHON_DEPENDENCIES})
		SET(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_SOURCE_DIR}/tool/debian/postinst" )
		SET(CPACK_DEBIAN_PACKAGE_PREDEPENDS "ca-certificates")
		IF(USE_PYTHON3)
			SET(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_SOURCE_DIR}/tool/debian/postinst" )
			IF (USE_CUDA)
				SET(CPACK_DEBIAN_PACKAGE_NAME "python3-singa-cuda")
				SET(CPACK_PACKAGE_FILE_NAME "python3-singa-cuda-${PACKAGE_VERSION}")
			ELSE()
				SET(CPACK_DEBIAN_PACKAGE_NAME "python3-singa")
				SET(CPACK_PACKAGE_FILE_NAME "python3-singa-${PACKAGE_VERSION}")
			ENDIF()
		ELSE()
			SET(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_SOURCE_DIR}/tool/debian-python2/postinst" )
			IF (USE_CUDA)
				SET(CPACK_DEBIAN_PACKAGE_NAME "python-singa-cuda")
				SET(CPACK_PACKAGE_FILE_NAME "python-singa-cuda-${PACKAGE_VERSION}")
			ELSE()
				SET(CPACK_DEBIAN_PACKAGE_NAME "python-singa")
				SET(CPACK_PACKAGE_FILE_NAME "python-singa-${PACKAGE_VERSION}")
			ENDIF()
		ENDIF()
	ELSE()
		SET(CPACK_DEBIAN_PACKAGE_NAME "singa")
		SET(CPACK_PACKAGE_FILE_NAME "singa-${PACKAGE_VERSION}")
		SET(CPACK_DEBIAN_PACKAGE_DEPENDS ${CORE_DEPENDENCIES})
	ENDIF()
	INCLUDE(CPack)
ENDIF()
