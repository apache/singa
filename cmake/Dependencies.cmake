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

SET(SINGA_LINKER_LIBS "")

IF(USE_MODULES)
    #IF(USE_SHARED_LIBS)
    #    include(FindProtobuf)
    #    SET(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib")
    #    link_directories(${CMAKE_BINARY_DIR}/lib)
    #    SET(PROTOBUF_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    #    SET(PROTOBUF_LIBRARY "${CMAKE_BINARY_DIR}/lib/libprotobuf.so")
    #    SET(PROTOBUF_PROTOC_LIBRARY "${CMAKE_BINARY_DIR}/lib/libprotoc.so")
    #    SET(PROTOBUF_PROTOC_EXECUTABLE "${CMAKE_BINARY_DIR}/bin/protoc")
    #    INCLUDE_DIRECTORIES(SYSTEM ${PROTOBUF_INCLUDE_DIR})
    #    LIST(APPEND SINGA_LINKER_LIBS ${PROTOBUF_LIBRARY})
    #    #IF(USE_CBLAS)
    #        SET(CBLAS_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    #        SET(CBLAS_LIBRARIES "${CMAKE_BINARY_DIR}/lib/libopenblas.so")
    #        INCLUDE_DIRECTORIES(SYSTEM ${CBLAS_INCLUDE_DIR})
    #        LIST(APPEND SINGA_LINKER_LIBS ${CBLAS_LIBRARIES})
    #ENDIF()
    #ELSE()
    include(FindProtobuf)
    SET(PROTOBUF_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    SET(PROTOBUF_LIBRARY "${CMAKE_BINARY_DIR}/lib/libprotobuf.a")
    SET(PROTOBUF_PROTOC_LIBRARY "${CMAKE_BINARY_DIR}/lib/libprotobuf.a")
    SET(PROTOBUF_PROTOC_EXECUTABLE "${CMAKE_BINARY_DIR}/bin/protoc")
    INCLUDE_DIRECTORIES( ${PROTOBUF_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${PROTOBUF_LIBRARY})
    #IF(USE_CBLAS)
    SET(CBLAS_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    SET(CBLAS_LIBRARIES "${CMAKE_BINARY_DIR}/lib/libopenblas.a")
    INCLUDE_DIRECTORIES( ${CBLAS_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${CBLAS_LIBRARIES})
    #ENDIF()
    #ENDIF()
ELSE()
    FIND_PACKAGE( Protobuf 3.0 REQUIRED )
    #MESSAGE(STATUS "proto libs " ${PROTOBUF_LIBRARY})
    LIST(APPEND SINGA_LINKER_LIBS ${PROTOBUF_LIBRARY})
    #IF(USE_CBLAS)
    FIND_PACKAGE(CBLAS REQUIRED)
    INCLUDE_DIRECTORIES( ${CBLAS_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${CBLAS_LIBRARIES})
    #MESSAGE(STATUS "Found cblas at ${CBLAS_LIBRARIES}")
    #ENDIF()
ENDIF()

#INCLUDE("cmake/ProtoBuf.cmake")
#INCLUDE("cmake/Protobuf.cmake")

FIND_PACKAGE(Glog)
IF(GLOG_FOUND)
    MESSAGE(STATUS "FOUND GLOG at ${GLOG_INCLUDE_DIR}")
    #ADD_DEFINITIONS("-DUSE_GLOG")
    SET(USE_GLOG TRUE)
    LIST(APPEND SINGA_LINKER_LIBS ${GLOG_LIBRARIES})
    INCLUDE_DIRECTORIES(${GLOG_INCLUDE_DIR})
ENDIF()

IF(USE_LMDB)
    FIND_PACKAGE(LMDB REQUIRED)
    INCLUDE_DIRECTORIES( ${LMDB_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${LMDB_LIBRARIES})
    #MESSAGE(STATUS "FOUND lmdb at ${LMDB_INCLUDE_DIR}")
ENDIF()

IF(USE_CUDA)
    INCLUDE("cmake/Cuda.cmake")
    SET(CNMEM_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
    SET(CNMEM_LIBRARY "${CMAKE_BINARY_DIR}/lib/libcnmem.a")
    LIST(APPEND SINGA_LINKER_LIBS ${CNMEM_LIBRARY})
ELSE()
    SET(USE_CUDNN FALSE)
ENDIF()

IF(USE_OPENCL)
    FIND_PACKAGE(OpenCL REQUIRED)
    IF(NOT OPENCL_FOUND)
        MESSAGE(SEND_ERROR "OpenCL was requested, but not found.")
    ELSE()
        INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIR})
        LIST(APPEND SINGA_LINKER_LIBS ${OPENCL_LIBRARIES})
        FIND_PACKAGE(ViennaCL REQUIRED)
        IF(NOT ViennaCL_FOUND)
            MESSAGE(SEND_ERROR "ViennaCL is required if OpenCL is enabled.")
        ELSE()
            #MESSAGE(STATUS "Found ViennaCL headers at ${ViennaCL_INCLUDE_DIR}")
            INCLUDE_DIRECTORIES( ${ViennaCL_INCLUDE_DIR})
            LIST(APPEND SINGA_LINKER_LIBS ${ViennaCL_LIBRARIES})
        ENDIF()
    ENDIF()
ENDIF()

#FIND_PACKAGE(Glog REQUIRED)
#INCLUDE_DIRECTORIES(SYSTEM ${GLOG_INCLUDE_DIRS})
#LIST(APPEND SINGA_LINKER_LIBS ${GLOG_LIBRARIES})
#MESSAGE(STATUS "Found glog at ${GLOG_INCLUDE_DIRS}")

IF(USE_OPENCV)
    FIND_PACKAGE(OpenCV REQUIRED)
    MESSAGE(STATUS "Found OpenCV_${OpenCV_VERSION} at ${OpenCV_INCLUDE_DIRS}")
    INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})
    LIST(APPEND SINGA_LINKER_LIBS ${OpenCV_LIBRARIES})
ENDIF()

#LIST(APPEND SINGA_LINKER_LIBS "/home/wangwei/local/lib/libopenblas.so")
#MESSAGE(STATUS "link lib : " ${SINGA_LINKER_LIBS})

IF(USE_PYTHON)
    IF(USE_PYTHON3)
        set(Python_ADDITIONAL_VERSIONS 3.6 3.5 3.4)        
        FIND_PACKAGE(PythonInterp 3 REQUIRED)
        FIND_PACKAGE(PythonLibs 3 REQUIRED)
	    FIND_PACKAGE(SWIG 3.0.10 REQUIRED)
    ELSE()        
        FIND_PACKAGE(PythonInterp 2.7 REQUIRED)
        FIND_PACKAGE(PythonLibs 2.7 REQUIRED)
	    FIND_PACKAGE(SWIG 3.0.8 REQUIRED)
    ENDIF()
ENDIF()

IF(USE_JAVA)
    FIND_PACKAGE(Java REQUIRED)
    FIND_PACKAGE(JNI REQUIRED)
    FIND_PACKAGE(SWIG 3.0 REQUIRED)
ENDIF()

IF(USE_DNNL)
    FIND_PATH(DNNL_INCLUDE_DIR NAME "dnnl.hpp" PATHS "$ENV{DNNL_ROOT}/include")
    FIND_LIBRARY(DNNL_LIBRARIES NAME "dnnl" PATHS "$ENV{DNNL_ROOT}/lib")
    MESSAGE(STATUS "Found DNNL at ${DNNL_INCLUDE_DIR}")
    INCLUDE_DIRECTORIES(${DNNL_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${DNNL_LIBRARIES})
ENDIF()

IF(USE_DIST)
    FIND_PATH(MPI_INCLUDE_DIR NAME "mpi.h" PATHS "$ENV{HOME}/mpich-3.3.2/build/include/")
    FIND_LIBRARY(MPI_LIBRARIES NAME "mpi" PATHS "$ENV{HOME}/mpich-3.3.2/build/lib")
    FIND_LIBRARY(MPICXX_LIBRARIES NAME "mpicxx" PATHS "$ENV{HOME}/mpich-3.3.2/build/lib")
    MESSAGE(STATUS "Found MPI at ${MPI_INCLUDE_DIR}")
    INCLUDE_DIRECTORIES(${MPI_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${MPI_LIBRARIES})
    LIST(APPEND SINGA_LINKER_LIBS ${MPICXX_LIBRARIES})
    MESSAGE(STATUS "Found MPI lib at ${MPI_LIBRARIES}")
    MESSAGE(STATUS "Found all lib at ${SINGA_LINKER_LIBS}")
    FIND_PATH(NCCL_INCLUDE_DIR NAME "nccl.h" PATHS "/usr/include/")
    FIND_LIBRARY(NCCL_LIBRARIES NAME "nccl" PATHS "/usr/lib/x86_64-linux-gnu/")
    MESSAGE(STATUS "Found NCCL at ${NCCL_INCLUDE_DIR}")
    INCLUDE_DIRECTORIES(${NCCL_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${NCCL_LIBRARIES})
    MESSAGE(STATUS "Found NCCL lib at ${NCCL_LIBRARIES}")
ENDIF()
