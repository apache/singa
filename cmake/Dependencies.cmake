#
# Copyright 2015 The Apache Software Foundation
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

#INCLUDE("cmake/ProtoBuf.cmake")

FIND_PACKAGE( Protobuf REQUIRED )
INCLUDE_DIRECTORIES(SYSTEM ${PROTOBUF_INCLUDE_DIR})
MESSAGE(STATUS "proto libs " ${PROTOBUF_LIBRARIES})
LIST(APPEND SINGA_LINKER_LIBS ${PROTOBUF_LIBRARIES})
INCLUDE("cmake/Protobuf.cmake")

#FIND_PACKAGE(Glog)
#IF(GLOG_FOUND)
#    MESSAGE(STATUS "GLOG FOUND at ${GLOG_INCLUDE_DIR}")
#    ADD_DEFINITIONS("-DUSE_GLOG")
#    LIST(APPEND SINGA_LINKER_LIBS ${GLOG_LIBRARIES})
#ENDIF()

IF(USE_LMDB)
    FIND_PACKAGE(LMDB REQUIRED)
    INCLUDE_DIRECTORIES(SYSTEM ${LMDB_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${LMDB_LIBRARIES})
    MESSAGE(STATUS "FOUND lmdb at ${LMDB_INCLUDE_DIR}")
ENDIF()

IF(USE_CUDA)
    INCLUDE("cmake/Cuda.cmake")
ELSE()
    SET(USE_CUDNN FALSE)
ENDIF()

IF(USE_CBLAS)
    FIND_PACKAGE(CBLAS REQUIRED)
    INCLUDE_DIRECTORIES(SYSTEM ${CBLAS_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${CBLAS_LIBRARIES})
    MESSAGE(STATUS "FOUND cblas at ${CBLAS_LIBRARIES}")
ENDIF()

IF(USE_OPENCL)
    FIND_PACKAGE(OpenCL REQUIRED)
    IF(NOT OPENCL_FOUND)
        MESSAGE(SEND_ERROR "OpenCL was requested, but not found.")
    ELSE()
        INCLUDE_DIRECTORIES(SYSTEM ${OpenCL_INCPATH})
        LIST(APPEND SINGA_LINKER_LIBS ${OPENCL_LIBRARIES})
        MESSAGE(STATUS "Found OpenCL at ${OPENCL_INCLUDE_DIRS}")
        IF(NOT OPENCL_HAS_CPP_BINDINGS)
            MESSAGE(SEND_ERROR "OpenCL C++ bindings cl2.hpp was not found.")
        ELSE()
            MESSAGE(STATUS "Found OpenCL C++ bindings.")
        ENDIF()
    ENDIF()
ENDIF()

FIND_PACKAGE(Glog REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${GLOG_INCLUDE_DIRS})
LIST(APPEND SINGA_LINKER_LIBS ${GLOG_LIBRARIES})
#MESSAGE(STATUS "Found glog at ${GLOG_INCLUDE_DIRS}")

IF(USE_OPENCV)
    FIND_PACKAGE(OpenCV REQUIRED)
    MESSAGE(STATUS "Found OpenCV_${OpenCV_VERSION} at ${OpenCV_INCLUDE_DIRS}")
    INCLUDE_DIRECTORIES(SYSTEM ${OpenCV_INCLUDE_DIRS})
    LIST(APPEND SINGA_LINKER_LIBS ${OpenCV_LIBRARIES})
ENDIF()

#LIST(APPEND SINGA_LINKER_LIBS "/home/wangwei/local/lib/libopenblas.so")
#MESSAGE(STATUS "link lib : " ${SINGA_LINKER_LIBS})

IF(USE_PYTHON)
    FIND_PACKAGE(PythonLibs 2.7 REQUIRED)
    FIND_PACKAGE(PythonInterp 2.7 REQUIRED)
    FIND_PACKAGE(SWIG 3.0 REQUIRED)
ENDIF()
