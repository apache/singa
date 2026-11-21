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

SET(NEURSTORE_ROOT "$ENV{NEURSTORE_ROOT}" CACHE PATH
    "Path to NeurStore with its core library and dependencies built")

FIND_PACKAGE(PostgreSQL REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED COMPONENTS CXX)

# Search the dependency layout used by NeurStore's CMakeLists.txt.
FIND_LIBRARY(NEURSTORE_CORE_LIBRARY NAMES neurstore_core
    HINTS "${NEURSTORE_ROOT}/build")
FIND_PATH(NEURSTORE_EIGEN_INCLUDE_DIR NAMES Eigen/Core
    HINTS "${NEURSTORE_ROOT}/external/eigen" PATH_SUFFIXES eigen3)
FIND_PATH(NEURSTORE_HNSWLIB_INCLUDE_DIR NAMES hnswlib/space_quantized.h
    HINTS "${NEURSTORE_ROOT}/external/hnswlib")
FIND_PATH(NEURSTORE_ONNX_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
    HINTS "${NEURSTORE_ROOT}/external/onnx/include")
FIND_PATH(NEURSTORE_ONNX_PROTO_INCLUDE_DIR NAMES onnx.pb.h
    HINTS "${NEURSTORE_ROOT}/external/onnx/onnx_protobuf/build")
FIND_LIBRARY(NEURSTORE_ONNX_LIBRARY NAMES onnxruntime
    HINTS "${NEURSTORE_ROOT}/external/onnx/lib")

# Use the Protobuf built for NeurStore's generated ONNX headers. SINGA may
# use a different Protobuf version, so keep its variables and paths separate.
FIND_PATH(NEURSTORE_PROTOBUF_INCLUDE_DIR NAMES google/protobuf/message.h
    HINTS "${NEURSTORE_ROOT}/external/protobuf/protobuf_install/include"
    NO_DEFAULT_PATH)
FIND_LIBRARY(NEURSTORE_PROTOBUF_LIBRARY NAMES protobuf
    HINTS "${NEURSTORE_ROOT}/external/protobuf/protobuf_install/lib"
    NO_DEFAULT_PATH)
FIND_PATH(NEURSTORE_TOKENIZERS_INCLUDE_DIR NAMES tokenizers_cpp.h
    HINTS "${NEURSTORE_ROOT}/external/tokenizers-cpp/include")
FIND_LIBRARY(NEURSTORE_TOKENIZERS_CPP_LIBRARY NAMES tokenizers_cpp
    HINTS "${NEURSTORE_ROOT}/external/tokenizers-cpp/build")
FIND_LIBRARY(NEURSTORE_TOKENIZERS_C_LIBRARY NAMES tokenizers_c
    HINTS "${NEURSTORE_ROOT}/external/tokenizers-cpp/build")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(NeurStore DEFAULT_MSG
    NEURSTORE_CORE_LIBRARY
    NEURSTORE_EIGEN_INCLUDE_DIR
    NEURSTORE_HNSWLIB_INCLUDE_DIR
    NEURSTORE_ONNX_INCLUDE_DIR
    NEURSTORE_ONNX_PROTO_INCLUDE_DIR
    NEURSTORE_ONNX_LIBRARY
    NEURSTORE_PROTOBUF_INCLUDE_DIR
    NEURSTORE_PROTOBUF_LIBRARY
    NEURSTORE_TOKENIZERS_INCLUDE_DIR
    NEURSTORE_TOKENIZERS_CPP_LIBRARY
    NEURSTORE_TOKENIZERS_C_LIBRARY)

IF(NeurStore_FOUND)
  SET(NEURSTORE_INCLUDE_DIRS
      ${NEURSTORE_EIGEN_INCLUDE_DIR}
      ${NEURSTORE_HNSWLIB_INCLUDE_DIR}
      ${NEURSTORE_ONNX_INCLUDE_DIR}
      ${NEURSTORE_ONNX_PROTO_INCLUDE_DIR}
      ${NEURSTORE_PROTOBUF_INCLUDE_DIR}
      ${NEURSTORE_TOKENIZERS_INCLUDE_DIR}
      ${PostgreSQL_INCLUDE_DIRS})
  SET(NEURSTORE_LIBRARIES
      ${NEURSTORE_CORE_LIBRARY}
      ${PostgreSQL_LIBRARIES}
      ${NEURSTORE_PROTOBUF_LIBRARY}
      ${NEURSTORE_ONNX_LIBRARY}
      ${NEURSTORE_TOKENIZERS_CPP_LIBRARY}
      ${NEURSTORE_TOKENIZERS_C_LIBRARY}
      OpenMP::OpenMP_CXX)
  MARK_AS_ADVANCED(
      NEURSTORE_CORE_LIBRARY
      NEURSTORE_EIGEN_INCLUDE_DIR
      NEURSTORE_HNSWLIB_INCLUDE_DIR
      NEURSTORE_ONNX_INCLUDE_DIR
      NEURSTORE_ONNX_PROTO_INCLUDE_DIR
      NEURSTORE_ONNX_LIBRARY
      NEURSTORE_PROTOBUF_INCLUDE_DIR
      NEURSTORE_PROTOBUF_LIBRARY
      NEURSTORE_TOKENIZERS_INCLUDE_DIR
      NEURSTORE_TOKENIZERS_CPP_LIBRARY
      NEURSTORE_TOKENIZERS_C_LIBRARY)
ENDIF()
