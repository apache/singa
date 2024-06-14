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


FIND_PACKAGE(CUDA 5.5 QUIET)

IF(NOT CUDA_FOUND)
    return()
ENDIF()

SET(HAVE_CUDA TRUE)
MESSAGE(STATUS "Found cuda_v${CUDA_VERSION}")
#ADD_DEFINITIONS(-DUSE_CUDA)
#message(STATUS "linking: ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES}")

IF(USE_CUDNN)
#include(cmake/Modules/Cudnn.cmake)
    FIND_PACKAGE(CUDNN REQUIRED)
    INCLUDE_DIRECTORIES( ${CUDNN_INCLUDE_DIR})
    LIST(APPEND SINGA_LINKER_LIBS ${CUDNN_LIBRARIES})
ENDIF()

INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS})
LIST(APPEND SINGA_LINKER_LIBS ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
#MESSAGE(STATUS "libs " ${SINGA_LINKER_LIBS})
