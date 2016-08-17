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


FIND_PATH(CUDNN_INCLUDE_DIR NAME "cudnn.h" PATHS "$ENV{CMAKE_INCLUDE_PATH}")
FIND_LIBRARY(CUDNN_LIBRARIES NAME "libcudnn.so" PATHS "$ENV{CMAKE_LIBRARY_PATH}")

#message("cudnn include path:${CUDNN_INCLUDE_DIR}  lib path: ${CUDNN_LIBRARIES}")
#message("env include path:$ENV{CUDNN_DIR} next: $ENV{CMAKE_INCLUDE_PATH}")
INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARIES)

IF(CUDNN_FOUND)
    FILE(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
    STRING(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
        CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    STRING(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
        CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    STRING(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
        CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    STRING(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
        CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    STRING(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
        CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    STRING(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
        CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

    IF(NOT CUDNN_VERSION_MAJOR)
        SET(CUDNN_VERSION "???")
    ELSE()
      MATH(EXPR CUDNN_VERSION_SWIG "${CUDNN_VERSION_MAJOR} * 1000 + ${CUDNN_VERSION_MINOR} * 100 + ${CUDNN_VERSION_PATCH}")
    ENDIF()
    MESSAGE(STATUS "Found Cudnn_v${CUDNN_VERSION_SWIG} at ${CUDNN_INCLUDE_DIR} ${CUDNN_LIBRARIES}")
    MARK_AS_ADVANCED(CUDNN_INCLUDE_DIR CUDNN_LIBRARIES)

ENDIF()
