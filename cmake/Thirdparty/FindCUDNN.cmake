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
      CUDNN_MAJOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
    STRING(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
      CUDNN_MAJOR_VERSION "${CUDNN_MAJOR_VERSION}")
    STRING(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
        CUDNN_MINOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
    STRING(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
        CUDNN_MINOR_VERSION "${CUDNN_MINOR_VERSION}")
    STRING(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
        CUDNN_PATCH_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
    STRING(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
        CUDNN_PATCH_VERSION "${CUDNN_PATCH_VERSION}")

    IF(NOT CUDNN_MAJOR_VERSION)
        SET(CUDNN_VERSION "???")
    ELSE()
      MATH(EXPR CUDNN_VERSION "${CUDNN_MAJOR_VERSION} * 1000 + ${CUDNN_MINOR_VERSION} * 100 + ${CUDNN_PATCH_VERSION}")
    ENDIF()
    MESSAGE(STATUS "Found Cudnn_${CUDNN_VERSION} at ${CUDNN_INCLUDE_DIR} ${CUDNN_LIBRARIES}")
    MARK_AS_ADVANCED(CUDNN_INCLUDE_DIR CUDNN_LIBRARIES)

ENDIF()
