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


FIND_PATH(CBLAS_INCLUDE_DIR NAMES cblas.h PATHS "$ENV{CMAKE_INCLUDE_PATH}")
FIND_LIBRARY(CBLAS_LIBRARIES NAMES openblas PATHS "$ENV{CMAKE_LIBRARY_PATH}")

INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_INCLUDE_DIR CBLAS_LIBRARIES)

IF(CBLAS_FOUND)
    MARK_AS_ADVANCED(CBLAS_INCLUDE_DIR CBLAS_LIBRARIES)
ENDIF()
