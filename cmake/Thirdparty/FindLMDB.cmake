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


FIND_PATH(LMDB_INCLUDE_DIR NAMES lmdb.h PATHS "$ENV{LMDB_DIR}/include")
FIND_LIBRARY(LMDB_LIBRARIES NAMES lmdb PATHS "$ENV{LMDB_DIR}/include")

INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LMDB DEFAULT_MSG LMDB_INCLUDE_DIR LMDB_LIBRARIES)

IF(LMDB_FOUND)
    MESSAGE(STATUS "Found lmdb at ${LMDB_INCLUDE_DIR}")
    MARK_AS_ADVANCED(LMDB_INCLUDE_DIR LMDB_LIBRARIES)

ENDIF()
