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


FIND_PATH(GLOG_INCLUDE_DIR NAMES glog/logging.h PATHS "$ENV{GLOG_DIR}/include")
FIND_LIBRARY(GLOG_LIBRARIES NAMES glog)

INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Glog DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARIES)

IF(GLOG_FOUND)
    #    MESSAGE(STATUS "Found glog at ${GLOG_INCLUDE_DIR}")
    MARK_AS_ADVANCED(GLOG_INCLUDE_DIR GLOG_LIBRARIES)
ENDIF()
