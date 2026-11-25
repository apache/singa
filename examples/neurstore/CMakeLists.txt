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

IF(USE_NEURSTORE)
  INCLUDE_DIRECTORIES(BEFORE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      ${NEURSTORE_INCLUDE_DIRS})
  ADD_EXECUTABLE(model_compression model_compression.cpp)
  SET_TARGET_PROPERTIES(model_compression PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON)
  TARGET_LINK_LIBRARIES(model_compression ${NEURSTORE_LIBRARIES})
ENDIF()
