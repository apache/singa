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

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)

IF(ENABLE_DIST)
  ADD_EXECUTABLE(test_ep "singa/test_ep.cc")
  ADD_DEPENDENCIES(test_ep singa)
  TARGET_LINK_LIBRARIES(test_ep singa protobuf ${SINGA_LINKER_LIBS})
ENDIF()

ADD_LIBRARY(gtest STATIC EXCLUDE_FROM_ALL "gtest/gtest.h" "gtest/gtest-all.cc")

AUX_SOURCE_DIRECTORY(singa singa_test_source)
LIST(REMOVE_ITEM singa_test_source "singa/test_ep.cc")

ADD_EXECUTABLE(test_singa "gtest/gtest_main.cc" ${singa_test_source})
ADD_DEPENDENCIES(test_singa singa)
#MESSAGE(STATUS "link libs" ${singa_linker_libs})
TARGET_LINK_LIBRARIES(test_singa gtest singa )
IF(UNIX AND (NOT APPLE))
    LIST(APPEND LINK_FLAGS "-pthread")
ENDIF()
SET_TARGET_PROPERTIES(test_singa PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
