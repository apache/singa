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


ADD_EXECUTABLE(cnn cnn.cc)
ADD_DEPENDENCIES(cnn singa)
TARGET_LINK_LIBRARIES(cnn singa)

IF(USE_CUDNN)

ADD_EXECUTABLE(cnn-parallel cnn-parallel.cc)
ADD_DEPENDENCIES(cnn-parallel singa)
TARGET_LINK_LIBRARIES(cnn-parallel singa)
#SET_TARGET_PROPERTIES(cnn-parallel PROPERTIES LINK_FLAGS "${LINK_FLAGS} -pthread")

ADD_EXECUTABLE(vgg-parallel vgg-parallel.cc)
ADD_DEPENDENCIES(vgg-parallel singa)
TARGET_LINK_LIBRARIES(vgg-parallel singa)
#SET_TARGET_PROPERTIES(vgg-parallel PROPERTIES LINK_FLAGS "${LINK_FLAGS} -pthread")

ENDIF(USE_CUDNN)
