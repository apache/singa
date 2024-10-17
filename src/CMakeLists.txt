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

# generate protobuf sources
FILE(GLOB proto_files proto/*.proto)
protobuf_generate_cpp(proto_srcs proto_hdrs ${proto_files})
#MESSAGE(STATUS "proto_srcs: ${proto_srcs}")

LIST(APPEND singa_sources ${proto_hdrs} ${proto_srcs})

AUX_SOURCE_DIRECTORY(utils utils_source)
LIST(APPEND singa_sources ${utils_source})

AUX_SOURCE_DIRECTORY(io io_source)
AUX_SOURCE_DIRECTORY(io/network io_source)
LIST(APPEND singa_sources ${io_source})

AUX_SOURCE_DIRECTORY(core/common core_source)
AUX_SOURCE_DIRECTORY(core/device core_source)
AUX_SOURCE_DIRECTORY(core/memory core_source)
AUX_SOURCE_DIRECTORY(core/scheduler core_source)
AUX_SOURCE_DIRECTORY(core/tensor core_source)
LIST(APPEND singa_sources ${core_source})

IF (USE_DIST)
    AUX_SOURCE_DIRECTORY(dist dist_source)
    LIST(APPEND singa_sources ${dist_source})
ENDIF (USE_DIST)

IF (USE_CUDA)
    FILE(GLOB_RECURSE cuda_source core "*.cu")
    SET(FLAGS_BACKUP ${CMAKE_CXX_FLAGS})
    SET(CMAKE_CXX_FLAGS "")
    IF (CMAKE_BUILD_TYPE MATCHES DEBUG)
        CUDA_COMPILE(cuda_objs SHARED ${cuda_source}
            OPTIONS "-Xcompiler -fPIC -G -g")
    ELSE (CMAKE_BUILD_TYPE MATCHES  DEBUG)
        CUDA_COMPILE(cuda_objs SHARED ${cuda_source} OPTIONS "-Xcompiler -fPIC")
    ENDIF (CMAKE_BUILD_TYPE MATCHES DEBUG)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/core/tensor")
    SET(CMAKE_CXX_FLAGS ${FLAGS_BACKUP})
ENDIF (USE_CUDA)

SET(global_cuda_objs ${cuda_objs} PARENT_SCOPE)

AUX_SOURCE_DIRECTORY(model model_source)
AUX_SOURCE_DIRECTORY(model/layer model_source)
AUX_SOURCE_DIRECTORY(model/optimizer model_source)
AUX_SOURCE_DIRECTORY(model/loss model_source)
AUX_SOURCE_DIRECTORY(model/metric model_source)
AUX_SOURCE_DIRECTORY(model/updater model_source)
AUX_SOURCE_DIRECTORY(model/operation model_source)
LIST(APPEND singa_sources ${model_source})


ADD_CUSTOM_TARGET(
  copy_protobuf ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/include/singa/proto"
  DEPENDS ${proto_hdrs}
  COMMENT "Copying Protobuf headers"
  )
FOREACH(fil ${proto_hdrs})
  GET_FILENAME_COMPONENT(filename ${fil} NAME)
  ADD_CUSTOM_COMMAND(
    TARGET copy_protobuf POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${fil}
    "${CMAKE_BINARY_DIR}/include/singa/proto/${filename}"
    DEPENDS ${proto_files}
 )
ENDFOREACH()

ADD_LIBRARY(singa_objects OBJECT ${singa_sources})
IF(WIN32)
  ADD_LIBRARY(singa STATIC $<TARGET_OBJECTS:singa_objects> ${cuda_objs})
  TARGET_LINK_LIBRARIES(singa ${SINGA_LINKER_LIBS})
  IF (MSVC)
	ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
	IF(DISABLE_WARNINGS)
	  ADD_DEFINITIONS(-DDISABLE_WARNINGS)
	  SET_TARGET_PROPERTIES(singa_objects
	    PROPERTIES COMPILE_FLAGS "/wd4244 /wd4267 /wd4018 /wd4005 /wd4804 /wd4800")
	ENDIF()
  ENDIF()
ENDIF()
ADD_DEPENDENCIES(singa_objects copy_protobuf)

IF(UNIX OR APPLE)
  ADD_LIBRARY(singa SHARED $<TARGET_OBJECTS:singa_objects> ${cuda_objs})
  TARGET_LINK_LIBRARIES(singa ${SINGA_LINKER_LIBS})
  SET_TARGET_PROPERTIES(singa PROPERTIES LINK_FLAGS "")
ENDIF()

IF(CODE_COVERAGE)
    MESSAGE("-- Enabling Code Coverage")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g --coverage")
ENDIF(CODE_COVERAGE)

#pass configure infor to swig
FILE(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/api/config.i")
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/api/config.i.in" "${CMAKE_CURRENT_SOURCE_DIR}/api/config.i")
