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
IF (USE_PYTHON)
    protobuf_generate_python(proto_pys ${proto_files})
ENDIF()
INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/include")

LIST(APPEND singa_sources ${proto_hdrs} ${proto_srcs} ${proto_pys})
SET(PREVIOUS_LINKER_LIBS ${SINGA_LINKER_LIBS})

AUX_SOURCE_DIRECTORY(utils utils_source)
LIST(APPEND singa_sources ${utils_source})

AUX_SOURCE_DIRECTORY(core/device core_source)
AUX_SOURCE_DIRECTORY(core/memory core_source)
AUX_SOURCE_DIRECTORY(core/scheduler core_source)
AUX_SOURCE_DIRECTORY(core/tensor core_source)
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
LIST(APPEND singa_sources ${core_source} ${cuda_objs})

AUX_SOURCE_DIRECTORY(model model_source)
AUX_SOURCE_DIRECTORY(model/layer model_source)
AUX_SOURCE_DIRECTORY(model/optimizer model_source)
AUX_SOURCE_DIRECTORY(model/loss model_source)
AUX_SOURCE_DIRECTORY(model/metric model_source)
AUX_SOURCE_DIRECTORY(model/updater model_source)
LIST(APPEND singa_sources ${model_source})

AUX_SOURCE_DIRECTORY(io io_source)
AUX_SOURCE_DIRECTORY(io/network io_source)
LIST(APPEND singa_sources ${io_source})

IF(USE_PYTHON)

    FILE(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/python/swig/config.i")
    CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/python/swig/config.i.in" "${CMAKE_CURRENT_SOURCE_DIR}/python/swig/config.i")

    FILE(GLOB python_files python/swig/singa.i)
    # delete old .cxx file
    FILE(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/python/swig/singa_wrap.cxx")

    # generate cxx and wrap.py
    swig_generate_cxx(python_srcs ${python_files})

    #FILE(COPY python/ DESTINATION ${CMAKE_BINARY_DIR}/python/singa FILES_MATCHING PATTERN "swig" EXCLUDE PATTERN "*.py")
    #Create symlinks for all python source files  Do not omit !!!RELATIVE!!!
    file(GLOB_RECURSE python_source_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)

    create_symlinks(${python_source_files})

    SET(python_cxxs "${core_source};${io_source};${model_source};${utils_source}")
    ADD_LIBRARY(_singa_wrap SHARED ${python_srcs} ${python_cxxs} ${cuda_objs})
    SET(WRAPPER_LINKER_LIBS "${PREVIOUS_LINKER_LIBS}")
    TARGET_LINK_LIBRARIES(_singa_wrap ${WRAPPER_LINKER_LIBS})
    TARGET_INCLUDE_DIRECTORIES(_singa_wrap PRIVATE ${PYTHON_INCLUDE_DIRS})
    #message(STATUS "PREVIOUS_LINKER_LIBS ${PREVIOUS_LINKER_LIBS}")

    SET_TARGET_PROPERTIES(_singa_wrap
        PROPERTIES PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/singa
        )

    #SETUP
    SET(SETUP_PY_IN "python/setup.py.in")
    SET(SETUP_PY    "${CMAKE_BINARY_DIR}/python/setup.py")
    CONFIGURE_FILE(${SETUP_PY_IN} ${SETUP_PY})

    #create python/singa/proto/__init__.py
    FILE(WRITE ${CMAKE_BINARY_DIR}/python/singa/proto/__init__.py "")

ENDIF(USE_PYTHON)

ADD_LIBRARY(singa SHARED ${singa_sources})
ADD_CUSTOM_TARGET(
  copy_protobuf
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/include/singa/proto"
  COMMAND ${CMAKE_COMMAND} -E copy ${proto_hdrs} "${CMAKE_BINARY_DIR}/include/singa/proto/"
  DEPENDS ${proto_hdrs}
  COMMENT "Copying Protobuf files"
  )
ADD_DEPENDENCIES(singa copy_protobuf)
TARGET_LINK_LIBRARIES(singa ${SINGA_LINKER_LIBS})
#MESSAGE(STATUS "HEADERS: ${proto_hdrs}")

