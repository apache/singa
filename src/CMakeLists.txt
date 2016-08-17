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

#message(STATUS "include: ${CMAKE_BINARY_DIR} ")
#message(STATUS "srcs: ${proto_srcs}")
#message(STATUS "hdrs: ${proto_hdrs}")
#message(STATUS "pys: ${proto_pys}")
ADD_LIBRARY(singa_proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_pys})
FOREACH(fil ${proto_hdrs})
    ADD_CUSTOM_COMMAND(
        TARGET singa_proto PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/include/singa/proto"
        COMMAND ${CMAKE_COMMAND} -E copy ${fil} "${CMAKE_BINARY_DIR}/include/singa/proto"
        #COMMAND ${CMAKE_COMMAND} -E echo "copy done"
        )
ENDFOREACH()
LIST(APPEND SINGA_LINKER_LIBS singa_proto)

SET(PREVIOUS_LINKER_LIBS ${SINGA_LINKER_LIBS})

#FILE(GLOB_RECURSE utils_source ${CMAKE_CURRENT_SOURCE_DIR}/utils/ "*.cc")
AUX_SOURCE_DIRECTORY(utils utils_source)
#message(STATUS "UTILS ${utils_source}")
ADD_LIBRARY(singa_utils SHARED ${utils_source})
TARGET_LINK_LIBRARIES(singa_utils ${SINGA_LINKER_LIBS})
LIST(APPEND SINGA_LINKER_LIBS singa_utils)

#FILE(GLOB_RECURSE core_source ${CMAKE_CURRENT_SOURCE_DIR}/core/ "*.cc")
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
#message(STATUS "FLAGS ${CMAKE_CXX_FLAGS}")
#message(STATUS "CORE ${cuda_source}")
#message(STATUS "OBJ ${cuda_objs}")
ADD_LIBRARY(singa_core SHARED ${core_source} ${cuda_objs})
TARGET_LINK_LIBRARIES(singa_core ${SINGA_LINKER_LIBS})
LIST(APPEND SINGA_LINKER_LIBS singa_core)
#MESSAGE(STATUS "link libs " ${SINGA_LINKER_LIBS})

#FILE(GLOB_RECURSE model_source ${CMAKE_CURRENT_SOURCE_DIR}/model/ "*.cc")
AUX_SOURCE_DIRECTORY(model model_source)
AUX_SOURCE_DIRECTORY(model/layer model_source)
AUX_SOURCE_DIRECTORY(model/optimizer model_source)
AUX_SOURCE_DIRECTORY(model/loss model_source)
AUX_SOURCE_DIRECTORY(model/metric model_source)
AUX_SOURCE_DIRECTORY(model/updater model_source)
#MESSAGE(STATUS "MODEL ${model_source}")
ADD_LIBRARY(singa_model SHARED ${model_source})
MESSAGE(STATUS "model linker libs ${SINGA_LINKER_LIBS}")
TARGET_LINK_LIBRARIES(singa_model ${SINGA_LINKER_LIBS})
LIST(APPEND SINGA_LINKER_LIBS singa_model)

AUX_SOURCE_DIRECTORY(io io_source)
AUX_SOURCE_DIRECTORY(io/network io_source)
ADD_LIBRARY(singa_io SHARED ${io_source})
TARGET_LINK_LIBRARIES(singa_io ${SINGA_LINKER_LIBS})
LIST(APPEND SINGA_LINKER_LIBS singa_io)

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
    ADD_LIBRARY(_singa_wrap SHARED "${python_srcs} ${python_cxxs} ${cuda_objs}")
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
