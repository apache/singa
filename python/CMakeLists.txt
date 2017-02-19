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

# This following function is taken from
# https://github.com/Kitware/CMake/blob/master/Modules/FindProtobuf.cmake
# and modified to our compilation.
function(PROTOBUF_GENERATE_PYTHON OUTPUT)
    if(NOT ARGN)
        message(SEND_ERROR "Error: PROTOBUF_GENERATE_PYTHON() called
        without any proto files")
        return()
    endif(NOT ARGN)

    set(${OUTPUT})
    foreach(FIL ${ARGN})
        get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
        get_filename_component(FIL_WE ${FIL} NAME_WE)
        get_filename_component(PATH ${FIL} PATH)

        list(APPEND ${OUTPUT} "${CMAKE_BINARY_DIR}/python/singa/proto/${FIL_WE}_pb2.py")

        add_custom_command(
            OUTPUT "${CMAKE_BINARY_DIR}/python/singa/proto/${FIL_WE}_pb2.py"
            COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
            ARGS --python_out ${CMAKE_BINARY_DIR}/python/singa/proto
                 --proto_path ${PATH} ${ABS_FIL}
            DEPENDS ${ABS_FIL}
            COMMENT "Running Python protocol buffer compiler on ${FIL}" VERBATIM)
    endforeach()

    set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
    set(${OUTPUT} ${${OUTPUT}} PARENT_SCOPE)
endfunction()

function (create_symlinks)
    # Do nothing if building in-source
    if (${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
        return()
    endif()

    foreach (path_file ${ARGN})
        get_filename_component(folder ${path_file} PATH)

        # Create REAL folder
        #file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${folder}")

        # Delete symlink if it exists
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${path_file}")

        # Get OS dependent path to use in `execute_process`
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${path_file}" link)
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${path_file}" target)

        if (UNIX)
            set(command ln -s ${target} ${link})
        else()
            set(command cmd.exe /c mklink ${link} ${target})
        endif()

        execute_process(COMMAND ${command}
                        RESULT_VARIABLE result
                        ERROR_VARIABLE output)

        if (NOT ${result} EQUAL 0)
            message(FATAL_ERROR "Could not create symbolic link for: ${target} --> ${output}")
        endif()

    endforeach(path_file)
endfunction(create_symlinks)


# generate protobuf sources
FILE(GLOB proto_files ${CMAKE_SOURCE_DIR}/src/proto/*.proto)
PROTOBUF_GENERATE_PYTHON(proto_pys ${proto_files})
#MESSAGE(STATUS "proto pys: ${proto_pys}")

# generate cxx and wrap.py
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/singa/proto)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/rafiki)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/api)
execute_process(
    COMMAND swig -c++ -python -I${CMAKE_SOURCE_DIR}/include
    -outdir ${CMAKE_BINARY_DIR}/python/singa
    -o ${CMAKE_BINARY_DIR}/src/api/singa_wrap.cxx
    ${CMAKE_SOURCE_DIR}/src/api/singa.i)

set(python_srcs "${CMAKE_BINARY_DIR}/src/api/singa_wrap.cxx")

#Create symlinks for all python source files  Do not omit !!!RELATIVE!!!
file(GLOB_RECURSE python_source_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
create_symlinks(${python_source_files})


IF(USE_CUDA)
# remain this custom command to avoid cuda objs can't find
ADD_CUSTOM_COMMAND(
    OUTPUT ${global_cuda_objs}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/"
    )
ENDIF(USE_CUDA)

ADD_LIBRARY(_singa_wrap SHARED $<TARGET_OBJECTS:singa_objects>  ${python_srcs} ${proto_pys} ${global_cuda_objs})
TARGET_LINK_LIBRARIES(_singa_wrap ${SINGA_LINKER_LIBS} ${PYTHON_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(_singa_wrap PRIVATE ${PYTHON_INCLUDE_DIRS})
SET_TARGET_PROPERTIES(_singa_wrap
PROPERTIES PREFIX ""
LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/singa
)

#SETUP
SET(SETUP_PY_IN "setup.py.in")
SET(SETUP_PY    "${CMAKE_BINARY_DIR}/python/setup.py")
CONFIGURE_FILE(${SETUP_PY_IN} ${SETUP_PY})

#create python/singa/proto/__init__.py
FILE(WRITE ${CMAKE_BINARY_DIR}/python/singa/proto/__init__.py "")
