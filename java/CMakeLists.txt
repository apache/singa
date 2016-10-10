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
function(CAPTALIZE_STRING INPUT OUTPUT)
  string(TOUPPER ${ARGV0} UPP)
  string(LENGTH ${ARGV0} LEN)
  string(SUBSTRING ${UPP} 0 1 PREFIX)
  math(EXPR LEN "${LEN} - 1")
  string(SUBSTRING ${ARGV0} 1 ${LEN} SUFFIX)
  string(CONCAT RET ${PREFIX} ${SUFFIX})
  set(${OUTPUT} ${RET} PARENT_SCOPE)
endfunction()

function(PROTOBUF_GENERATE_JAVA OUTPUT)
    if(NOT ARGN)
        message(SEND_ERROR "Error: PROTOBUF_GENERATE_JAVA() called
        without any proto files")
        return()
    endif(NOT ARGN)

    set(${OUTPUT})
    foreach(FIL ${ARGN})
        get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
        get_filename_component(FIL_WE ${FIL} NAME_WE)
        get_filename_component(PATH ${FIL} PATH)
        CAPTALIZE_STRING(${FIL_WE} FIL_WE1)
        list(APPEND ${OUTPUT} "${CMAKE_BINARY_DIR}/java/src/main/java/org/apache/singa/proto/${FIL_WE1}.java")

        add_custom_command(
            OUTPUT "${CMAKE_BINARY_DIR}/java/src/main/java/org/apache/singa/proto/${FIL_WE1}.java"
            COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
            ARGS --java_out ${CMAKE_BINARY_DIR}/java/src/main/java
                 --proto_path ${PATH} ${ABS_FIL}
            DEPENDS ${ABS_FIL}
            COMMENT "Running java protocol buffer compiler on ${FIL}" VERBATIM)
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

        # Delete symlink if it exists
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${path_file}")

        # Get OS dependent path to use in `execute_process`
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${path_file}" link)
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${path_file}" target)
        # MESSAGE(STATUS  "${link}->${target}")

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

# generate cxx and wrap.py
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/java/src/main/java/org/apache/singa/swig)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/java/src/main/java/org/apache/singa/proto)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/java/src/test/java/org/apache/singa)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/java/lib)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/api)

# generate protobuf sources
FILE(GLOB proto_files ${CMAKE_SOURCE_DIR}/src/proto/*.proto)
PROTOBUF_GENERATE_JAVA(proto_javas ${proto_files})
# MESSAGE(STATUS "proto javas: ${proto_javas}")

execute_process(
    COMMAND swig -c++ -java -I${CMAKE_SOURCE_DIR}/include
    -I${JAVA_INCLUDE_PATH} -I${JAVA_INCLUDE_PATH2}
    -package org.apache.singa.swig
    -outdir ${CMAKE_BINARY_DIR}/java/src/main/java/org/apache/singa/swig
    -o ${CMAKE_BINARY_DIR}/src/api/singa_java_wrap.cxx
    ${CMAKE_SOURCE_DIR}/src/api/singa.i)

#MESSAGE(STATUS "java include: ${JAVA_INCLUDE_DIRS} ${JNI_INCLUDE_DIRS} ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2}")
set(java_srcs "${CMAKE_BINARY_DIR}/src/api/singa_java_wrap.cxx")

#Create symlinks for all java source files  Do not omit !!!RELATIVE!!!
file(GLOB_RECURSE java_source_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.java)
file(GLOB_RECURSE java_xml_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.xml)
list(APPEND java_source_files ${java_xml_files})
create_symlinks(${java_source_files})

IF(USE_CUDA)
# remain this custom command to avoid cuda objs can't find
ADD_CUSTOM_COMMAND(
    OUTPUT ${global_cuda_objs}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/"
    )
ENDIF(USE_CUDA)


ADD_LIBRARY(singa_wrap SHARED $<TARGET_OBJECTS:singa_objects>  ${java_srcs} ${proto_javas} ${global_cuda_objs})
TARGET_LINK_LIBRARIES(singa_wrap ${SINGA_LINKER_LIBS} ${JNI_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(singa_wrap PRIVATE ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})

SET_TARGET_PROPERTIES(singa_wrap
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/java/lib
)

IF(APPLE)
ADD_CUSTOM_TARGET(
    change_suffix ALL
    COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_BINARY_DIR}/java/lib/libsinga_wrap.dylib" "${CMAKE_BINARY_DIR}/java/lib/libsinga_wrap.so"
    COMMENT "change .dylib to .so in mac system"
)
ADD_DEPENDENCIES(change_suffix _singa_wrap)
ENDIF(APPLE)
