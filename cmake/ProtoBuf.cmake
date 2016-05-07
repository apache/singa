#/**
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

# This file is adpated from Caffe cmake/ProtoBuf.cmake.
# We changed 'caffe' to 'singa'

# Finds Google Protocol Buffers library and compilers and extends
# the standard cmake script with version and python generation support

find_package( Protobuf REQUIRED )
include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})
MESSAGE(STATUS "proto libs " ${PROTOBUF_LIBRARIES})
list(APPEND singa_linker_libs ${PROTOBUF_LIBRARIES})

# As of Ubuntu 14.04 protoc is no longer a part of libprotobuf-dev package
# and should be installed separately as in: sudo apt-get install
# protobuf-compiler
if(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
  message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
else()
  message(FATAL_ERROR "Could not find PROTOBUF Compiler")
endif()

#if(PROTOBUF_FOUND)
#  # fetches protobuf version
#  caffe_parse_header(${PROTOBUF_INCLUDE_DIR}/google/protobuf/stubs/common.h VERION_LINE GOOGLE_PROTOBUF_VERSION)
#  string(REGEX MATCH "([0-9])00([0-9])00([0-9])" PROTOBUF_VERSION ${GOOGLE_PROTOBUF_VERSION})
#  set(PROTOBUF_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
#  unset(GOOGLE_PROTOBUF_VERSION)
#endif()

# place where to generate protobuf sources
set(proto_gen_folder "${PROJECT_BINARY_DIR}/include/singa/proto")
include_directories("${PROJECT_BINARY_DIR}/include")

set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)

###############################################################################
# Modification of standard 'protobuf_generate_cpp()' with output dir parameter
# and python support
# Usage:
#   singa_protobuf_generate_cpp_py(<output_dir> <srcs_var> <hdrs_var>
#                                  <python_var> <proto_files>)
function(singa_protobuf_generate_cpp_py output_dir srcs_var hdrs_var python_var)
  if(NOT ARGN)
    message(SEND_ERROR
      "Error: singa_protobuf_generate_cpp_py() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protoc_include -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(dir ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(abs_path ${dir} ABSOLUTE)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  set(${python_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${output_dir}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${output_dir}/${fil_we}.pb.h")
    list(APPEND ${python_var} "${output_dir}/${fil_we}_pb2.py")

    add_custom_command(
      OUTPUT "${output_dir}/${fil_we}.pb.cc"
             "${output_dir}/${fil_we}.pb.h"
             "${output_dir}/${fil_we}_pb2.py"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out    ${output_dir} ${_protoc_include} ${abs_fil}
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${output_dir} ${_protoc_include} ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()
