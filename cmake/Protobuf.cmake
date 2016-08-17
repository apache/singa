# This script is taken from
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
