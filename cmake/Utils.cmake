
macro(swig_generate_cxx pylist_variable)
    if(NOT EXISTS "${CMKAE_BINARY_DIR}/python")
        execute_process(
            COMMAND mkdir ${CMAKE_BINARY_DIR}/python
            COMMAND mkdir ${CMAKE_BINARY_DIR}/python/singa
            COMMAND mkdir ${CMAKE_BINARY_DIR}/python/singa/proto
            ERROR_QUIET)
    endif()
    execute_process(
        COMMAND swig -c++ -python -I${CMAKE_SOURCE_DIR}/include 
        -outdir ${CMAKE_BINARY_DIR}/python/singa
        ${ARGN})

    set(${pylist_variable} "${CMAKE_SOURCE_DIR}/src/python/swig/singa_wrap.cxx")
endmacro()

function (create_symlinks)
    # Do nothing if building in-source
    if (${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
        return()
    endif()

    foreach (path_file ${ARGN})
        get_filename_component(folder ${path_file} PATH)

        # Create REAL folder
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${folder}")

        # Delete symlink if it exists
        file(REMOVE "${CMAKE_BINARY_DIR}/${path_file}")

        # Get OS dependent path to use in `execute_process`
        file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/${path_file}" link)
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