
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

