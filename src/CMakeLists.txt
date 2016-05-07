# generate protobuf sources
file(GLOB proto_files proto/*.proto)
singa_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})
# include python files either to force generation
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
list(APPEND singa_linker_libs proto)

FILE(GLOB_RECURSE utils_source ${CMAKE_CURRENT_SOURCE_DIR}/utils/ "*.cc")
ADD_LIBRARY(singa_utils SHARED ${utils_source})
TARGET_LINK_LIBRARIES(singa_utils ${singa_linker_libs})
list(APPEND singa_linker_libs singa_utils)


FILE(GLOB_RECURSE core_source ${CMAKE_CURRENT_SOURCE_DIR}/core/ "*.cc")
ADD_LIBRARY(singa_core SHARED ${core_source})
TARGET_LINK_LIBRARIES(singa_core ${singa_linker_libs})
list(APPEND singa_linker_libs singa_core)
MESSAGE(STATUS "link libs " ${singa_linker_libs})

#ADD_LIBRARY(singa_layer SHARED ${LAYER_SOURCE})
#ADD_LIBRARY(singa_model SHARED ${MODEL_SOURCE})
#ADD_LIBRARY(singa_utils SHARED ${UTILS_SOURCE})

#TARGET_LINK_LIBRARIES(singa_core singa_utils)
#TARGET_LINK_LIBRARIES(singa_layer singa_core singa_utils)
#TARGET_LINK_LIBRARIES(singa_model singa_layer singa_core singa_utils)

#ADD_LIBRARY(singa SHARED ${singa_linker_libs})
