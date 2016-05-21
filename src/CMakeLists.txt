# generate protobuf sources
FILE(GLOB proto_files proto/*.proto)
singa_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})
# include python files either to force generation
ADD_LIBRARY(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
LIST(APPEND singa_linker_libs proto)

#FILE(GLOB_RECURSE utils_source ${CMAKE_CURRENT_SOURCE_DIR}/utils/ "*.cc")
AUX_SOURCE_DIRECTORY(utils utils_source)
#message(STATUS "UTILS ${utils_source}")
ADD_LIBRARY(singa_utils SHARED ${utils_source})
TARGET_LINK_LIBRARIES(singa_utils ${singa_linker_libs})
LIST(APPEND singa_linker_libs singa_utils)


#FILE(GLOB_RECURSE core_source ${CMAKE_CURRENT_SOURCE_DIR}/core/ "*.cc")
AUX_SOURCE_DIRECTORY(core/device core_source)
AUX_SOURCE_DIRECTORY(core/memory core_source)
AUX_SOURCE_DIRECTORY(core/scheduler core_source)
AUX_SOURCE_DIRECTORY(core/tensor core_source)
#message(STATUS "CORE ${core_source}")
ADD_LIBRARY(singa_core SHARED ${core_source})
TARGET_LINK_LIBRARIES(singa_core ${singa_linker_libs})
LIST(APPEND singa_linker_libs singa_core)
#MESSAGE(STATUS "link libs " ${singa_linker_libs})

#FILE(GLOB_RECURSE model_source ${CMAKE_CURRENT_SOURCE_DIR}/model/ "*.cc")
AUX_SOURCE_DIRECTORY(model model_source)
AUX_SOURCE_DIRECTORY(model/layer model_source)
#MESSAGE(STATUS "MODEL ${model_source}")
ADD_LIBRARY(singa_model SHARED ${model_source})
TARGET_LINK_LIBRARIES(singa_model ${singa_linker_libs})
LIST(APPEND singa_linker_libs singa_model)

#ADD_LIBRARY(singa_layer SHARED ${LAYER_SOURCE})
#ADD_LIBRARY(singa_model SHARED ${MODEL_SOURCE})
#ADD_LIBRARY(singa_utils SHARED ${UTILS_SOURCE})

#TARGET_LINK_LIBRARIES(singa_core singa_utils)
#TARGET_LINK_LIBRARIES(singa_layer singa_core singa_utils)
#TARGET_LINK_LIBRARIES(singa_model singa_layer singa_core singa_utils)

#ADD_LIBRARY(singa SHARED ${singa_linker_libs})
