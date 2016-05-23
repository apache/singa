# generate protobuf sources
FILE(GLOB proto_files proto/*.proto)
singa_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})
# include python files either to force generation
ADD_LIBRARY(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
LIST(APPEND SINGA_LINKER_LIBS proto)

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
#message(STATUS "CORE ${core_source}")
ADD_LIBRARY(singa_core SHARED ${core_source})
TARGET_LINK_LIBRARIES(singa_core ${SINGA_LINKER_LIBS})
LIST(APPEND SINGA_LINKER_LIBS singa_core)
#MESSAGE(STATUS "link libs " ${SINGA_LINKER_LIBS})

#FILE(GLOB_RECURSE model_source ${CMAKE_CURRENT_SOURCE_DIR}/model/ "*.cc")
AUX_SOURCE_DIRECTORY(model model_source)
AUX_SOURCE_DIRECTORY(model/layer model_source)
#MESSAGE(STATUS "MODEL ${model_source}")
ADD_LIBRARY(singa_model SHARED ${model_source})
TARGET_LINK_LIBRARIES(singa_model ${SINGA_LINKER_LIBS})
LIST(APPEND SINGA_LINKER_LIBS singa_model)

#ADD_LIBRARY(singa_layer SHARED ${LAYER_SOURCE})
#ADD_LIBRARY(singa_model SHARED ${MODEL_SOURCE})
#ADD_LIBRARY(singa_utils SHARED ${UTILS_SOURCE})

#TARGET_LINK_LIBRARIES(singa_core singa_utils)
#TARGET_LINK_LIBRARIES(singa_layer singa_core singa_utils)
#TARGET_LINK_LIBRARIES(singa_model singa_layer singa_core singa_utils)

#ADD_LIBRARY(singa SHARED ${SINGA_LINKER_LIBS})
