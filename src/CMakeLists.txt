# generate protobuf sources
FILE(GLOB proto_files proto/*.proto)
protobuf_generate_cpp(proto_srcs proto_hdrs ${proto_files})
protobuf_generate_python(proto_pys ${proto_files})
INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/include")
#message(STATUS "include: ${CMAKE_BINARY_DIR} ")
#message(STATUS "srcs: ${proto_srcs}")
#message(STATUS "hdrs: ${proto_hdrs}")
#message(STATUS "pys: ${proto_pys}")
ADD_LIBRARY(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_pys})
FOREACH(fil ${proto_hdrs})
    ADD_CUSTOM_COMMAND(
        TARGET proto PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/include/singa/proto"
        COMMAND ${CMAKE_COMMAND} -E copy ${fil} "${CMAKE_BINARY_DIR}/include/singa/proto"
        #COMMAND ${CMAKE_COMMAND} -E echo "copy done"
        )
ENDFOREACH()
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
FILE(GLOB_RECURSE cuda_source core "*.cu")
SET(FLAGS_BACKUP ${CMAKE_CXX_FLAGS})
SET(CMAKE_CXX_FLAGS "")
CUDA_COMPILE(cuda_objs SHARED ${cuda_source} OPTIONS "-Xcompiler -fPIC")
#message(STATUS "FLAGS ${CMAKE_CXX_FLAGS}")
#message(STATUS "CORE ${cuda_source}")
#message(STATUS "OBJ ${cuda_objs}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/core/tensor")
SET(CMAKE_CXX_FLAGS ${FLAGS_BACKUP})
ADD_LIBRARY(singa_core SHARED ${core_source} ${cuda_objs})
TARGET_LINK_LIBRARIES(singa_core ${SINGA_LINKER_LIBS})
LIST(APPEND SINGA_LINKER_LIBS singa_core)
#MESSAGE(STATUS "link libs " ${SINGA_LINKER_LIBS})

#FILE(GLOB_RECURSE model_source ${CMAKE_CURRENT_SOURCE_DIR}/model/ "*.cc")
AUX_SOURCE_DIRECTORY(model model_source)
AUX_SOURCE_DIRECTORY(model/layer model_source)
AUX_SOURCE_DIRECTORY(model/optimizer model_source)
AUX_SOURCE_DIRECTORY(model/loss model_source)
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
