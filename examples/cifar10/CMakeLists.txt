INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)

IF(USE_CUDNN)
ADD_EXECUTABLE(alexnet alexnet.cc)
ADD_DEPENDENCIES(alexnet singa_core singa_model singa_utils)
TARGET_LINK_LIBRARIES(alexnet singa_core singa_utils singa_model protobuf ${SINGA_LIBKER_LIBS})

ADD_EXECUTABLE(alexnet-parallel alexnet-parallel.cc)
ADD_DEPENDENCIES(alexnet-parallel singa_core singa_model singa_utils)
TARGET_LINK_LIBRARIES(alexnet-parallel singa_core singa_utils singa_model protobuf ${SINGA_LIBKER_LIBS})
SET_TARGET_PROPERTIES(alexnet-parallel PROPERTIES LINK_FLAGS "${LINK_FLAGS} -pthread")

ADD_EXECUTABLE(vgg-parallel vgg-parallel.cc)
ADD_DEPENDENCIES(vgg-parallel singa_core singa_model singa_utils)
TARGET_LINK_LIBRARIES(vgg-parallel singa_core singa_utils singa_model protobuf ${SINGA_LIBKER_LIBS})
SET_TARGET_PROPERTIES(vgg-parallel PROPERTIES LINK_FLAGS "${LINK_FLAGS} -pthread")
ENDIF(USE_CUDNN)
