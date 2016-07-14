INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)

AUX_SOURCE_DIRECTORY(cifar10 cifar_source)

IF(USE_CUDNN)
ADD_EXECUTABLE(alexnet ${cifar_source})
ADD_DEPENDENCIES(alexnet singa_core singa_model singa_utils)
TARGET_LINK_LIBRARIES(alexnet singa_core singa_utils singa_model protobuf ${SINGA_LIBKER_LIBS})
ENDIF(USE_CUDNN)

AUX_SOURCE_DIRECTORY(cifar10-parallel cifar_parallel_source)

IF(USE_CUDNN)
ADD_EXECUTABLE(alexnet-parallel ${cifar_parallel_source})
ADD_DEPENDENCIES(alexnet-parallel singa_core singa_model singa_utils)
TARGET_LINK_LIBRARIES(alexnet-parallel singa_core singa_utils singa_model protobuf ${SINGA_LIBKER_LIBS})
SET_TARGET_PROPERTIES(alexnet-parallel PROPERTIES LINK_FLAGS "${LINK_FLAGS} -pthread")
ENDIF(USE_CUDNN)
