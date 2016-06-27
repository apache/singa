INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)

AUX_SOURCE_DIRECTORY(cifar10 cifar_source)

ADD_EXECUTABLE(alexnet ${cifar_source})
ADD_DEPENDENCIES(alexnet singa_core singa_model singa_utils)
TARGET_LINK_LIBRARIES(alexnet singa_core singa_utils singa_model protobuf ${SINGA_LIBKER_LIBS})
