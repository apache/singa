
find_path(CUDNN_INCLUDE_DIR NAME "cudnn.h" PATHS "$ENV{CMAKE_INCLUDE_PATH}")
find_library(CUDNN_LIBRARIES NAME "libcudnn.so" PATHS "$ENV{CMAKE_LIBRARY_PATH}")

#message("cudnn include path:${CUDNN_INCLUDE_DIR}  lib path: ${CUDNN_LIBRARIES}")
#message("env include path:$ENV{CUDNN_DIR} next: $ENV{CMAKE_INCLUDE_PATH}")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARIES)

if(CUDNN_FOUND)
    file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
        CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
        CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
        CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
        CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
        CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
        CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

    if(NOT CUDNN_VERSION_MAJOR)
        set(CUDNN_VERSION "???")
    else()
        set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    endif()
    message(STATUS "Found Cudnn_v${CUDNN_VERSION} at ${CUDNN_INCLUDE_DIR}")
    mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARIES)

endif()
