#ifndef GLOBAL_H
#define GLOBAL_H

#include "neurstore/utils/timer.h"
#include "hnswlib/hnswlib.h"

// ###### CONFIGURABLE PARAMETERS (neurstore-core) ######

inline int64_t MAXIMUM_INDEX_SIZE = 1024LL * 1024 * 1024 * 2;   // 2 GB
inline int64_t INDEX_CACHE_SIZE = 32 * 1024LL * 1024 * 1024;    // 32 GB
inline int INDEX_RESIZE_TIMES = 5;                              // 2^5 = 32 times the initial size

struct HNSWIndexParams {
    int m;
    int ef_construction;
    int ef_search;
};

inline HNSWIndexParams getHNSWParamsForDim(int dimension) {
    if (dimension <= 512) {
        return {24, 64, 64};
    } else if (dimension <= 4096) {
        return {24, 32, 48};
    } else if (dimension <= 65536) {
        return {24, 16, 24};
    } else if (dimension <= 589822) {
        return {12, 8, 12};
    } else if (dimension <= 1048574) {
        return {6, 6, 8};
    } else if (dimension <= 2359294) {
        return {4, 4, 6};
    } else {
        return {4, 3, 6};
    }
}

// ###### DO NOT CHANGE ######

inline Timer *TIMER = new Timer();

inline int getInitialIndexCapacity(const int dimension, const int resize_times = 4) {
    const auto params = getHNSWParamsForDim(dimension);
    const int maxM0 = params.m * 2;
    const size_t size_links_level0 = maxM0 * sizeof(hnswlib::tableint) + sizeof(hnswlib::linklistsizeint);
    const size_t size_data_per_element = size_links_level0 + dimension * sizeof(uint8_t) + sizeof(hnswlib::labeltype);
    const size_t per_element_bytes = size_data_per_element + sizeof(void*) + sizeof(uint8_t) + sizeof(std::mutex) + sizeof(int);
    size_t denominator = per_element_bytes << resize_times;
    if (denominator == 0) {
        denominator = 1;
    }
    return static_cast<int>(MAXIMUM_INDEX_SIZE / denominator);
}

#endif //GLOBAL_H
