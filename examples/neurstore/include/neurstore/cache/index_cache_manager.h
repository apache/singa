#ifndef INDEX_CACHE_MANAGER_H
#define INDEX_CACHE_MANAGER_H

#ifdef __cplusplus
/*********************************** C++ ***********************************/
#include "neurstore/compress/builder/tensor_similarity_index.h"
#include "neurstore/compress/index/router/router.h"
#include "neurstore/utils/global.h"

/**
 * IndexCacheManager is a global cache manager for indexes.
 * The only two exposed methods are `get` and `release`. Note that HNSWLIB indexes are thread-safe,
 * thus, the cache manager can be used in a multi-threaded environment. Just make sure not to
 * evict the cache while an index is being used by another thread.
 */
class IndexCacheManager {
public:
    explicit IndexCacheManager(std::string store_path, int64_t max_bytes = INDEX_CACHE_SIZE);

    /**
     * Get an index from the cache.
     * @param dimension
     * @param index_id
     * @return
     */
    std::shared_ptr<VectorUInt8Index> get(
        int dimension,
        int index_id
    );

    /**
     * Release an index.
     * @param index_ptr
     * @param dimension
     * @param index_id
     * @param is_updated
     */
    void release(
        std::shared_ptr<VectorUInt8Index> index_ptr,
        int dimension,
        int index_id,
        bool is_updated
    );

    /**
     * Save an index to the disk.
     * @param key
     * @param index
     */
    void saveIndex(
        const std::string &key,
        const std::shared_ptr<VectorUInt8Index> &index
    ) const;

    /**
     * Get a router for a specific dimension.
     * @param dimension
     */
    std::shared_ptr<VectorFloat64Router> getRouter(int dimension);

    /**
     * Save all routers to the disk.
     */
    void saveRouters() const;

    /**
     * Load all routers from the disk.
     * This method is called during the initialization of the IndexCacheManager.
     */
    void loadRouters();

    /**
     * Save a list of indexes to the disk.
     * @param indexes
     */
    void saveIndexes(
        const std::list<std::pair<std::string, std::shared_ptr<VectorUInt8Index> > > &indexes
    ) const;

    void clearCache();

private:
    std::string store_path_;
    int64_t max_bytes_;
    int64_t used_bytes_ = 0;
    std::unordered_map<std::string, int64_t> size_map_;
    std::unordered_map<std::string, bool> updated_map_;
    std::unordered_map<std::string, int> use_count_map_;

    std::unordered_map<
        std::string, std::list<std::pair<std::string, std::shared_ptr<VectorUInt8Index> > >::iterator
    > cache_map_; // key: index name, value: iterator to cache_list_
    std::list<std::pair<std::string, std::shared_ptr<VectorUInt8Index> > > cache_list_; // LRU

    std::mutex index_cache_mutex_;

    // router related
    std::unordered_map<int, std::shared_ptr<VectorFloat64Router>> router_map_;
    std::mutex router_mutex_;

    /**
     * Evict the cache if the cache size exceeds the maximum limit.
     */
    void evictCacheIfNeeded();

    /**
     * Mark an index as dirty, meaning it has been updated and needs to be saved when being
     * evicted.
     * @param key
     */
    void markDirty(
        const std::string &key
    );
};

extern "C" {
#endif // __cplusplus
/*********************************** C ***********************************/

typedef struct IndexCacheManager IndexCacheManagerC;

IndexCacheManagerC* icm_create(const char *store_path);

void icm_clear_cache(IndexCacheManagerC *icm);

void icm_destroy(IndexCacheManagerC *icm);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif //INDEX_CACHE_MANAGER_H
