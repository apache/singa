#ifndef QUANTIZED_HNSW_HNSWLIB_H
#define QUANTIZED_HNSW_HNSWLIB_H


#include <condition_variable>
#include <shared_mutex>
#include <hnswlib/hnswlib.h>

#include "neurstore/compress/index/similarity_index.h"


class QueryInsertionBarrier {
public:
    void enterQuery() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this]() { return active_insertions == 0; });
        active_queries++;
    }

    void exitQuery() {
        std::unique_lock lock(mutex_);
        active_queries--;
        if (active_queries == 0) {
            cv_.notify_all();
        }
    }

    void enterInsertion() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this]() { return active_queries == 0; });
        active_insertions++;
    }

    void exitInsertion() {
        std::unique_lock lock(mutex_);
        active_insertions--;
        if (active_insertions == 0) {
            cv_.notify_all();
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int active_insertions = 0;
    int active_queries = 0;
};


class QuantizedHNSWIndexHNSWLIB final : public SimilarityIndex<TensorType::VectorUInt8> {
public:
    explicit QuantizedHNSWIndexHNSWLIB(
        int dimension,
        int64_t next_id = 0,
        size_t max_elements = 400
    );

    ~QuantizedHNSWIndexHNSWLIB() override;

    QuantizedHNSWIndexHNSWLIB(
        std::unique_ptr<hnswlib::HierarchicalNSW<float> > index,
        int64_t next_id,
        std::unique_ptr<hnswlib::QuantizedL2Space> space
    );

    int64_t insert(const TensorType::VectorUInt8 &vector, double scale, double zero_point) override;

    std::vector<int64_t> insertMany(
        std::vector<TensorType::VectorUInt8> &vectors,
        std::vector<double> &scales,
        std::vector<double> &zero_points
    ) override;

    std::vector<int64_t> query(
        const TensorType::VectorUInt8 &vector,
        int k,
        double scale,
        double zero_point
    ) const override;

    TensorType::VectorFloat64 retrieve(int64_t id) const override;

    TensorType::VectorFloat16 retrieveF16(int64_t id) const override;

    UINT8QuantizedTensorPacket retrieveUINT8Quantized(int64_t id) const override;

    int64_t size() const override;

    int64_t space() const override;

    void resize();

    void serialize(std::ostream &out) const override;

    static std::shared_ptr<QuantizedHNSWIndexHNSWLIB> deserialize(std::istream &in);

    static std::shared_ptr<QuantizedHNSWIndexHNSWLIB> deserializeFromMMap(const uint8_t *data, size_t size);

private:
    std::unique_ptr<hnswlib::SpaceInterface<float> > space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float> > index_;

    int ef_search_;
    int ef_construction_;
    int m_;
    size_t max_elements_;
    std::atomic<int64_t> next_id_;

    bool is_mmap_ = false;
    void *mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;

    mutable QueryInsertionBarrier query_insertion_barrier_;
    mutable std::shared_mutex resize_mutex_;

    void addPointInternal(
        const TensorType::VectorUInt8 &vector,
        double scale,
        double zero_point,
        int64_t id
    );
};


#endif //QUANTIZED_HNSW_HNSWLIB_H
