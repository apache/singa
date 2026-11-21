#ifndef TENSOR_SIMILARITY_INDEX_H
#define TENSOR_SIMILARITY_INDEX_H

#include <memory>
#include <shared_mutex>
#include <vector>

#include "neurstore/utils/tensor.h"
#include "neurstore/compress/index/similarity_index.h"
#include "neurstore/compress/index/quantized_hnsw_hnswlib.h"


template<typename T>
class TensorSimilarityIndex {
public:
    TensorSimilarityIndex(
        std::shared_ptr<SimilarityIndex<T> > index,
        int original_bit_width,
        int dimension
    ) {
        index_ = std::move(index);
        original_bit_width_ = original_bit_width;
        dimension_ = dimension;
        if (original_bit_width != 16 && original_bit_width != 32 && original_bit_width != 64) {
            throw std::invalid_argument("TensorIndex::TensorIndex: unsupported bit width.");
        }
    }

    /**
     * Insert a tensor into the index
     * @param tensor the tensor to insert
     * @param scale the scale of the tensor
     * @param zero_point the zero point of the tensor
     * @return the id of the inserted tensor
     */
    int64_t insert(const T &tensor, double scale, double zero_point) const {
        if (tensor.size() != dimension_) {
            throw std::invalid_argument("TensorIndex::insert: tensor has wrong shape.");
        }
        return index_->insert(tensor, scale, zero_point);
    }

    /**
     * Insert multiple tensors into the index
     * @param tensors the tensors to insert
     * @param scales the scales of the tensors
     * @param zero_points the zero points of the tensors
     * @return the ids of the inserted tensors
     */
    std::vector<int64_t> insertMany(
        const std::vector<T> &tensors,
        const std::vector<double> &scales,
        const std::vector<double> &zero_points
    ) const {
        std::vector<T> flattened_tensors;
        for (const auto &tensor: tensors) {
            if (tensor.size() != dimension_) {
                throw std::invalid_argument("TensorIndex::insertMany: tensor has wrong shape.");
            }
            flattened_tensors.emplace_back(Eigen::Map<const T>(tensor.data(), tensor.size()));
        }
        return index_->insertMany(flattened_tensors, scales, zero_points);
    }

    /**
     * Retrieve a tensor from the index
     * @param id the id of the tensor
     * @return the retrieved tensor
     */
    TensorType::VectorFloat64 retrieve(int64_t id) const {
        auto tensor = index_->retrieve(id);
        if (tensor.size() != dimension_) {
            throw std::invalid_argument("TensorIndex::retrieve: tensor has wrong shape.");
        }
        return tensor;
    }

    TensorType::VectorFloat16 retrieveF16(int64_t id) const {
        auto tensor = index_->retrieveF16(id);
        if (tensor.size() != dimension_) {
            throw std::invalid_argument("TensorIndex::retrieveF16: tensor has wrong shape.");
        }
        return tensor;
    }

    UINT8QuantizedTensorPacket retrieveUINT8Quantized(int64_t id) const {
        auto tensor = index_->retrieveUINT8Quantized(id);
        if (tensor.data.size() != dimension_) {
            throw std::invalid_argument("TensorIndex::retrieveUINT8Quantized: tensor has wrong shape.");
        }
        return tensor;
    }

    /**
     * Query the index for the most similar tensor
     * @param tensor the tensor to query
     * @param scale the scale of the tensor
     * @param zero_point the zero point of the tensor
     * @return the id of the most similar tensor
     */
    int64_t query(const T &tensor, double scale, double zero_point) const {
        if (tensor.size() != dimension_) {
            throw std::invalid_argument("TensorIndex::query: tensor has wrong shape.");
        }
        std::vector<int64_t> result = index_->query(tensor, 1, scale, zero_point);
        if (result.empty()) {
            return -1;
        }
        return result.front();
    }

    /**
    * Get the number of tensors in the index
    * @return number of tensors
    */
    int64_t size() const {
        return index_->size();
    }

    /**
     * Get the space used by the index
     * @return space used in bytes
     */
    int64_t space() const {
        return index_->space();
    }

    /**
     * Serialize the index into a byte array
     * @return serialized byte array
     */
    std::vector<uint8_t> serialize() const {
        std::ostringstream oss(std::ios::binary);
        oss.write(reinterpret_cast<const char *>(&original_bit_width_), sizeof(original_bit_width_));
        const size_t dimension = dimension_;
        oss.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
        index_->serialize(oss);
        const std::string serialized_index = oss.str();
        return std::vector<uint8_t>(serialized_index.begin(), serialized_index.end());
    }

    void serializeToStream(std::ostream &out) const {
        out.write(reinterpret_cast<const char *>(&original_bit_width_), sizeof(original_bit_width_));
        const size_t dimension = dimension_;
        out.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
        index_->serialize(out);
    }

    /**
     * Get the original bit width of the tensors
     * @return the original bit width
     */
    int getOriginalBitWidth() const {
        return original_bit_width_;
    }

    int getDimension() const {
        return dimension_;
    }

    /**
     * Load the index from a byte array (static method)
     */
    static std::shared_ptr<TensorSimilarityIndex> deserialize(std::istream &in) {
        int original_bit_width;
        size_t dimension;
        in.read(reinterpret_cast<char*>(&original_bit_width), sizeof original_bit_width);
        in.read(reinterpret_cast<char*>(&dimension), sizeof dimension);

        auto index = QuantizedHNSWIndexHNSWLIB::deserialize(in);

        return std::make_shared<TensorSimilarityIndex>(
            index,
            original_bit_width,
            dimension
        );
    }

    static std::shared_ptr<TensorSimilarityIndex> deserializeFromMMap(const uint8_t* mmap_ptr, size_t size) {
        size_t offset = 0;
        int original_bit_width = 0;
        std::memcpy(&original_bit_width, mmap_ptr + offset, sizeof(original_bit_width));
        offset += sizeof(original_bit_width);

        size_t dimension = 0;
        std::memcpy(&dimension, mmap_ptr + offset, sizeof(dimension));
        offset += sizeof(dimension);

        auto index = QuantizedHNSWIndexHNSWLIB::deserializeFromMMap(
            mmap_ptr + offset,
            size - offset
        );
        if (!index) {
            return nullptr;
        }
        return std::make_shared<TensorSimilarityIndex>(
            index,
            original_bit_width,
            static_cast<int>(dimension)
        );
    }

private:
    std::shared_ptr<SimilarityIndex<T> > index_;
    int original_bit_width_;
    int dimension_;
};

typedef TensorSimilarityIndex<TensorType::VectorUInt8> VectorUInt8Index;

#endif //TENSOR_SIMILARITY_INDEX_H
