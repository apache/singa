#ifndef SIMILAR_INDEX_H
#define SIMILAR_INDEX_H

#include "neurstore/utils/tensor.h"


/**
 * Similarity index interface (e.g., to extend HNSW, IVF, etc.)
 */
template<typename T>
class SimilarityIndex {
public:
    explicit SimilarityIndex(const int dimension): dimension_(dimension) {
    }

    virtual ~SimilarityIndex() = default;

    /**
     * Insert a vector into the index
     * @param vector the vector to insert
     * @param scale the scale of the vector
     * @param zero_point the zero point of the vector
     * @return the id of the inserted vector
     */
    virtual int64_t insert(const T &vector, double scale, double zero_point) = 0;

    /**
     * Insert a list of vectors into the index
     * @param vectors the vectors to insert
     * @param scales the scales of the vectors
     * @param zero_points the zero points of the vectors
     * @return the ids of the inserted vectors
     */
    virtual std::vector<int64_t> insertMany(
        std::vector<T> &vectors,
        std::vector<double> &scales,
        std::vector<double> &zero_points
    ) = 0;

    /**
     * Query the index for the k most similar vectors
     * @param vector The vector to query
     * @param k k-nearest neighbors
     * @param scale the scale of the vector
     * @param zero_point the zero point of the vector
     * @return the ids of the k most similar vectors
     */
    virtual std::vector<int64_t> query(const T &vector, int k, double scale, double zero_point) const = 0;

    /**
     * Retrieve a vector from the index by id
     * @param id the id of the vector to retrieve
     * @return the retrieved vector
     */
    virtual TensorType::VectorFloat64 retrieve(int64_t id) const = 0;

    virtual TensorType::VectorFloat16 retrieveF16(int64_t id) const = 0;

    virtual UINT8QuantizedTensorPacket retrieveUINT8Quantized(int64_t id) const = 0;

    /**
     * Get the number of vectors in the index
     * @return number of vectors
     */
    virtual int64_t size() const = 0;

    /**
     * Get the space used by the index
     * @return space used in bytes
     */
    virtual int64_t space() const = 0;

    /**
     * Serialize the index to a byte array
     */
    virtual void serialize(std::ostream &out) const = 0;

protected:
    int64_t dimension_;
};

#endif //SIMILAR_INDEX_H
