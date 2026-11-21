#ifndef DELTA_COMPRESS_H
#define DELTA_COMPRESS_H

#include <memory>

#include "neurstore/utils/tensor.h"

/**
 * Interface for a delta compression method
 */
class DeltaCompress {
public:
    explicit DeltaCompress(const double tolerance): tolerance_(tolerance) {
    }

    virtual ~DeltaCompress() = default;

    /**
     * @brief Compress a tensor
     * @param base_tensor The base tensor to calculate the delta
     * @param tensor The tensor to compress
     * @param shape The shape of the tensor
     * @return
     */
    virtual std::shared_ptr<TensorPacket> compress(
        const TensorType::VectorFloat64 &base_tensor,
        const TensorType::VectorFloat64 &tensor,
        const std::vector<int> &shape
    ) = 0;

    /**
     * @brief Decompress a tensor from a packet
     * @param base_tensor The base tensor to calculate the delta
     * @param t_packet The tensor packet to decompress
     * @return
     */
    virtual std::shared_ptr<TensorF64> decompress(
        const TensorType::VectorFloat64 &base_tensor,
        const std::shared_ptr<TensorPacket> &t_packet
    ) = 0;

    virtual std::shared_ptr<TensorF16> decompressF16(
        const TensorType::VectorFloat16 &base_tensor,
        const std::shared_ptr<TensorPacket> &t_packet
    ) = 0;

protected:
    double tolerance_;
};

#endif //DELTA_COMPRESS_H
