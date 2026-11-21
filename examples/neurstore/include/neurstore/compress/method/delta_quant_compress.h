#ifndef DELTA_QUANT_COMPRESS_H
#define DELTA_QUANT_COMPRESS_H

#include "neurstore/compress/method/delta_compress.h"

class DeltaQuantPacket final : public TensorPacket {
public:
    /**
     * original tensor = zero_point + delta_tensor * scale,
     * quantized from original_bit_width to quantized_bit_width
     * @param delta_tensor
     * @param scale
     * @param zero_point
     * @param original_bit_width
     * @param quantized_bit_width
     * @param shape
     */
    DeltaQuantPacket(
        const TensorType::VectorUInt32 &delta_tensor,
        double scale,
        double zero_point,
        int original_bit_width,
        int quantized_bit_width,
        std::vector<int> shape
    );

    ~DeltaQuantPacket() override = default;

    /**
     * Get the size of the packet
     * @return the size of the packet in bytes
     */
    int64_t size() const override;

    /**
     * Unpack the packet into a tensor
     * @return decompressed vector
     */
    TensorF64 toTensor64() const override;

    TensorF16 toTensor16() const override;

    UINT8QuantizedTensorPacket toUINT8QuantizedTensorPacket() const;

    /**
     * Serialize the packet into a byte array
     * @return serialized byte array
     */
    std::vector<uint8_t> serialize() const override;

    /**
     * Set the base tensor id
     * @param base_tensor_id the base tensor id
     */
    void setBaseTensorId(int64_t base_tensor_id) override;

    /**
     * Get the base tensor id
     * @return the id of the base tensor
     */
    int64_t getBaseTensorId() const override;

    /**
     * Get the shape of the tensor
     * @return the shape of the tensor
     */
    std::vector<int> getShape() const override;

    int getDimension() const override;

    void setIndexId(int index_id) override;

    int getIndexId() const override;

    /**
     * Load the packet from a byte array (static method)
     * @param data The byte array to load
     * @return DeltaQuantPacket (TensorPacket)
     */
    static std::shared_ptr<TensorPacket> deserialize(const std::vector<uint8_t> &data);

    static std::shared_ptr<TensorPacket> deserialize(const uint8_t *data, size_t size);

private:
    double scale_;
    double zero_point_;
    int original_bit_width_;
    int quantized_bit_width_;
    int64_t base_tensor_id_; // id of the base tensor
    std::vector<int> shape_;
    int64_t nbytes_; // size of the packet in bytes
    int64_t nelements_; // number of elements in the tensor
    std::vector<uint8_t> serialized_data_; // serialized delta tensor
    int64_t index_id_;
};


class DeltaQuantCompress final : public DeltaCompress {
public:
    explicit DeltaQuantCompress(
        double tolerance,
        bool dynamic = true,
        int default_quantized_bit_width = -1
    );

    /**
     * @brief Compress a tensor
     * @param base_tensor the base tensor to calculate the delta
     * @param tensor the tensor to compress
     * @param shape the shape of the tensor
     * @return DeltaQuantPacket (subclass of TensorPacket)
     */
    std::shared_ptr<TensorPacket> compress(
        const TensorType::VectorFloat64 &base_tensor,
        const TensorType::VectorFloat64 &tensor,
        const std::vector<int> &shape
    ) override;

    /**
     * @brief Decompress a tensor from a packet
     * @param base_tensor the base tensor to calculate the delta
     * @param t_packet the tensor packet to decompress
     * @return decompressed vector
     */
    std::shared_ptr<TensorF64> decompress(
        const TensorType::VectorFloat64 &base_tensor,
        const std::shared_ptr<TensorPacket> &t_packet
    ) override;

    std::shared_ptr<TensorF16> decompressF16(
        const TensorType::VectorFloat16 &base_tensor,
        const std::shared_ptr<TensorPacket> &t_packet
    ) override;

private:
    bool dynamic_; // dynamic quantized bit width if true
    int default_quantized_bit_width_; // default quantized bit width
};

#endif //DELTA_QUANT_COMPRESS_H
