/**
 * @file tensor.h
 * This file provides the definition of Vector and Matrix struct.
 * @note Similar to model.h, we define the detailed logic in C++ and then wrap
 * it with C interface, such that we can use the C interface in the PostgreSQL
 * extension.
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>

#ifdef __cplusplus
/*********************************** C++ ***********************************/

#include <Eigen/Core>
#include <memory>

namespace TensorType {
    using VectorFloat64 = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using MatrixFloat64 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorFloat32 = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    using MatrixFloat32 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorFloat16 = Eigen::Matrix<Eigen::half, Eigen::Dynamic, 1>;
    using MatrixFloat16 = Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorUInt32 = Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>;
    using MatrixUInt32 = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorBoolean = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
    using MatrixInt32 = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorInt32 = Eigen::Matrix<int32_t, Eigen::Dynamic, 1>;
    using MatrixInt16 = Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixUInt16 = Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorInt16 = Eigen::Matrix<int16_t, Eigen::Dynamic, 1>;
    using VectorUInt16 = Eigen::Matrix<uint16_t, Eigen::Dynamic, 1>;
    using MatrixInt8 = Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixUInt8 = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorInt8 = Eigen::Matrix<int8_t, Eigen::Dynamic, 1>;
    using VectorUInt8 = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;
}

class TensorF64 {
public:
    explicit TensorF64(TensorType::VectorFloat64 tensor, const std::vector<int> &shape);

    ~TensorF64() = default;

    const TensorType::VectorFloat64 &getTensor() const;

    const std::vector<int> &getShape() const;

    void setShape(const std::vector<int> &shape);

    int getDimension() const;

    double getValue(int index) const;

private:
    TensorType::VectorFloat64 tensor_;
    std::vector<int> shape_;
};

class TensorF16 {
public:
    explicit TensorF16(TensorType::VectorFloat16 tensor, const std::vector<int> &shape);

    ~TensorF16() = default;

    const TensorType::VectorFloat16 &getTensor() const;

    const std::vector<int> &getShape() const;

    void setShape(const std::vector<int> &shape);

    int getDimension() const;

    double getValue(int index) const;

private:
    TensorType::VectorFloat16 tensor_;
    std::vector<int> shape_;
};


/**
 * TensorPacket is a package of Tensor, which is serialized and can be stored
 * in any persistent storage like such as database, file system, and etc.
 */
class TensorPacket {
public:
    virtual ~TensorPacket() = default;

    /**
     * Get the size of the packet
     * @return The size of the packet in bytes
     */
    virtual int64_t size() const = 0;

    /**
    * Unpack the packet into a vector
    * @return decompressed vector
    */
    virtual TensorF64 toTensor64() const = 0;

    virtual TensorF16 toTensor16() const = 0;

    /**
    * Serialize the packet into a byte array
    * @return serialized byte array
    */
    virtual std::vector<uint8_t> serialize() const = 0;

    /**
     * Set the base tensor id (for delta packet)
     * @param base_tensor_id the base tensor id
     */
    virtual void setBaseTensorId(int64_t base_tensor_id) = 0;

    virtual void setIndexId(int index_id) = 0;

    /**
     * Get the base tensor id (for delta packet)
     * @return the id of the base tensor
     */
    virtual int64_t getBaseTensorId() const = 0;

    /**
     * Get the shape of the tensor
     * @return the shape of the tensor
     */
    virtual std::vector<int> getShape() const = 0;

    virtual int getDimension() const = 0;

    virtual int getIndexId() const = 0;
};

class IntTensorPacket final : public TensorPacket {
public:
    enum IntType {
        INT8,
        INT16,
        INT32,
        UINT8,
        UINT16,
        UINT32
    };

    IntTensorPacket(
        IntType int_type,
        const std::vector<int> &shape,
        const std::vector<uint8_t> &serialized_data
    );

    IntTensorPacket(
        IntType int_type,
        const std::shared_ptr<TensorF64>& tensor
    );

    ~IntTensorPacket() override = default;

    int64_t size() const override;

    TensorF64 toTensor64() const override;

    TensorF16 toTensor16() const override;

    std::vector<uint8_t> serialize() const override;

    void setBaseTensorId(int64_t base_tensor_id) override;

    int64_t getBaseTensorId() const override;

    std::vector<int> getShape() const override;

    int getDimension() const override;

    void setIndexId(int index_id) override;

    int getIndexId() const override;

    static std::shared_ptr<TensorPacket> deserialize(const std::vector<uint8_t> &data);

    static std::shared_ptr<TensorPacket> deserialize(const uint8_t *data, size_t size);

    IntType getIntType() const;

    static int getIntTypeInBytes(IntTensorPacket::IntType int_type);

private:
    IntType int_type_;
    std::vector<int> shape_;
    int64_t nelements_;
    std::vector<uint8_t> serialized_data_;
};

struct UINT8QuantizedTensorPacket {
    TensorType::VectorUInt8 data;
    double scale;
    double zero_point;
    int full_quantized_bit_width;
};

extern "C" {
#endif
/************************************ C ************************************/
typedef struct TensorF64 Tensor64C; // use 'struct' keyword in C

typedef struct TensorPacket TensorPacketC;

int64_t tpt_size(TensorPacketC *tensor_packet);

char *tpt_serialize(TensorPacketC *tensor_packet);

#ifdef __cplusplus
}
#endif
#endif //TENSOR_H
