/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SINGA_CORE_TENSOR_H_
#define SINGA_CORE_TENSOR_H_

#include <glog/logging.h>
#include <vector>

#include "singa/core/common.h"
#include "singa/core/device.h"
#include "singa/core/math.h"
#include "singa/proto/core.pb.h"

using std::vector;
namespace singa {

typedef vector<int> Shape;
inline int Product(Shape shape) {
  if (shape.size() == 0)
    return 0;
  int v = 1;
  for (auto s : shape)
    v *= s;
  return v;
}

/// hardcode the width of types defined in DataType
const int kDataWidth[] = {4, 2, 4, 1};
inline int SizeOf(DataType t) {
  static_assert(kNumDataType == sizeof(kDataWidth) / sizeof(int),
      "Num of data types not match num of data width");
  CHECK_GT(kNumDataType, t);
  return kDataWidth[t];
}

/// A Tensor instance is a multi-dimensional array resident on a Device
/// (default device is the host CPU). The internal data is allocated in lazy
/// manner.
/// Linear algebra, neural net and random operations are provided against
/// Tensor.
/// For all operations, if the result tensor is passed as an argument,
/// then it must be set up correctly (shape, device). Otherwise, runtime error
/// like SegmentFault would happen. Simply type/device check would be conducted.
class Tensor {
 public:
  ~Tensor();
  Tensor() = default;
  explicit Tensor(const Shape& shape, DataType dtype = kFloat32);
  Tensor(const Shape& shape, Device* dev, DataType dtype = kFloat32);

  /// Copy Tensor to share the internal data.  No deep copy.
  Tensor(const Tensor& from);

  /// Copy Tensor to share the internal data.  No deep copy.
  Tensor(Tensor&& from);

  /// For functions in xx_math.cc to access the blob.
  /// Users should not operate against Blob directly.
  /// It will malloc memory for the tensor if not allocated before.
  Blob* blob() const {
    return blob_;
  }

  Device* device() const {
    return device_;
  }

  /// Return immutable Tensor values with given type.
  template <typename T>
  const T* data() {
    return static_cast<const T*> (blob()->data());
  }

  /// data type, including kFloat16, kFloat32, kInt
  const DataType data_type() const {
    return data_type_;
  }

  const Shape& shape() const {
    return shape_;
  }

  bool transpose() const {
    return transpose_;
  }

  int Size() const {
    return blob_->size() / SizeOf(data_type_);
  }

  int MemSize() const {
    return blob_->size();
  }

  void ReShape(const Shape& shape);

  void AsType(DataType type);

  /// Reset the device.
  /// If the target device is a diff device, then do deep data copy.
  void ToDevice(Device* dev);

  /// Equivalent to ToDevice(host_dev).
  void ToHost();

  /// For init the tensor values, copy 'size' bytes data.
  void CopyDataFromHostPtr(const void* src, size_t size);

  /// Copy data from another Tensor which may be on a diff device.
  /// Meta data would not be copied!
  void CopyData(const Tensor& other);

  /// Return an exactly the same Tensor with data been deep copied.
  Tensor Clone();

  // Tensor operations

  /// Matrix transpose.  Valid only if shape.size() == 2.
  /// No data copy, just set the transpose_ filed of the returned tensor.
  Tensor T() const;

  /// Copy the meta info with data blob shared.
  void operator=(const Tensor& t);

  /// Copy the meta info with data blob shared.
  void operator=(Tensor&& t);

  void operator+=(const Tensor& t);
  void operator+=(Tensor&& t);
  void operator-=(const Tensor& t);
  void operator-=(Tensor&& t);
  void operator*=(const Tensor& t);
  void operator*=(Tensor&& t);
  void operator/=(const Tensor& t);
  void operator/=(Tensor&& t);

  // Scalar operations.

  /// T is a scalar type
  template <typename T>
  void operator+=(const T x);

  /*
  /// T is a scalar type
  template <typename T>
  void operator-=(const T x);

  /// T is a scalar type
  template <typename T>
  void operator*=(const T x);

  /// T is a scalar type
  template <typename T>
  void operator/=(const T x);

  void Log(int base = 2);
  void Tanh();
  void Sigmoid();
  void ReLU();

  // random functions.
  void Uniform(float low, float high);
  template <typename T>
  void Gaussian(float mean, float std);

  /// save Tensor into a proto msg
  void ToProto(TensorProto* t);
  /// load Tensor from proto msg
  void FromProto(const TensorProto& t);
  */
 protected:
  bool transpose_ = false;
  DataType data_type_ = kFloat32;
  Device* device_ = nullptr;
  /// Note: blob_ is allocated in lazy manner to avoid frequent malloc/free.
  /// If you want to get an allocated Blob, use blob() instead of blob_.
  Blob* blob_ = nullptr;
  Shape shape_;
};

/// For tensors with sparse content, e.g., missing columns or rows.
// class SparseTensor : public Tensor {};

// ==================Simple Linear Algebra Operations=========================

/*
Tensor Tanh(const Tensor& t);
Tensor Log(const Tensor& t);
Tensor Sigmoid(const Tensor& t);
Tensor ReLU(const Tensor& t);
Tensor Softmax(const Tensor& t);
*/
void CopyData(Tensor* dst,
              const Tensor& src,
              int msize,
              int src_offset = 0,
              int dst_offset = 0);

// element-wise ops

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
void Add(const Tensor& lhs, const Tensor& rhs, Tensor* ret);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
void Sub(const Tensor& lhs, const Tensor& rhs, Tensor* ret);
/*
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
void operator*(const Tensor& lhs, const Tensor& rhs, Tensor* ret);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);
void operator/(const Tensor& lhs, const Tensor& rhs, Tensor* ret);

template <typename T>
Tensor operator+(const T x, const Tensor& t);
template <typename T>
void operator+(const T x, const Tensor& t, Tensor* ret);

template <typename T>
Tensor operator-(const T x, const Tensor& t);
template <typename T>
void operator-(const T x, const Tensor& t, Tensor* ret);

template <typename T>
Tensor operator*(const T x, const Tensor& t);
template <typename T>
void operator*(const T x, const Tensor& t, Tensor* ret);

template <typename T>
Tensor operator/(const T x, const Tensor& t);
template <typename T>
void operator/(const T x, const Tensor& t, Tensor* ret);

//================Blas operations============================================
Tensor Mult(const Tensor& lhs, const Tensor& rhs);
void Mult(const Tensor& lhs, const Tensor& rhs, Tensor* ret);

tempalte<typename T> T Dot(const Tensor& lhs, const Tensor& rhs);

//================Neural Net operations======================================

/// Convolution Op. 'Conf' is ConvConf;
void Conv(const OpConf* conf,
          const Tensor& input,
          const Tensor& W,
          const Tensor &b,
          Tensor* ret);


//================Random operations==========================================
Tensor Uniform(float low, float high, const Shape& shape, Device* dev);

Tensor Gaussian(float mean, float std, const Shape& shape, Device* dev);
*/
//============================================================================
/// typedef DType accroding to type value.
/// DType would be used in the code block __VA_ARGS__.
#define TYPE_SWITCH(type, DType, ...)                               \
  do {                                                              \
    switch (type) {                                                 \
      case kFloat32: {                                              \
        typedef float DType;                                        \
        { __VA_ARGS__ }                                             \
        break;                                                      \
      }                                                             \
      case kInt: {                                                  \
        typedef int DType;                                          \
        { __VA_ARGS__ }                                             \
        break;                                                      \
      }                                                             \
      case kChar: {                                                 \
        typedef char DType;                                         \
        { __VA_ARGS__ }                                             \
        break;                                                      \
      }                                                             \
      default:                                                      \
        LOG(FATAL) << "Unknow data type = " << DataType_Name(type); \
    }                                                               \
  } while (0)

/// typedef DType and Lib according to values of type and lib respectively.
/// type is from DataType, and lib is from LibType.
/// DType and Lib would be used in __VA_ARGS__.
#define TYPE_LIB_SWITCH(dtype, DType, ltype, Lib, ...)                 \
  do {                                                               \
    const int _SwitchShift = 3;                                      \
    int _SwitchHash = ((dtype) << _SwitchShift) + (ltype);                 \
    switch (_SwitchHash) {                                           \
      case ((kFloat32 << _SwitchShift) + kCuda): {                   \
        typedef float DType;                                          \
        typedef lib::Cuda Lib;                                            \
        { __VA_ARGS__ }                                              \
        break;                                                       \
      }                                                              \
      case ((kFloat32 << _SwitchShift) + kCudnn): {                  \
        typedef float DType;                                          \
        typedef lib::Cudnn Lib;                                           \
        { __VA_ARGS__ }                                              \
        break;                                                       \
      }                                                              \
      case ((kFloat32 << _SwitchShift) + kCpp): {                    \
        typedef float DType;                                          \
        typedef lib::Cpp Lib;                                             \
        { __VA_ARGS__ }                                              \
        break;                                                       \
      }                                                              \
      case ((kFloat32 << _SwitchShift) + kOpencl): {                \
        typedef float DType;                                          \
        typedef lib::Opencl Lib;                                          \
        { __VA_ARGS__ }                                              \
        break;                                                       \
      }                                                              \
      default:                                                       \
        LOG(FATAL) << "Unknown combination of data type "            \
                   << DataType_Name(dtype) << " and library "        \
                   << LibType_Name(ltype);                             \
    }                                                                \
  } while (0)



}  // namespace singa

#endif  // SINGA_CORE_TENSOR_H_
