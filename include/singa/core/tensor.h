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

#include <vector>
#include <tuple>

#include "singa/core/common.h"
#include "singa/core/device.h"
#include "singa/proto/core.pb.h"
#include "singa/utils/logging.h"

using std::vector;
using std::tuple;
namespace singa {

typedef vector<size_t> Shape;
typedef Shape::iterator ShapeIter;
inline size_t Product(const Shape& shape, int start = 0, size_t len = 0) {
  if (len == 0)
    len = shape.size();
  CHECK_LE(len, shape.size());
  size_t v = 1;
  for (unsigned int i = start; i < len; i ++)
    v *= shape[i];
  return v;
}

/// hardcode the width of types defined in DataType
const size_t kDataWidth[] = {sizeof(float), sizeof(float) / 2, sizeof(int),
                          sizeof(char), sizeof(double)};
inline size_t SizeOf(DataType t) {
  static_assert(kNumDataType == sizeof(kDataWidth) / sizeof(size_t),
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
  Tensor();
  explicit Tensor(Shape&& shape, DataType dtype = kFloat32);
  explicit Tensor(const Shape& shape, DataType dtype = kFloat32);
  Tensor(Shape&& shape, Device* dev, DataType dtype = kFloat32);
  Tensor(const Shape& shape, Device* dev, DataType dtype = kFloat32);

  /// Copy Tensor to share the internal data.  No deep copy.
  Tensor(const Tensor& from);
  /// Copy Tensor to share the internal data.  No deep copy.
  Tensor(Tensor&& from);

  /// For functions in xx_math.cc to access the blob.
  /// Users should not operate against Blob directly.
  /// blob_ is allocated in constructors.
  Blob* blob() const {
    return blob_;
  }

  Device* device() const {
    return device_;
  }

  /// Return immutable Tensor values with given type.
  template <typename DType>
  const DType* data() const {
    return static_cast<const DType*> (blob()->data());
  }

  /// data type, including kFloat16, kFloat32, kInt
  const DataType data_type() const {
    return data_type_;
  }

  const Shape& shape() const {
    return shape_;
  }

  int nDim() const {
    return shape_.size();
  }

  bool transpose() const {
    return transpose_;
  }

  /// Return number of total elements
  size_t Size() const {
    return blob_->size() / SizeOf(data_type_);
  }

  /// Return memory size (i.e., Bytes)
  size_t MemSize() const {
    return blob_->size();
  }

  /// Reset the tensor shape, it may reallocate blob, if MemSize() changes.
  void ReShape(const Shape& shape);

  /// Reset the shape, device, and data type as given tensor.
  /// If blob size changes, then reallocate a new blob. The previous blob would
  /// be deleted.
  void ResetLike(const Tensor& t);

  /// Reset the data type, it would reallocate blob if type changes.
  void AsType(DataType type);

  /// Reset the device.
  /// If the target device is a diff device, then do deep data copy.
  void ToDevice(Device* dev);

  /// Equivalent to ToDevice(host_dev).
  void ToHost();

  /// For init the tensor values, copy 'num' elements.
  template<typename DType>
  void CopyDataFromHostPtr(const DType* src, size_t num);

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
  Tensor& operator=(const Tensor& t);

  /// Copy the meta info with data blob shared.
  Tensor& operator=(Tensor&& t);


  Tensor& operator+=(const Tensor& t);
  // void operator+=(Tensor&& t);
  Tensor& operator-=(const Tensor& t);
  // void operator-=(Tensor&& t);
  Tensor& operator*=(const Tensor& t);
  // void operator*=(Tensor&& t);
  Tensor& operator/=(const Tensor& t);
  // void operator/=(Tensor&& t);

  // Scalar operations.

  /// T is a scalar type
  template<typename DType>
  Tensor& operator+=(DType x);

  /// T is a scalar type
  template <typename DType>
  Tensor& operator-=(const DType x);

  /// T is a scalar type
  template <typename DType>
  Tensor& operator*=(const DType x);

  /// T is a scalar type
  template <typename DType>
  Tensor& operator/=(const DType x);

  /// save Tensor into a proto msg
  // void ToProto(TensorProto* t);
  /// load Tensor from proto msg
  // void FromProto(const TensorProto& t);

 protected:
  bool transpose_ = false;
  DataType data_type_ = kFloat32;
  Device* device_ = nullptr;
  /// Note: blob_ is allocated in lazy manner to avoid frequent malloc/free.
  /// If you want to get an allocated Blob, use blob() instead of blob_.
  Blob* blob_ = nullptr;
  Shape shape_;
};

// For tensors with sparse content, e.g., missing columns or rows.
// class SparseTensor : public Tensor {};

/// Copy 'num' elements of src to dst.
/// The first 'src_offset' ('dst_offset') elements will be skipped.
void CopyDataToFrom(Tensor* dst,
              const Tensor& src,
              size_t num,
              size_t src_offset = 0,
              size_t dst_offset = 0);

// ==================Simple Linear Algebra Operations=========================
Tensor Abs(const Tensor& t);
Tensor Exp(const Tensor& t);
Tensor Log(const Tensor& t);
Tensor ReLU(const Tensor& t);
Tensor Sigmoid(const Tensor& t);
Tensor Sign(const Tensor& t);
Tensor Sqrt(const Tensor& t);
Tensor Tanh(const Tensor& t);

/// Regarding the internal data as 2d, with shape_[0]*...*shape_[axis] rows,
/// and shape_[axis+1]*...*shape_[nDim()] columns.
/// and do softmax along each row.
Tensor Softmax(const Tensor& t, int axis = -1);
void Softmax(const Tensor& t, Tensor* ret, int axis = -1);

/// Element-wise opeartion, ret[i]=t[i]^x
template<typename DType>
Tensor Pow(const Tensor& t, DType x);
/// Element-wise opeartion, ret[i]=t[i]^x
template<typename DType>
void Pow(const Tensor& t, DType x, Tensor* ret);
/// Element-wise opeartion, ret[i]=baes[i]^exp[i]
Tensor Pow(const Tensor& base, Tensor exp);
/// Element-wise opeartion, ret[i]=baes[i]^exp[i]
void Pow(const Tensor& base, const Tensor& exp, Tensor* ret);

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
void Add(const Tensor& lhs, const Tensor& rhs, Tensor* ret);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
void Sub(const Tensor& lhs, const Tensor& rhs, Tensor* ret);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
void EltwiseMult(const Tensor& lhs, const Tensor& rhs, Tensor* ret);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);
void Div(const Tensor& lhs, const Tensor& rhs, Tensor* ret);

template <typename DType>
Tensor operator+(const Tensor& t, DType x);
template <typename DType>
void Add(const Tensor& t, DType x, Tensor* ret);

template <typename DType>
Tensor operator-(const Tensor& t, DType x);
template <typename DType>
void Sub(const Tensor& t, DType x, Tensor* ret);

template <typename DType>
Tensor operator*(const Tensor& t, DType x);
template <typename DType>
void EltwiseMult(const Tensor& t, DType x, Tensor* ret);

template <typename DType>
Tensor operator/(const Tensor& t, DType x);
template <typename DType>
void Div(const Tensor& t, DType x, Tensor* ret);

// ================Blas operations============================================
// We fix the scalar argument type to be float.

// ===== Level 1
// TODO(wangwei) make amax/amin/asum a member function of tensor
// void Amax(Tensor, Context* ctx); Get the index of the max value in a vector
// void Asum(Tensor Context* ctx);

// template <typename DType>
// void Axpy(DType x, const Blob& t, Blob* ret, Context* ctx);

/// Do matrix vector multipication or matrix matrix multiplication depdending
/// on the Tensor shape.  ret = lhs * rhs
Tensor Mult(const Tensor& lhs, const Tensor& rhs);
/// Do matrix vector multipication or matrix matrix multiplication depdending
/// on the Tensor shape.  ret = lhs * rhs
void Mult(const Tensor& lhs, const Tensor& rhs, Tensor* ret);

/// Do matrix vector multipication or matrix matrix multiplication depdending
/// on the Tensor shape.  ret = alpha lhs * rhs + beta * ret
Tensor Mult(float alpha, const Tensor& lhs, float beta, const Tensor& rhs);
/// Do matrix vector multipication or matrix matrix multiplication depdending
/// on the Tensor shape. ret = alpha lhs * rhs + beta * ret
void Mult(float alpha, const Tensor& lhs, float beta, const Tensor& rhs,
    Tensor* C);

// ================Random operations==========================================
/// For each element x set x = 1 if random() < p; otherwise x = 1.
void Bernoulli(float p, Tensor* t);
/// Fill in Tensor 't' following uniform distribution.
void Uniform(float low, float high, Tensor* t);
/// Fill in Tensor 't' following Gaussian distribution.
void Gaussian(float mean, float std, Tensor* t);

}  // namespace singa

#endif  // SINGA_CORE_TENSOR_H_
