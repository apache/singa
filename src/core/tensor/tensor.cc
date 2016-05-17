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
#include "singa/core/tensor.h"
#include "./tensor_math.h"
#include "./tensor_math_cpp.h"
#include "./tensor_math_cuda.h"
#include "./tensor_math_opencl.h"
#include <utility>

namespace singa {

Tensor::~Tensor() {
  if (blob_ != nullptr && blob_->DecRefCount() == 0)
    device_->FreeBlob(blob_);
  blob_ = nullptr;
}

Tensor::Tensor() {
  device_ = &hostDeviceSingleton;
}

Tensor::Tensor(const Shape& shape, DataType dtype)
    : data_type_(dtype), device_(&hostDeviceSingleton), shape_(shape) {
  device_ = &hostDeviceSingleton;
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(Shape&& shape, DataType dtype)
    : data_type_(dtype), device_(&hostDeviceSingleton), shape_(shape) {
  device_ = &hostDeviceSingleton;
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(const Shape& shape, Device* device, DataType dtype)
    : data_type_(dtype), device_(device), shape_(shape) {
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(Shape&& shape, Device* device, DataType dtype)
    : data_type_(dtype), device_(device), shape_(shape) {
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(const Tensor& t)
    : transpose_(t.transpose_),
      data_type_(t.data_type_),
      device_(t.device_),
      blob_(t.blob()),
      shape_(t.shape_) {
  blob_->IncRefCount();
}

Tensor::Tensor(Tensor&& t)
    : transpose_(t.transpose_),
      data_type_(t.data_type_),
      device_(t.device_),
      shape_(std::move(t.shape_)) {
  blob_ = t.blob_;
  t.blob_ = nullptr;
}

void Tensor::ResetLike(const Tensor& t) {
  if (blob_ == nullptr || blob_->size() != t.MemSize()) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0) device_->FreeBlob(blob_);
    shape_ = t.shape_;
    device_ = t.device_;
    data_type_ = t.data_type_;
    blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
  }
}

void Tensor::ReShape(const Shape& shape) {
  if (shape_ != shape) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0)
      device_->FreeBlob(blob_);
    blob_ = device_->NewBlob(Product(shape) * SizeOf(data_type_));
    shape_ = shape;
  }
}

void Tensor::AsType(DataType type) {
  if (data_type_ != type) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0)
      device_->FreeBlob(blob_);
    blob_ = device_->NewBlob(Product(shape_) * SizeOf(type));
    data_type_ = type;
  }
}

void Tensor::ToDevice(Device* dst) {
  // TODO(wangwei) the comparison is very strict. May compare against device ID?
  if (device_ != dst) {
    Tensor tmp(shape_, dst, data_type_);
    tmp.CopyData(*this);
    if (blob_ != nullptr && blob_->DecRefCount() == 0)
      device_->FreeBlob(blob_);
    blob_ = tmp.blob_;
    tmp.blob_ = nullptr;
    device_ = dst;
  }
}

void Tensor::ToHost() {
  ToDevice(device_->host());
}

template<typename DType>
void Tensor::CopyDataFromHostPtr(const DType* src, size_t num) {
  CHECK_EQ(sizeof(DType), SizeOf(data_type_)) << "data_type is "
                                              << DataType_Name(data_type_)
                                              << " user given type is of size "
                                              << sizeof(DType);
  if (src != nullptr)
    device_->CopyDataFromHostPtr(blob(), src, sizeof(DType) * num);
  else
    LOG(WARNING) << "Copy data from null host ptr";
}
template void Tensor::CopyDataFromHostPtr(const float* src, size_t num);

void Tensor::CopyData(const Tensor& src) {
  CHECK_EQ(Size(), src.Size());
  CHECK(blob_ != nullptr);
  // Do copy only if the src's blob is already initialized.
  if (src.blob_ != nullptr) {
    singa::CopyData(this, src, Size(), 0, 0);
  }
}

Tensor Tensor::Clone() {
  Tensor t(shape_, device_, data_type_);
  t.transpose_ = transpose_;
  t.CopyData(*this);
  return t;
}

Tensor Tensor::T() const {
  CHECK_EQ(shape_.size(), 2u);
  Tensor t(*this);
  t.transpose_ = ~transpose_;
  std::swap(t.shape_[0], t.shape_[1]);
  return t;
}

Tensor& Tensor::operator=(const Tensor& t) {
  if (blob_ != nullptr && blob_->DecRefCount() == 0)
    device_->FreeBlob(blob_);
  transpose_ = t.transpose_;
  data_type_ = t.data_type_;
  shape_ = t.shape_;
  device_ = t.device_;
  blob_ = t.blob();
  blob_->IncRefCount();
  return *this;
}

Tensor& Tensor::operator=(Tensor&& t) {
  if (blob_ != nullptr && blob_->DecRefCount() == 0)
    device_->FreeBlob(blob_);
  transpose_ = t.transpose_;
  shape_ = std::move(t.shape_);
  device_ = t.device_;
  blob_ = t.blob_;
  t.blob_ = nullptr;
  return *this;
}

#define GenUnaryTensorArgMemberFunction(op, fn) \
  Tensor& Tensor::op(const Tensor& t) { fn(*this, t, this); return *this; }

GenUnaryTensorArgMemberFunction(operator+=, Add);
GenUnaryTensorArgMemberFunction(operator-=, Sub);
GenUnaryTensorArgMemberFunction(operator*=, EltwiseMult);
GenUnaryTensorArgMemberFunction(operator/=, Div);

#define GenUnaryScalarArgMemberFunction(op, fn) \
  template <typename DType>                     \
  Tensor& Tensor::op(DType x) {                 \
    fn(*this, x, this);                         \
    return *this;                               \
  }                                             \
  template Tensor& Tensor::op<float>(float x)

GenUnaryScalarArgMemberFunction(operator-=, Sub);
GenUnaryScalarArgMemberFunction(operator+=, Add);
GenUnaryScalarArgMemberFunction(operator*=, EltwiseMult);
GenUnaryScalarArgMemberFunction(operator/=, Div);

// ====================Tensor Operations=======================================
void CopyData(Tensor* dst,
              const Tensor& src,
              size_t num,
              size_t dst_offset,
              size_t src_offset) {
  CHECK_GE(src.Size(), src_offset + num);
  CHECK_GE(dst->Size(), dst_offset + num);
  auto width = SizeOf(src.data_type());
  CHECK_EQ(width, SizeOf(dst->data_type()));
  CopyRawData(dst, src, num * width, dst_offset * width, src_offset * width);
}

void CopyRawData(Tensor* dst,
              const Tensor& src,
              size_t nBytes,
              size_t dst_offset,
              size_t src_offset) {
  CHECK_GE(src.MemSize(), src_offset + nBytes);
  CHECK_GE(dst->MemSize(), dst_offset + nBytes);
  Device* src_dev = src.device(), *dst_dev = dst->device();
  Blob* src_blob = src.blob(), *dst_blob = dst->blob();
  if (dst_dev->device_lib() != src_dev->device_lib()) {
    // let the none cpp device conduct copy op
    if (dst_dev->device_lib() == kCpp) {
      src_dev->CopyData(dst_blob, *src_blob, nBytes, dst_offset, src_offset);
    } else if (src_dev->device_lib() == kCpp) {
      dst_dev->CopyData(dst_blob, *src_blob, nBytes, dst_offset, src_offset);
    } else {
      LOG(FATAL) << "Not support mem copy betwee Cuda and OpenCL device";
    }
  } else {
    src_dev->CopyData(dst_blob, *src_blob, nBytes, dst_offset, src_offset);
  }
}
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
#define TYPE_LIB_SWITCH(dtype, DType, ltype, Lib, ...)        \
  do {                                                        \
    const int _SwitchShift = 3;                               \
    int _SwitchHash = ((dtype) << _SwitchShift) + (ltype);    \
    switch (_SwitchHash) {                                    \
      case ((kFloat32 << _SwitchShift) + kCuda): {            \
        typedef float DType;                                  \
        typedef lib::Cuda Lib;                                \
        { __VA_ARGS__ }                                       \
        break;                                                \
      }                                                       \
      case ((kFloat32 << _SwitchShift) + kCudnn): {           \
        typedef float DType;                                  \
        typedef lib::Cudnn Lib;                               \
        { __VA_ARGS__ }                                       \
        break;                                                \
      }                                                       \
      case ((kFloat32 << _SwitchShift) + kCpp): {             \
        typedef float DType;                                  \
        typedef lib::Cpp Lib;                                 \
        { __VA_ARGS__ }                                       \
        break;                                                \
      }                                                       \
      case ((kFloat32 << _SwitchShift) + kOpencl): {          \
        typedef float DType;                                  \
        typedef lib::Opencl Lib;                              \
        { __VA_ARGS__ }                                       \
        break;                                                \
      }                                                       \
      default:                                                \
        LOG(FATAL) << "Unknown combination of data type "     \
                   << DataType_Name(dtype) << " and library " \
                   << LibType_Name(ltype);                    \
    }                                                         \
  } while (0)


#define EltwiseUnaryTensorFn(fn, t, ret)                                   \
  do {                                                                     \
    TYPE_LIB_SWITCH(t.data_type(), DType, t.device()->device_lib(), Lib, { \
      ret->device()->Exec(                                               \
          [t, ret](Context* ctx) {                                         \
            fn<DType, Lib>(t.Size(), t.blob(), ret->blob(), ctx);          \
          },                                                               \
          {t.blob()}, {ret->blob()});                                      \
    });                                                                    \
  } while (0)

#define GenUnaryTensorFunction(fn)                    \
  Tensor fn(const Tensor& t) {                        \
    Tensor ret(t.shape(), t.device(), t.data_type()); \
    auto* retptr = &ret;                              \
    EltwiseUnaryTensorFn(fn, t, retptr);              \
    return ret;                                       \
  }

GenUnaryTensorFunction(Abs);
GenUnaryTensorFunction(Exp);
GenUnaryTensorFunction(Log);
GenUnaryTensorFunction(ReLU);
GenUnaryTensorFunction(Sigmoid);
GenUnaryTensorFunction(Sign);
GenUnaryTensorFunction(Sqrt);
GenUnaryTensorFunction(Tanh);

Tensor Softmax(const Tensor& t, int axis) {
  Tensor ret(t.shape(), t.device(), t.data_type());
  Softmax(t, &ret, axis);
  return ret;
}

void Softmax(const Tensor& t, Tensor* ret, int axis) {
  int nrow = 1, ncol = t.Size(), size = ncol;
  CHECK_GE(axis, -1);
  CHECK_GT(t.shape().size(), 0u);
  if (axis > -1) {
    nrow = Product(t.shape(), 0, axis + 1);
    CHECK_EQ(size % nrow, 0) << "Size = " << size << " nrow = " << nrow;
    ncol = size / nrow;
  }
  TYPE_LIB_SWITCH(t.data_type(), DType, t.device()->device_lib(), Lib, {
    ret->device()->Exec(
        [nrow, ncol, t, ret](Context* ctx) {
          Softmax<DType, Lib>(nrow, ncol, t.blob(), ret->blob(), ctx);
        },
        {t.blob()}, {ret->blob()});
    });
}

#define EltwiseBinaryTensorFn(fn, lhs, rhs, ret)                               \
  do {                                                                         \
    TYPE_LIB_SWITCH(lhs.data_type(), DType, lhs.device()->device_lib(), Lib, { \
      CHECK_EQ(sizeof(DType), SizeOf(rhs.data_type()));                        \
      ret->device()->Exec(                                                     \
          [lhs, rhs, ret](Context* ctx) {                                      \
            fn<DType, Lib>(lhs.Size(), lhs.blob(), rhs.blob(), ret->blob(),    \
                           ctx);                                               \
          },                                                                   \
          {lhs.blob(), rhs.blob()}, {ret->blob()});                            \
    });                                                                        \
  } while (0)

#define GenBinaryTensorFunction(op, fn)                        \
  Tensor op(const Tensor& lhs, const Tensor& rhs) {            \
    Tensor ret(lhs.shape(), lhs.device(), lhs.data_type());    \
    fn(lhs, rhs, &ret);                                        \
    return ret;                                                \
  }                                                            \
  void fn(const Tensor& lhs, const Tensor& rhs, Tensor* ret) { \
    EltwiseBinaryTensorFn(fn, lhs, rhs, ret);                  \
  }

GenBinaryTensorFunction(operator+, Add);
GenBinaryTensorFunction(operator-, Sub);
GenBinaryTensorFunction(operator*, EltwiseMult);
GenBinaryTensorFunction(operator/, Div);
GenBinaryTensorFunction(Pow, Pow);

#define EltwiseTensorScalarFn(fn, t, x, ret)                               \
  do {                                                                     \
    TYPE_LIB_SWITCH(t.data_type(), DType, t.device()->device_lib(), Lib, { \
      static_assert(std::is_same<SType, DType>::value,                             \
                    "The Scalar type must match the Tensor data type");    \
      ret->device()->Exec(                                                 \
          [t, x, ret](Context* ctx) {                                      \
            fn<DType, Lib>(t.Size(), t.blob(), x, ret->blob(), ctx);       \
          },                                                               \
          {t.blob()}, {ret->blob()});                                      \
    });                                                                    \
  } while (0)

#define GenTensorScalarFunction(op, fn)                \
  template <typename SType>                            \
  Tensor op(const Tensor& t, SType x) {                \
    Tensor ret(t.shape(), t.device(), t.data_type());  \
    fn(t, x, &ret);                                    \
    return ret;                                        \
  }                                                    \
  template <typename SType>                            \
  void fn(const Tensor& t, SType x, Tensor* ret) {     \
    EltwiseTensorScalarFn(fn, t, x, ret);              \
  }                                                    \
  template Tensor op<float>(const Tensor& t, float x); \
  template void fn<float>(const Tensor& t, const float x, Tensor* ret)

GenTensorScalarFunction(operator+, Add);
GenTensorScalarFunction(operator-, Sub);
GenTensorScalarFunction(operator*, EltwiseMult);
GenTensorScalarFunction(operator/, Div);
GenTensorScalarFunction(Pow, Pow);

// ================Blas operations============================================
template <typename DType>
Tensor Mult(const Tensor& lhs, const Tensor& rhs) {
  Tensor ret(lhs.shape(), lhs.device(), lhs.data_type());
  Mult<DType>(lhs, rhs, &ret);
  return ret;
}
template Tensor Mult<float>(const Tensor& lhs, const Tensor& rhs);

template <typename DType>
void Mult(const Tensor& lhs, const Tensor& rhs, Tensor* ret) {
  Mult(DType(1), lhs, DType(1), rhs, ret);
}
template void Mult<float>(const Tensor& lhs, const Tensor& rhs, Tensor* ret);

template <typename DType>
Tensor Mult(DType alpha, const Tensor& A, DType beta, const Tensor& B) {
  Tensor ret(A.shape(), A.device(), A.data_type());
  Mult<DType>(alpha, A, beta, B, &ret);
  return ret;
}
template Tensor Mult<float>(float alpha, const Tensor& lhs, float beta,
    const Tensor& rhs);

template <typename SType>
void Mult(SType alpha, const Tensor& A, SType beta, const Tensor& B, Tensor* C)
{
  CHECK_EQ(A.shape().size(), 2u);
  bool transA = A.transpose();
  size_t m = transA ? A.shape()[1] : A.shape()[0], n = 0;
  if (B.shape().size() == 1u) {
    n = C->Size();
    TYPE_LIB_SWITCH(A.data_type(), DType, A.device()->device_lib(), Lib, {
      static_assert(std::is_same<SType, DType>::value,
        "The scalar type must be the same as the tensor data type");
      C->device()->Exec(
        [transA, m, n, alpha, A, beta, B, C](Context* ctx) {
        GEMV<DType, Lib>(transA, m, n, alpha, A.blob(),
          B.blob(), beta, C->blob(), ctx);
        },
        {A.blob(), B.blob()}, {C->blob()});
      });
  } else {
    CHECK(!C->transpose());
    bool transB = B.transpose();
    size_t k = transB ? B.shape()[1] : B.shape()[0];
    n = C->shape()[1];
    CHECK_EQ(C->shape()[0], m);
    CHECK_EQ(A.Size(), m * k);
    CHECK_EQ(B.Size(), n * k);
    TYPE_LIB_SWITCH(A.data_type(), DType, A.device()->device_lib(), Lib, {
        static_assert(std::is_same<SType, DType>::value,
          "The scalar type must be the same as the tensor data type");
        C->device()->Exec(
          [transA, transB, m, n, k, alpha, A, beta, B, C](Context* ctx) {
          GEMM<DType, Lib>(transA, transB, m, n, k, alpha, A.blob(),
            B.blob(), beta, C->blob(), ctx);
          },
          {A.blob(), B.blob()}, {C->blob()});
        });
  }
}
template void Mult<float>(float alpha, const Tensor& lhs, float beta,
    const Tensor& rhs, Tensor* ret);


// ================Neural Net operations======================================
/*
void Conv(const OpConf* conf, const Tensor& input, const Tensor& W,
          const Tensor& b, Tensor* ret) {
  TYPE_LIB_SWITCH(input.data_type(), DType, input.device()->nn_lib(), Lib, {
    ret->device()->Exec(
        [conf, input, W, b, ret](Context* ctx) {
          Conv<DType, Lib>(conf, input.blob(), W.blob(), b.blob(), ret->blob(),
                           ctx);
        },
        {input.blob(), W.blob(), b.blob()}, {ret->blob()});
  });
}
*/
void Bernoulli(float p, Tensor* t) {
  TYPE_LIB_SWITCH(t->data_type(), DType, t->device()->nn_lib(), Lib, {
    t->device()->Exec(
        [p, t](Context* ctx) {
          Bernoulli<DType, Lib>(t->Size(), p, t->blob(), ctx);
        },
        {}, {t->blob()}, true);
  });
}

void Uniform(float low, float high, Tensor* t) {
  TYPE_LIB_SWITCH(t->data_type(), DType, t->device()->nn_lib(), Lib, {
    t->device()->Exec(
        [low, high, t](Context* ctx) {
          Uniform<DType, Lib>(t->Size(), low, high, t->blob(), ctx);
        },
        {}, {t->blob()}, true);
  });
}

void Gaussian(float mean, float std, Tensor* t) {
  TYPE_LIB_SWITCH(t->data_type(), DType, t->device()->nn_lib(), Lib, {
    t->device()->Exec(
        [mean, std, t](Context* ctx) {
          Gaussian<DType, Lib>(t->Size(), mean, std, t->blob(), ctx);
        },
        {}, {t->blob()}, true);
  });
}
}  // namespace singa
