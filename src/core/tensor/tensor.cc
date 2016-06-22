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

Tensor::Tensor() { device_ = defaultDevice; }

Tensor::Tensor(const Shape &shape, DataType dtype)
    : data_type_(dtype), device_(defaultDevice), shape_(shape) {
  device_ = defaultDevice;
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(Shape &&shape, DataType dtype)
    : data_type_(dtype), device_(defaultDevice), shape_(shape) {
  device_ = defaultDevice;
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(const Shape &shape, std::shared_ptr<Device> device, DataType dtype)
    : data_type_(dtype), device_(device), shape_(shape) {
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(Shape &&shape, std::shared_ptr<Device> device, DataType dtype)
    : data_type_(dtype), device_(device), shape_(shape) {
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(const Tensor &t)
    : transpose_(t.transpose_), data_type_(t.data_type_), device_(t.device_),
      blob_(t.blob()), shape_(t.shape_) {
  blob_->IncRefCount();
  // LOG(ERROR) << "const&";
}

Tensor::Tensor(Tensor &&t)
    : transpose_(t.transpose_), data_type_(t.data_type_), device_(t.device_),
      shape_(std::move(t.shape_)) {
  blob_ = t.blob_;
  t.blob_ = nullptr;
  // LOG(ERROR) << "&&";
}

void Tensor::ResetLike(const Tensor &t) {
  if (blob_ == nullptr || device_ != t.device_ || MemSize() != t.MemSize()) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0)
      device_->FreeBlob(blob_);
    shape_ = t.shape_;
    device_ = t.device_;
    data_type_ = t.data_type_;
    blob_ = device_->NewBlob(t.MemSize());
  }
}

void Tensor::Reshape(const Shape &shape) {
  if (Product(shape_) != Product(shape)) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0)
      device_->FreeBlob(blob_);
    blob_ = device_->NewBlob(Product(shape) * SizeOf(data_type_));
  }
  shape_ = shape;
}

void Tensor::Reshape(Shape &&shape) {
  if (Product(shape_) != Product(shape)) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0)
      device_->FreeBlob(blob_);
    blob_ = device_->NewBlob(Product(shape) * SizeOf(data_type_));
  }
  shape_ = std::move(shape);
}

void Tensor::AsType(DataType type) {
  if (data_type_ != type) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0)
      device_->FreeBlob(blob_);
    blob_ = device_->NewBlob(Product(shape_) * SizeOf(type));
    data_type_ = type;
  }
}

void Tensor::ToDevice(std::shared_ptr<Device> dst) {
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

void Tensor::ToHost() { ToDevice(device_->host()); }

template <typename DType>
void Tensor::CopyDataFromHostPtr(const DType *src, size_t num) {
  CHECK_EQ(sizeof(DType), SizeOf(data_type_))
      << "data_type is " << DataType_Name(data_type_)
      << " user given type is of size " << sizeof(DType);
  if (src != nullptr) {
    device_->CopyDataFromHostPtr(blob(), src, sizeof(DType) * num, 0);
  } else {
    LOG(WARNING) << "Copy data from null host ptr";
  }
}
template void Tensor::CopyDataFromHostPtr(const float *src, size_t num);
template void Tensor::CopyDataFromHostPtr(const int *src, size_t num);

void Tensor::CopyData(const Tensor &src) {
  CHECK_EQ(Size(), src.Size());
  CHECK(blob_ != nullptr);
  // Do copy only if the src's blob is already initialized.
  if (src.blob_ != nullptr) {
    singa::CopyDataToFrom(this, src, Size(), 0, 0);
  }
}

Tensor Tensor::Clone() const {
  Tensor t(shape_, device_, data_type_);
  t.transpose_ = transpose_;
  t.CopyData(*this);
  return t;
}

Tensor Tensor::T() const {
  CHECK_EQ(shape_.size(), 2u);
  Tensor t;
  t.device_ = device_;
  t.data_type_ = data_type_;
  t.transpose_ = ~transpose_;
  t.shape_.push_back(shape_[1]);
  t.shape_.push_back(shape_[0]);
  t.blob_ = blob_;
  blob_->IncRefCount();
  return t;
}

Tensor &Tensor::operator=(const Tensor &t) {
  // LOG(ERROR) << "= const &";
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

Tensor &Tensor::operator=(Tensor &&t) {
  // LOG(ERROR) << "= &&";
  if (blob_ != nullptr && blob_->DecRefCount() == 0)
    device_->FreeBlob(blob_);
  transpose_ = t.transpose_;
  data_type_ = t.data_type_;
  shape_ = std::move(t.shape_);
  device_ = t.device_;
  blob_ = t.blob_;
  t.blob_ = nullptr;
  return *this;
}

Tensor Reshape(const Tensor &in, const Shape &s) {
  Tensor out(in);
  out.Reshape(s);
  return out;
}

Tensor Reshape(const Tensor &in, Shape &&s) {
  Tensor out(in);
  out.Reshape(std::move(s));
  return out;
}

#define GenUnaryTensorArgMemberFn(op, fn)                                \
  Tensor &Tensor::op(const Tensor &t) {                                        \
    fn(*this, t, this);                                                        \
    return *this;                                                              \
  }

GenUnaryTensorArgMemberFn(operator+=, Add);
GenUnaryTensorArgMemberFn(operator-=, Sub);
GenUnaryTensorArgMemberFn(operator*=, EltwiseMult);
GenUnaryTensorArgMemberFn(operator/=, Div);

#define GenUnaryScalarArgMemberFn(op, fn)                                \
  template <typename DType> Tensor &Tensor::op(DType x) {                      \
    fn(*this, x, this);                                                        \
    return *this;                                                              \
  }                                                                            \
  template Tensor &Tensor::op<float>(float x)

GenUnaryScalarArgMemberFn(operator-=, Sub);
GenUnaryScalarArgMemberFn(operator+=, Add);
GenUnaryScalarArgMemberFn(operator*=, EltwiseMult);
GenUnaryScalarArgMemberFn(operator/=, Div);

// ====================Tensor Operations=======================================
void CopyDataToFrom(Tensor *dst, const Tensor &src, size_t num,
                    size_t dst_offset, size_t src_offset) {
  auto width = SizeOf(src.data_type());
  CHECK_EQ(width, SizeOf(dst->data_type()));
  size_t nBytes = num * width;
  dst_offset *= width;
  src_offset *= width;
  CHECK_GE(src.MemSize(), src_offset + nBytes);
  CHECK_GE(dst->MemSize(), dst_offset + nBytes);

  std::shared_ptr<Device> src_dev = src.device(), dst_dev = dst->device();
  Blob *from = src.blob(), *to = dst->blob();
  if (dst_dev->lang() != src_dev->lang()) {
    // let the none cpp device conduct copy op
    if (dst_dev->lang() == kCpp) {
      src_dev->CopyDataToFrom(to, from, nBytes, kDeviceToHost, dst_offset,
                              src_offset);
    } else if (src_dev->lang() == kCpp) {
      dst_dev->CopyDataToFrom(to, from, nBytes, kHostToDevice, dst_offset,
                              src_offset);
    } else {
      LOG(FATAL) << "Not support mem copy betwee Cuda and OpenCL device";
    }
  } else {
    auto direct = src_dev->lang() == kCpp ? kHostToHost : kDeviceToDevice;
    src_dev->CopyDataToFrom(to, from, nBytes, direct, dst_offset, src_offset);
  }
}
//============================================================================
/// typedef DType accroding to type value.
/// DType would be used in the code block __VA_ARGS__.
#define TYPE_SWITCH(type, DType, ...)                                          \
  do {                                                                         \
    switch (type) {                                                            \
    case kFloat32: {                                                           \
      typedef float DType;                                                     \
      { __VA_ARGS__ }                                                          \
      break;                                                                   \
    }                                                                          \
    case kInt: {                                                               \
      typedef int DType;                                                       \
      { __VA_ARGS__ }                                                          \
      break;                                                                   \
    }                                                                          \
    case kChar: {                                                              \
      typedef char DType;                                                      \
      { __VA_ARGS__ }                                                          \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      LOG(FATAL) << "Unknow data type = " << DataType_Name(type);              \
    }                                                                          \
  } while (0)

/// typedef DType and Lang according to data type and device programming
/// language respectively.
/// type is from DataType, and lang is from LangType.
/// DType and Lang would be used in __VA_ARGS__.
#define TYPE_LANG_SWITCH(dtype, DType, ltype, Lang, ...)                       \
  do {                                                                         \
    const int _SwitchShift = 3;                                                \
    int _SwitchHash = ((dtype) << _SwitchShift) + (ltype);                     \
    switch (_SwitchHash) {                                                     \
    case ((kFloat32 << _SwitchShift) + kCuda): {                               \
      typedef float DType;                                                     \
      typedef lang::Cuda Lang;                                                 \
      { __VA_ARGS__ }                                                          \
      break;                                                                   \
    }                                                                          \
    case ((kFloat32 << _SwitchShift) + kCpp): {                                \
      typedef float DType;                                                     \
      typedef lang::Cpp Lang;                                                  \
      { __VA_ARGS__ }                                                          \
      break;                                                                   \
    }                                                                          \
    case ((kFloat32 << _SwitchShift) + kOpencl): {                             \
      typedef float DType;                                                     \
      typedef lang::Opencl Lang;                                               \
      { __VA_ARGS__ }                                                          \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      LOG(FATAL) << "Unknown combination of data type "                        \
                 << DataType_Name(dtype) << " and language "                   \
                 << LangType_Name(ltype);                                      \
    }                                                                          \
  } while (0)

template <typename SType> void Tensor::SetValue(const SType x) {
  CHECK_EQ(sizeof(SType), SizeOf(data_type_));
  auto size = Size();
  auto ptr = blob_;
  TYPE_LANG_SWITCH(data_type_, DType, device_->lang(), Lang, {
    // cast x to DType
    device_->Exec(
        [size, x, ptr](Context *ctx) { Set<DType, Lang>(size, x, ptr, ctx); },
        {}, {ptr});
  });
}
template void Tensor::SetValue<float>(const float x);

#define EltwiseUnaryTensorFn(fn, t, ret)                               \
  do {                                                                 \
    TYPE_LANG_SWITCH(t.data_type(), DType, t.device()->lang(), Lang, { \
      ret->device()->Exec(                                             \
          [t, ret](Context* ctx) {                                     \
            fn<DType, Lang>(t.Size(), t.blob(), ret->blob(), ctx);     \
          },                                                           \
          {t.blob()}, {ret->blob()});                                  \
    });                                                                \
  } while (0)

#define GenUnaryTensorFn(fn)                          \
  Tensor fn(const Tensor &t) {                        \
    Tensor ret(t.shape(), t.device(), t.data_type()); \
    auto *retptr = &ret;                              \
    EltwiseUnaryTensorFn(fn, t, retptr);              \
    return ret;                                       \
  }                                                   \
  void fn(const Tensor &in, Tensor *out) { EltwiseUnaryTensorFn(fn, in, out); }

GenUnaryTensorFn(Abs);
GenUnaryTensorFn(Exp);
GenUnaryTensorFn(Log);
GenUnaryTensorFn(ReLU);
GenUnaryTensorFn(Sigmoid);
GenUnaryTensorFn(Sign);
GenUnaryTensorFn(Sqrt);
GenUnaryTensorFn(Square);
GenUnaryTensorFn(Tanh);

// TODO(wangwei) conside async exec
template <> float Sum<float>(const Tensor &t) {
  float s = 0.0f;
  TYPE_LANG_SWITCH(t.data_type(), DType, t.device()->lang(), Lang, {
    t.device()->Exec(
        [t, &s](Context *ctx) {
          Sum<DType, Lang>(t.Size(), t.blob(), &s, ctx);
        },
        {t.blob()}, {});
  });
  return s;
}

Tensor Sum(const Tensor &M, int axis) {
  if (axis == 0) {
    Tensor out(Shape{M.shape(1)}, M.device(), M.data_type());
    SumRows(M, &out);
    return out;
  } else {
    CHECK_EQ(axis, 1) << "Not support Sum over axis = " << axis;
    Tensor out(Shape{M.shape(0)}, M.device(), M.data_type());
    SumColumns(M, &out);
    return out;
  }
}

Tensor Average(const Tensor &t, int axis) {
  // operator/ only has implementation for float scalar type, hence it is
  // necessary to cast the denominator to a float.
  // TODO(wangwei) implement function for cast scalar type involved in Tensor
  // functions. E.g.,
  // template<S, D>
  // D CastTo(S x) {
  //   return D(x);
  // }
  // for speical types, e.g., fp16:
  // tempalte<>
  // fp16 CastType(float x) {
  //    ....
  // }
  if (axis == 0) {
    return Sum(t, 0) / (1.0f * t.shape().at(0));
  } else {
    CHECK_EQ(axis, 1);
    return Sum(t, 1) / (1.0f * t.shape().at(1));
  }
}

Tensor SoftMax(const Tensor &in, int axis) {
  Tensor out(in.shape(), in.device(), in.data_type());
  SoftMax(in, axis, &out);
  return out;
}

void SoftMax(const Tensor &in, int axis, Tensor *out) {
  size_t nrow = 1, ncol = in.Size(), size = ncol;
  CHECK_GE(axis, 0);
  if (axis > 0) {
    nrow = Product(in.shape(), 0, axis);
    CHECK_EQ(size % nrow, 0u) << "Size = " << size << " nrow = " << nrow;
    ncol = size / nrow;
  }
  Exp(in, out);
  out->Reshape(Shape{nrow, ncol});
  Tensor sum(Shape{nrow}, in.device(), in.data_type());
  SumColumns(*out, &sum);
  DivColumn(sum, out);
}

#define EltwiseBinaryTensorFn(fn, lhs, rhs, ret)                               \
  do {                                                                         \
    TYPE_LANG_SWITCH(lhs.data_type(), DType, lhs.device()->lang(), Lang, {     \
      CHECK_EQ(sizeof(DType), SizeOf(rhs.data_type()));                        \
      ret->device()->Exec(                                                     \
          [lhs, rhs, ret](Context *ctx) {                                      \
            fn<DType, Lang>(lhs.Size(), lhs.blob(), rhs.blob(), ret->blob(),   \
                            ctx);                                              \
          },                                                                   \
          {lhs.blob(), rhs.blob()}, {ret->blob()});                            \
    });                                                                        \
  } while (0)

#define GenBinaryTensorFn(op, fn)                                        \
  Tensor op(const Tensor &lhs, const Tensor &rhs) {                            \
    Tensor ret(lhs.shape(), lhs.device(), lhs.data_type());                    \
    fn(lhs, rhs, &ret);                                                        \
    return ret;                                                                \
  }                                                                            \
  void fn(const Tensor &lhs, const Tensor &rhs, Tensor *ret) {                 \
    EltwiseBinaryTensorFn(fn, lhs, rhs, ret);                                  \
  }

GenBinaryTensorFn(operator+, Add);
GenBinaryTensorFn(operator-, Sub);
GenBinaryTensorFn(operator*, EltwiseMult);
GenBinaryTensorFn(operator/, Div);
GenBinaryTensorFn(Pow, Pow);

#define EltwiseTensorScalarFn(fn, t, x, ret)                                   \
  do {                                                                         \
    TYPE_LANG_SWITCH(t.data_type(), DType, t.device()->lang(), Lang, {         \
      static_assert(std::is_same<SType, DType>::value,                         \
                    "The Scalar type must match the Tensor data type");        \
      ret->device()->Exec(                                                     \
          [t, x, ret](Context *ctx) {                                          \
            fn<DType, Lang>(t.Size(), t.blob(), x, ret->blob(), ctx);          \
          },                                                                   \
          {t.blob()}, {ret->blob()});                                          \
    });                                                                        \
  } while (0)

#define GenTensorScalarFn(op, fn)                                        \
  template <typename SType> Tensor op(const Tensor &t, SType x) {              \
    Tensor ret(t.shape(), t.device(), t.data_type());                          \
    fn(t, x, &ret);                                                            \
    return ret;                                                                \
  }                                                                            \
  template <typename SType> void fn(const Tensor &t, SType x, Tensor *ret) {   \
    EltwiseTensorScalarFn(fn, t, x, ret);                                      \
  }                                                                            \
  template Tensor op<float>(const Tensor &t, float x);                         \
  template void fn<float>(const Tensor &t, const float x, Tensor *ret)

GenTensorScalarFn(operator+, Add);
GenTensorScalarFn(operator-, Sub);
GenTensorScalarFn(operator*, EltwiseMult);
GenTensorScalarFn(operator/, Div);
GenTensorScalarFn(Pow, Pow);
GenTensorScalarFn(operator<, LT);
GenTensorScalarFn(operator<=, LE);
GenTensorScalarFn(operator>, GT);
GenTensorScalarFn(operator>=, GE);

// ================Blas operations============================================
Tensor Mult(const Tensor &lhs, const Tensor &rhs) {
  Tensor ret(Shape{lhs.shape(0), rhs.shape(1)}, lhs.device(), lhs.data_type());
  Mult(lhs, rhs, &ret);
  return ret;
}

void Mult(const Tensor &lhs, const Tensor &rhs, Tensor *ret) {
  Mult(1.0f, lhs, rhs, 0.0f, ret);
}

void Mult(const float alpha, const Tensor &A, const Tensor &B, const float beta,
          Tensor *C) {
  CHECK_EQ(A.shape().size(), 2u);
  if (B.nDim() == 1u) {
    TYPE_LANG_SWITCH(A.data_type(), DType, A.device()->lang(), Lang, {
      C->device()->Exec(
          [alpha, A, beta, B, C](Context *ctx) {
            GEMV<DType, Lang>(A.transpose(), A.shape(0), A.shape(1), alpha,
                              A.blob(), B.blob(), beta, C->blob(), ctx);
          },
          {A.blob(), B.blob()}, {C->blob()});
    });
  } else {
    CHECK(!C->transpose());
    TYPE_LANG_SWITCH(A.data_type(), DType, A.device()->lang(), Lang, {
      C->device()->Exec(
          [alpha, A, beta, B, C](Context *ctx) {
            GEMM<DType, Lang>(A.transpose(), B.transpose(), A.shape(0),
                              B.shape(1), A.shape(1), alpha, A.blob(), B.blob(),
                              beta, C->blob(), ctx);
          },
          {A.blob(), B.blob()}, {C->blob()});
    });
  }
}

void Bernoulli(float p, Tensor *t) {
  TYPE_LANG_SWITCH(t->data_type(), DType, t->device()->lang(), Lang, {
    t->device()->Exec(
        [p, t](Context *ctx) {
          Bernoulli<DType, Lang>(t->Size(), p, t->blob(), ctx);
        },
        {}, {t->blob()}, true);
  });
}

void Uniform(float low, float high, Tensor *t) {
  TYPE_LANG_SWITCH(t->data_type(), DType, t->device()->lang(), Lang, {
    t->device()->Exec(
        [low, high, t](Context *ctx) {
          Uniform<DType, Lang>(t->Size(), low, high, t->blob(), ctx);
        },
        {}, {t->blob()}, true);
  });
}

void Gaussian(float mean, float std, Tensor *t) {
  TYPE_LANG_SWITCH(t->data_type(), DType, t->device()->lang(), Lang, {
    t->device()->Exec(
        [mean, std, t](Context *ctx) {
          Gaussian<DType, Lang>(t->Size(), mean, std, t->blob(), ctx);
        },
        {}, {t->blob()}, true);
  });
}

// ======follow the consistency guide
void AddColumn(const Tensor &v, Tensor *M) { AddColumn(1, 1, v, M); }
/// Add column 'v' onto each column of matrix M;
void AddColumn(const float alpha, const float beta, const Tensor &v,
               Tensor *M) {
  if (M->transpose()) {
    Tensor X = M->T();
    AddRow(v, &X);
  } else {
    CHECK_EQ(M->nDim(), 2u);
    CHECK_EQ(v.nDim(), 1u);
    size_t nb_row = M->shape(0), nb_col = M->shape(1);
    CHECK_EQ(nb_row, v.Size());

    Tensor one(Shape{1, nb_col}, M->device(), M->data_type());
    one.SetValue(1.0f); // TODO(wangwei) cast type
    Tensor vmat = Reshape(v, Shape{nb_row, 1});
    Mult(alpha, vmat, one, beta, M);
  }
}
void AddRow(const Tensor &v, Tensor *M) { AddRow(1, 1, v, M); }

/// Sub column 'v' by each column of matrix M; write results into 'out'
void AddRow(const float alpha, const float beta, const Tensor &v, Tensor *M) {
  if (M->transpose()) {
    Tensor X = M->T();
    AddColumn(v, &X);
  } else {
    CHECK_EQ(M->nDim(), 2u);
    CHECK_EQ(v.nDim(), 1u);
    size_t nb_row = M->shape(0), nb_col = M->shape(1);
    CHECK_EQ(nb_col, v.Size());

    Tensor one(Shape{nb_row, 1}, M->device(), M->data_type());
    one.SetValue(1.0f);
    Tensor vmat = Reshape(v, Shape{1, nb_col});
    Mult(alpha, one, vmat, beta, M);
  }
}

template <typename SType> Tensor Div(const SType alpha, const Tensor &in) {
  Tensor out(in.shape(), in.device(), in.data_type());
  Div(alpha, in, &out);
  return out;
}

template Tensor Div<float>(const float, const Tensor &);

template <typename SType>
void Div(const SType alpha, const Tensor &in, Tensor *out) {
  CheckDataTypeAndLang(in, *out);
  CHECK(in.shape() == out->shape());
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    // TODO(wangwei) type cast SType to DType;
    in.device()->Exec(
        [alpha, in, out](Context *ctx) {
          Div<DType, Lang>(in.Size(), alpha, in.blob(), out->blob(), ctx);
        },
        {in.blob()}, {out->blob()});
  });
}
template void Div<float>(const float, const Tensor &, Tensor *);

/// Divide column 'v' by each column of matrix M; write results into 'out'
void DivColumn(const Tensor &v, Tensor *M) {
  Tensor inv;
  TYPE_SWITCH(v.data_type(), DType, { inv = Div(DType(1), v); });
  MultColumn(inv, M);
}

/// Divide row 'v' by each row of matrix M; write results into 'out'
void DivRow(const Tensor &v, Tensor *M) {
  Tensor inv;
  TYPE_SWITCH(v.data_type(), DType, { inv = Div(DType(1), v); });
  MultRow(inv, M);
}

/// Multiply column 'v' and each column of matrix M; write results into 'out'
void MultColumn(const Tensor &v, Tensor *M) {
  CHECK(!M->transpose()) << "Not supported yet";
  CHECK_EQ(M->nDim(), 2u);
  CHECK_EQ(v.nDim(), 1u);
  CHECK_EQ(v.Size(), M->shape(0));
  CheckDataTypeAndLang(*M, v);
  TYPE_LANG_SWITCH(v.data_type(), DType, v.device()->lang(), Lang, {
    v.device()->Exec(
        [M, v](Context *ctx) {
          DGMM<DType, Lang>(false, M->shape(0), M->shape(1), M->blob(),
                            v.blob(), M->blob(), ctx);
        },
        {M->blob(), v.blob()}, {M->blob()});
  });
}

/// Multiply row 'v' with each row of matrix M; write results into 'out'
void MultRow(const Tensor &v, Tensor *M) {
  CHECK(!M->transpose()) << "Not supported yet";
  CHECK_EQ(M->nDim(), 2u);
  CHECK_EQ(v.nDim(), 1u);
  CHECK_EQ(v.Size(), M->shape(1));
  CheckDataTypeAndLang(*M, v);
  TYPE_LANG_SWITCH(v.data_type(), DType, v.device()->lang(), Lang, {
    v.device()->Exec(
        [M, v](Context *ctx) {
          DGMM<DType, Lang>(true, M->shape(0), M->shape(1), M->blob(), v.blob(),
                            M->blob(), ctx);
        },
        {M->blob(), v.blob()}, {M->blob()});
  });
}

void SubColumn(const Tensor &v, Tensor *M) { AddColumn(-1, 1, v, M); }

void SubRow(const Tensor &v, Tensor *M) { AddRow(-1, 1, v, M); }

void SumColumns(const Tensor &M, Tensor *v) {
  if (M.transpose()) {
    Tensor X = M.T();
    SumRows(X, v);
  } else {
    CHECK_EQ(M.nDim(), 2u);
    CHECK_EQ(v->nDim(), 1u);
    size_t nb_row = M.shape().at(0), nb_col = M.shape().at(1);
    CHECK_EQ(nb_row, v->Size());

    Tensor one(Shape{nb_col, 1}, M.device(), M.data_type());
    one.SetValue(1.0f); // TODO(wangwei) cast type
    Mult(M, one, v);
  }
}
void SumRows(const Tensor &M, Tensor *v) {
  if (M.transpose()) {
    Tensor X = M.T();
    SumColumns(X, v);
  } else {
    CHECK_EQ(M.nDim(), 2u);
    CHECK_EQ(v->nDim(), 1u);
    size_t nb_row = M.shape(0), nb_col = M.shape(1);
    CHECK_EQ(nb_col, v->Size());

    Tensor one(Shape{nb_row, 1}, M.device(), M.data_type());
    one.SetValue(1.0f); // TODO(wangwei) cast type
    Tensor X = M.T();
    Mult(X, one, v);
  }
}
}  // namespace singa
