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
  // LOG(ERROR) << "~";
  if (blob_ != nullptr && blob_->DecRefCount() == 0) device_->FreeBlob(blob_);
  blob_ = nullptr;
}

Tensor::Tensor() { device_ = &defaultDevice; }

Tensor::Tensor(const Shape &shape, const DataType dtype)
    : data_type_(dtype), device_(&defaultDevice), shape_(shape) {
  device_ = &defaultDevice;
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(Shape &&shape, const DataType dtype)
    : data_type_(dtype), device_(&defaultDevice), shape_(shape) {
  device_ = &defaultDevice;
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(const Shape &shape, Device *device, const DataType dtype)
    : data_type_(dtype), device_(device), shape_(shape) {
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(Shape &&shape, Device *device, const DataType dtype)
    : data_type_(dtype), device_(device), shape_(shape) {
  blob_ = device_->NewBlob(Product(shape_) * SizeOf(data_type_));
}
Tensor::Tensor(const Tensor &in)
    : transpose_(in.transpose_),
      data_type_(in.data_type_),
      device_(in.device_),
      blob_(in.blob()),
      shape_(in.shape_) {
  blob_->IncRefCount();
}

Tensor::Tensor(Tensor &&in)
    : transpose_(in.transpose_),
      data_type_(in.data_type_),
      device_(in.device_),
      shape_(std::move(in.shape_)) {
  blob_ = in.blob_;
  in.blob_ = nullptr;
}

void Tensor::ResetLike(const Tensor &in) {
  if (blob_ == nullptr || device_ != in.device_ || MemSize() != in.MemSize()) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0) device_->FreeBlob(blob_);
    shape_ = in.shape_;
    device_ = in.device_;
    data_type_ = in.data_type_;
    blob_ = device_->NewBlob(in.MemSize());
  }
}

void Tensor::Reshape(const Shape &shape) {
  if (Product(shape_) != Product(shape)) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0) device_->FreeBlob(blob_);
    blob_ = device_->NewBlob(Product(shape) * SizeOf(data_type_));
  }
  shape_ = shape;
}

void Tensor::Reshape(Shape &&shape) {
  if (Product(shape_) != Product(shape)) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0) device_->FreeBlob(blob_);
    blob_ = device_->NewBlob(Product(shape) * SizeOf(data_type_));
  }
  shape_ = std::move(shape);
}

void Tensor::AsType(const DataType type) {
  if (data_type_ != type) {
    if (blob_ != nullptr && blob_->DecRefCount() == 0) device_->FreeBlob(blob_);
    blob_ = device_->NewBlob(Product(shape_) * SizeOf(type));
    data_type_ = type;
  }
}

void Tensor::ToDevice(Device *dst) {
  // TODO(wangwei) the comparison is very strict. May compare against device ID?
  if (device_ != dst) {
    Tensor tmp(shape_, dst, data_type_);
    tmp.CopyData(*this);
    if (blob_ != nullptr && blob_->DecRefCount() == 0) device_->FreeBlob(blob_);
    blob_ = tmp.blob_;
    tmp.blob_ = nullptr;
    device_ = dst;
  }
}

void Tensor::ToHost() { ToDevice(device_->host()); }

template <typename DType>
void Tensor::CopyDataFromHostPtr(const DType *src, const size_t num) {
  CHECK_EQ(sizeof(DType), SizeOf(data_type_))
      << "data_type is " << DataType_Name(data_type_)
      << " user given type is of size " << sizeof(DType);
  if (src != nullptr) {
    device_->CopyDataFromHostPtr(blob(), src, sizeof(DType) * num, 0);
  } else {
    LOG(WARNING) << "Copy data from null host ptr";
  }
}
template void Tensor::CopyDataFromHostPtr(const float *src, const size_t num);
template void Tensor::CopyDataFromHostPtr(const int *src, const size_t num);

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

Tensor &Tensor::operator=(const Tensor &in) {
  // LOG(ERROR) << "= const &";
  if (blob_ != nullptr && blob_->DecRefCount() == 0) device_->FreeBlob(blob_);
  transpose_ = in.transpose_;
  data_type_ = in.data_type_;
  shape_ = in.shape_;
  device_ = in.device_;
  blob_ = in.blob();
  blob_->IncRefCount();
  return *this;
}

Tensor &Tensor::operator=(Tensor &&in) {
  // LOG(ERROR) << "= &&";
  if (blob_ != nullptr && blob_->DecRefCount() == 0) device_->FreeBlob(blob_);
  transpose_ = in.transpose_;
  data_type_ = in.data_type_;
  shape_ = std::move(in.shape_);
  device_ = in.device_;
  blob_ = in.blob_;
  in.blob_ = nullptr;
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

#define GenUnaryTensorArgMemberFn(op, fn) \
  Tensor &Tensor::op(const Tensor &in) {  \
    fn(*this, in, this);                  \
    return *this;                         \
  }

GenUnaryTensorArgMemberFn(operator+=, Add);
GenUnaryTensorArgMemberFn(operator-=, Sub);
GenUnaryTensorArgMemberFn(operator*=, EltwiseMult);
GenUnaryTensorArgMemberFn(operator/=, Div);

#define GenUnaryScalarArgMemberFn(op, fn) \
  template <typename DType>               \
  Tensor &Tensor::op(const DType x) {     \
    fn(*this, x, this);                   \
    return *this;                         \
  }                                       \
  template Tensor &Tensor::op<float>(const float x)

GenUnaryScalarArgMemberFn(operator-=, Sub);
GenUnaryScalarArgMemberFn(operator+=, Add);
GenUnaryScalarArgMemberFn(operator*=, EltwiseMult);
GenUnaryScalarArgMemberFn(operator/=, Div);



// ====================Tensor Operations=======================================
void CopyDataToFrom(Tensor *dst, const Tensor &src, const size_t num,
                    const size_t dst_offset, const size_t src_offset) {
  auto width = SizeOf(src.data_type());
  CHECK_EQ(width, SizeOf(dst->data_type()));
  size_t nBytes = num * width;
  auto d_offset = dst_offset * width;
  auto s_offset = src_offset * width;
  CHECK_GE(src.MemSize(), s_offset + nBytes);
  CHECK_GE(dst->MemSize(), d_offset + nBytes);

  Device *src_dev = src.device(), *dst_dev = dst->device();
  Blob *from = src.blob(), *to = dst->blob();
  if (dst_dev->lang() != src_dev->lang()) {
    // let the none cpp device conduct copy op
    if (dst_dev->lang() == kCpp) {
      src_dev->CopyDataToFrom(to, from, nBytes, kDeviceToHost, d_offset,
                              s_offset);
    } else if (src_dev->lang() == kCpp) {
      dst_dev->CopyDataToFrom(to, from, nBytes, kHostToDevice, d_offset,
                              s_offset);
    } else {
      LOG(FATAL) << "Not support mem copy betwee Cuda and OpenCL device";
    }
  } else {
    auto direct = src_dev->lang() == kCpp ? kHostToHost : kDeviceToDevice;
    src_dev->CopyDataToFrom(to, from, nBytes, direct, d_offset, s_offset);
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

/// typedef DType and Lang according to data type and device programming
/// language respectively.
/// type is from DataType, and lang is from LangType.
/// DType and Lang would be used in __VA_ARGS__.
#define TYPE_LANG_SWITCH(dtype, DType, ltype, Lang, ...)       \
  do {                                                         \
    const int _SwitchShift = 3;                                \
    int _SwitchHash = ((dtype) << _SwitchShift) + (ltype);     \
    switch (_SwitchHash) {                                     \
      case ((kFloat32 << _SwitchShift) + kCuda): {             \
        typedef float DType;                                   \
        typedef lang::Cuda Lang;                               \
        { __VA_ARGS__ }                                        \
        break;                                                 \
      }                                                        \
      case ((kFloat32 << _SwitchShift) + kCpp): {              \
        typedef float DType;                                   \
        typedef lang::Cpp Lang;                                \
        { __VA_ARGS__ }                                        \
        break;                                                 \
      }                                                        \
      case ((kFloat32 << _SwitchShift) + kOpencl): {           \
        typedef float DType;                                   \
        typedef lang::Opencl Lang;                             \
        { __VA_ARGS__ }                                        \
        break;                                                 \
      }                                                        \
      default:                                                 \
        LOG(FATAL) << "Unknown combination of data type "      \
                   << DataType_Name(dtype) << " and language " \
                   << LangType_Name(ltype);                    \
    }                                                          \
  } while (0)

// =============Element-wise operations====================================
/// L2 norm, Do not use Nrm2 (name conflict).
float Tensor::L2() const {
  float nrm = 0.0f;
  TYPE_LANG_SWITCH(data_type_, DType, device_->lang(), Lang, {
    device_->Exec([&nrm, this](Context *ctx) {
      DType ret;
      Nrm2<DType, Lang>(this->Size(), this->blob(), &ret, ctx);
      nrm = TypeCast<DType, float>(ret);
    }, {this->blob()}, {});
  });
  return nrm;
}
template <typename SType>
void Tensor::SetValue(const SType x) {
  CHECK_EQ(sizeof(SType), SizeOf(data_type_));
  auto size = Size();
  auto ptr = blob_;
  TYPE_LANG_SWITCH(data_type_, DType, device_->lang(), Lang, {
    // cast x to DType
    device_->Exec([size, x, ptr](Context *ctx) {
      Set<DType, Lang>(size, x, ptr, ctx);
    }, {}, {ptr});
  });
}
template void Tensor::SetValue<float>(const float x);

#define EltwiseUnaryTensorFn(fn, t, ret)                               \
  do {                                                                 \
    TYPE_LANG_SWITCH(t.data_type(), DType, t.device()->lang(), Lang, { \
      ret->device()->Exec([t, ret](Context * ctx) {                    \
        fn<DType, Lang>(t.Size(), t.blob(), ret->blob(), ctx);         \
      }, {t.blob()}, {ret->blob()});                                   \
    });                                                                \
  } while (0)

#define GenUnaryTensorFn(fn)                             \
  Tensor fn(const Tensor &in) {                          \
    Tensor ret(in.shape(), in.device(), in.data_type()); \
    auto *retptr = &ret;                                 \
    EltwiseUnaryTensorFn(fn, in, retptr);                \
    return ret;                                          \
  }                                                      \
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

#define EltwiseBinaryTensorFn(fn, lhs, rhs, ret)                               \
  do {                                                                         \
    TYPE_LANG_SWITCH(lhs.data_type(), DType, lhs.device()->lang(), Lang, {     \
      CHECK_EQ(sizeof(DType), SizeOf(rhs.data_type()));                        \
      ret->device()->Exec([lhs, rhs, ret](Context * ctx) {                     \
        fn<DType, Lang>(lhs.Size(), lhs.blob(), rhs.blob(), ret->blob(), ctx); \
      }, {lhs.blob(), rhs.blob()}, {ret->blob()});                             \
    });                                                                        \
  } while (0)

#define GenBinaryTensorFn(op, fn)                              \
  Tensor op(const Tensor &lhs, const Tensor &rhs) {            \
    Tensor ret(lhs.shape(), lhs.device(), lhs.data_type());    \
    fn(lhs, rhs, &ret);                                        \
    return ret;                                                \
  }                                                            \
  void fn(const Tensor &lhs, const Tensor &rhs, Tensor *ret) { \
    EltwiseBinaryTensorFn(fn, lhs, rhs, ret);                  \
  }

GenBinaryTensorFn(operator+, Add);
GenBinaryTensorFn(operator-, Sub);
GenBinaryTensorFn(operator*, EltwiseMult);
GenBinaryTensorFn(operator/, Div);
GenBinaryTensorFn(Pow, Pow);

#define EltwiseTensorScalarFn(fn, t, x, ret)                            \
  do {                                                                  \
    TYPE_LANG_SWITCH(t.data_type(), DType, t.device()->lang(), Lang, {  \
      static_assert(std::is_same<SType, DType>::value,                  \
                    "The Scalar type must match the Tensor data type"); \
      ret->device()->Exec([t, x, ret](Context * ctx) {                  \
        fn<DType, Lang>(t.Size(), t.blob(), x, ret->blob(), ctx);       \
      }, {t.blob()}, {ret->blob()});                                    \
    });                                                                 \
  } while (0)

#define GenTensorScalarFn(op, fn)                             \
  template <typename SType>                                   \
  Tensor op(const Tensor &in, const SType x) {                \
    Tensor ret(in.shape(), in.device(), in.data_type());      \
    fn(in, x, &ret);                                          \
    return ret;                                               \
  }                                                           \
  template <typename SType>                                   \
  void fn(const Tensor &in, const SType x, Tensor *ret) {     \
    EltwiseTensorScalarFn(fn, in, x, ret);                    \
  }                                                           \
  template Tensor op<float>(const Tensor &in, const float x); \
  template void fn<float>(const Tensor &in, const float x, Tensor *ret)

GenTensorScalarFn(operator+, Add);
GenTensorScalarFn(operator-, Sub);
GenTensorScalarFn(operator*, EltwiseMult);
GenTensorScalarFn(operator/, Div);
GenTensorScalarFn(Pow, Pow);
GenTensorScalarFn(operator<, LT);
GenTensorScalarFn(operator<=, LE);
GenTensorScalarFn(operator>, GT);
GenTensorScalarFn(operator>=, GE);
template <typename SType>
Tensor Div(const SType alpha, const Tensor &in) {
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
    in.device()->Exec([alpha, in, out](Context *ctx) {
      Div<DType, Lang>(in.Size(), alpha, in.blob(), out->blob(), ctx);
    }, {in.blob()}, {out->blob()});
  });
}
template void Div<float>(const float, const Tensor &, Tensor *);

// =============Matrix operations============================================
Tensor Average(const Tensor &M, int axis) {
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
    return Sum(M, 0) / (1.0f * M.shape(0));
  } else {
    CHECK_EQ(axis, 1);
    return Sum(M, 1) / (1.0f * M.shape(1));
  }
}
// TODO(wangwei) conside async exec
template <>
float Sum<float>(const Tensor &in) {
  float s = 0.0f;
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    in.device()->Exec([in, &s](Context *ctx) {
      Sum<DType, Lang>(in.Size(), in.blob(), &s, ctx);
    }, {in.blob()}, {});
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

void AddColumn(const Tensor &v, Tensor *M) { AddColumn(1, 1, v, M); }
/// Add column 'v' onto each column of matrix M;
template <typename SType>
void AddColumn(const SType alpha, const SType beta, const Tensor &v,
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
    one.SetValue(1.0f);  // TODO(wangwei) cast type
    Tensor vmat = Reshape(v, Shape{nb_row, 1});
    Mult(alpha, vmat, one, beta, M);
  }
}
template <>
void AddColumn(const float alpha, const float beta, const Tensor &v, Tensor *M);

void AddRow(const Tensor &v, Tensor *M) { AddRow(1, 1, v, M); }

/// Sub column 'v' by each column of matrix M; write results into 'out'
template <typename SType>
void AddRow(const SType alpha, const SType beta, const Tensor &v, Tensor *M) {
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
template <>
void AddRow(const float alpha, const float beta, const Tensor &v, Tensor *M);

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
    v.device()->Exec([M, v](Context *ctx) {
      DGMM<DType, Lang>(false, M->shape(0), M->shape(1), M->blob(), v.blob(),
                        M->blob(), ctx);
    }, {M->blob(), v.blob()}, {M->blob()});
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
    v.device()->Exec([M, v](Context *ctx) {
      DGMM<DType, Lang>(true, M->shape(0), M->shape(1), M->blob(), v.blob(),
                        M->blob(), ctx);
    }, {M->blob(), v.blob()}, {M->blob()});
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

    Tensor one(Shape{nb_col}, M.device(), M.data_type());
    one.SetValue(1.0f);  // TODO(wangwei) cast type
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

    Tensor one(Shape{nb_row}, M.device(), M.data_type());
    one.SetValue(1.0f);  // TODO(wangwei) cast type
    Tensor X = M.T();
    Mult(X, one, v);
  }
}
// ====================Random operations=====================================
template <typename SType>
void Bernoulli(const SType p, Tensor *out) {
  TYPE_LANG_SWITCH(out->data_type(), DType, out->device()->lang(), Lang, {
    auto prob = TypeCast<SType, DType>(p);
    out->device()->Exec([prob, out](Context *ctx) {
      Bernoulli<DType, Lang>(out->Size(), prob, out->blob(), ctx);
    }, {}, {out->blob()}, true);
  });
}
template void Bernoulli<float>(const float p, Tensor *out);

template <typename SType>
void Uniform(const SType low, const SType high, Tensor *out) {
  TYPE_LANG_SWITCH(out->data_type(), DType, out->device()->lang(), Lang, {
    auto l = TypeCast<SType, DType>(low);
    auto h = TypeCast<SType, DType>(high);
    out->device()->Exec([l, h, out](Context *ctx) {
      Uniform<DType, Lang>(out->Size(), l, h, out->blob(), ctx);
    }, {}, {out->blob()}, true);
  });
}
template void Uniform<float>(const float low, const float high, Tensor *out);

template <typename SType>
void Gaussian(const SType mean, const SType std, Tensor *out) {
  TYPE_LANG_SWITCH(out->data_type(), DType, out->device()->lang(), Lang, {
    auto m = TypeCast<SType, DType>(mean);
    auto s = TypeCast<SType, DType>(std);
    out->device()->Exec([m, s, out](Context *ctx) {
      Gaussian<DType, Lang>(out->Size(), m, s, out->blob(), ctx);
    }, {}, {out->blob()}, true);
  });
}
template void Gaussian<float>(const float mean, const float std, Tensor *out);

// ================Blas operations============================================
template <typename SType>
void Axpy(const SType alpha, const Tensor &in, Tensor *out) {
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    auto a = TypeCast<SType, DType>(alpha);
    out->device()->Exec([a, in, out](Context *ctx) {
      Axpy<DType, Lang>(in.Size(), a, in.blob(), out->blob(), ctx);
    }, {in.blob(), out->blob()}, {out->blob()});
  });
}
template void Axpy(const float alpha, const Tensor &in, Tensor *out);

Tensor Mult(const Tensor &A, const Tensor &B) {
  Shape s;
  s.push_back(A.shape(0));
  if (B.nDim() == 2) s.push_back(B.shape(1));
  Tensor out(s, A.device(), A.data_type());
  Mult(A, B, &out);
  return out;
}

void Mult(const Tensor &A, const Tensor &B, Tensor *out) {
  Mult(1.0f, A, B, 0.0f, out);
}

template <typename SType>
void Mult(const SType alpha, const Tensor &A, const Tensor &B, const SType beta,
          Tensor *C) {
  CHECK_EQ(A.shape().size(), 2u);
  if (B.nDim() == 1u) {
    TYPE_LANG_SWITCH(A.data_type(), DType, A.device()->lang(), Lang, {
      auto a = TypeCast<SType, DType>(alpha);
      auto b = TypeCast<SType, DType>(beta);
      C->device()->Exec([a, A, b, B, C](Context *ctx) {
        GEMV<DType, Lang>(A.transpose(), A.shape(0), A.shape(1), a, A.blob(),
                          B.blob(), b, C->blob(), ctx);
      }, {A.blob(), B.blob()}, {C->blob()});
    });
  } else {
    CHECK(!C->transpose());
    TYPE_LANG_SWITCH(A.data_type(), DType, A.device()->lang(), Lang, {
      auto a = TypeCast<SType, DType>(alpha);
      auto b = TypeCast<SType, DType>(beta);
      C->device()->Exec([a, A, b, B, C](Context *ctx) {
        GEMM<DType, Lang>(A.transpose(), B.transpose(), A.shape(0), B.shape(1),
                          A.shape(1), a, A.blob(), B.blob(), b, C->blob(), ctx);
      }, {A.blob(), B.blob()}, {C->blob()});
    });
  }
}

}  // namespace singa
