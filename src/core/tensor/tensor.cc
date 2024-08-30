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

#include <algorithm>
#include <utility>

#include "./tensor_math.h"
#include "./tensor_math_cpp.h"
#include "./tensor_math_cuda.h"
#include "./tensor_math_opencl.h"

#define Noaxis 9999

namespace singa {

template half_float::half TypeCast(const float &x);
template float TypeCast(const half_float::half &x);
template int TypeCast(const float &x);
template float TypeCast(const int &x);

Tensor::~Tensor() {
  if (block_ != nullptr && block_->DecRefCount() == 0) {
    device_->FreeBlock(block_);
  }
  block_ = nullptr;
}

Tensor::Tensor() {
  device_ = defaultDevice;
  stride_ = {1};
}

// non-strided constructors
Tensor::Tensor(const Shape &shape, DataType dtype)
    : data_type_(dtype), device_(defaultDevice), shape_(shape) {
  size_t size = Product(shape_) * SizeOf(data_type_);
  if (size) {
    block_ = device_->NewBlock((int)size);
  }
  generate_stride();
}

// non-strided constructors with device
Tensor::Tensor(const Shape &shape, std::shared_ptr<Device> device,
               DataType dtype)
    : data_type_(dtype), device_(device), shape_(shape) {
  size_t size = Product(shape_) * SizeOf(data_type_);
  if (size) {
    block_ = device_->NewBlock((int)size);
  }
  generate_stride();
}

Tensor::Tensor(const Tensor &in)
    : data_type_(in.data_type_),
      device_(in.device_),
      block_(in.block()),
      shape_(in.shape_),
      stride_(in.stride_) {
  if (block_ != nullptr) block_->IncRefCount();
}

Tensor::Tensor(Tensor &&in)
    : data_type_(in.data_type_),
      device_(in.device_),
      shape_(std::move(in.shape_)),
      stride_(std::move(in.stride_)) {
  block_ = in.block_;
  in.block_ = nullptr;
}

Tensor &Tensor::ResetLike(const Tensor &in) {
  if (block_ == nullptr || device_ != in.device_ || MemSize() != in.MemSize()) {
    if (block_ != nullptr && block_->DecRefCount() == 0)
      device_->FreeBlock(block_);
    device_ = in.device_;
    data_type_ = in.data_type_;
    block_ = device_->NewBlock((int)in.MemSize());
  }
  shape_ = in.shape_;
  stride_ = in.stride_;
  return *this;
}

Tensor &Tensor::Resize(const Shape &shape) {
  if (Size() != Product(shape)) {
    if (block_ != nullptr && block_->DecRefCount() == 0)
      device_->FreeBlock(block_);
    block_ = device_->NewBlock((int)(Product(shape) * SizeOf(data_type_)));
  }
  shape_ = shape;
  generate_stride();
  return *this;
}

Tensor Resize(const Tensor &in, const Shape &shape) {
  Tensor out(in);
  out.Resize(shape);
  return out;
}

#define TYPE_TYPE_LANG_SWITCH(ldtype, LDType, rdtype, RDType, ltype, Lang,     \
                              ...)                                             \
  do {                                                                         \
    const int _SwitchShift = 3;                                                \
    int _SwitchHash =                                                          \
        ((ldtype) << _SwitchShift * 2) + ((rdtype) << _SwitchShift) + (ltype); \
    switch (_SwitchHash) {                                                     \
      case (((kFloat16) << _SwitchShift * 2) + (kFloat32 << _SwitchShift) +    \
            kCpp): {                                                           \
        typedef half_float::half LDType;                                       \
        typedef float RDType;                                                  \
        typedef lang::Cpp Lang;                                                \
        { __VA_ARGS__ }                                                        \
        break;                                                                 \
      }                                                                        \
      case (((kFloat32) << _SwitchShift * 2) + (kFloat16 << _SwitchShift) +    \
            kCpp): {                                                           \
        typedef float LDType;                                                  \
        typedef half_float::half RDType;                                       \
        typedef lang::Cpp Lang;                                                \
        { __VA_ARGS__ }                                                        \
        break;                                                                 \
      }                                                                        \
      case (((kFloat16) << _SwitchShift * 2) + (kFloat32 << _SwitchShift) +    \
            kCuda): {                                                          \
        typedef half_float::half LDType;                                       \
        typedef float RDType;                                                  \
        typedef lang::Cuda Lang;                                               \
        { __VA_ARGS__ }                                                        \
        break;                                                                 \
      }                                                                        \
      case (((kFloat32) << _SwitchShift * 2) + (kFloat16 << _SwitchShift) +    \
            kCuda): {                                                          \
        typedef float LDType;                                                  \
        typedef half_float::half RDType;                                       \
        typedef lang::Cuda Lang;                                               \
        { __VA_ARGS__ }                                                        \
        break;                                                                 \
      }                                                                        \
      case (((kFloat32) << _SwitchShift * 2) + (kInt << _SwitchShift) +        \
            kCuda): {                                                          \
        typedef float LDType;                                                  \
        typedef int RDType;                                                    \
        typedef lang::Cuda Lang;                                               \
        { __VA_ARGS__ }                                                        \
        break;                                                                 \
      }                                                                        \
      case (((kInt) << _SwitchShift * 2) + (kFloat32 << _SwitchShift) +        \
            kCuda): {                                                          \
        typedef int LDType;                                                    \
        typedef float RDType;                                                  \
        typedef lang::Cuda Lang;                                               \
        { __VA_ARGS__ }                                                        \
        break;                                                                 \
      }                                                                        \
      case (((kFloat32) << _SwitchShift * 2) + (kInt << _SwitchShift) +        \
            kCpp): {                                                           \
        typedef float LDType;                                                  \
        typedef int RDType;                                                    \
        typedef lang::Cpp Lang;                                                \
        { __VA_ARGS__ }                                                        \
        break;                                                                 \
      }                                                                        \
      case (((kInt) << _SwitchShift * 2) + (kFloat32 << _SwitchShift) +        \
            kCpp): {                                                           \
        typedef int LDType;                                                    \
        typedef float RDType;                                                  \
        typedef lang::Cpp Lang;                                                \
        { __VA_ARGS__ }                                                        \
        break;                                                                 \
      }                                                                        \
      default:                                                                 \
        LOG(FATAL) << "Unknown combination of left data type "                 \
                   << DataType_Name(ldtype) << " and right data type "         \
                   << DataType_Name(rdtype) << " and language "                \
                   << LangType_Name(ltype);                                    \
    }                                                                          \
  } while (0)

// return new tensor
Tensor Tensor::AsType(const DataType type) const {
  if (data_type_ != type) {
    const Tensor &thisRef = *this;
    Tensor ret(shape_, device_, type);
    TYPE_TYPE_LANG_SWITCH(
        data_type_, LDType, type, RDType, device_->lang(), Lang, {
          ret.device()->Exec(
              [thisRef, ret](Context *ctx) mutable {
                CastCopy<LDType, RDType, Lang>(&thisRef, &ret, ctx);
              },
              {this->block()}, {ret.block()}, "AsType");
        });
    return ret;
  } else {
    Tensor t = this->Clone();
    return t;
  }
}

Tensor &Tensor::ToType(const DataType type) {
  CHECK(block() && block()->initialized() == true)
      << "the data of the tensor needs be initialized before casting to "
         "another type";
  if (data_type_ != type) {
    auto ret = this->AsType(type);
    std::swap(ret.block_, block_);
    data_type_ = type;
  }
  return *this;
}

Tensor &Tensor::ToDevice(std::shared_ptr<Device> dst) {
  // TODO(wangwei) the comparison is restricted. May compare against device ID?
  if (device_ != dst) {
    // WARNING: this function can't be buffered
    Tensor tmp(shape_, dst, data_type_);
    if (block_ != nullptr && Size() && block_->initialized())
      tmp.CopyData(*this);
    if (block_ != nullptr && block_->DecRefCount() == 0)
      device_->FreeBlock(block_);
    block_ = tmp.block_;
    tmp.block_ = nullptr;
    device_ = dst;
  }
  return *this;
}

Tensor &Tensor::ToHost() {
  if (device_ != defaultDevice) ToDevice(device_->host());
  return *this;
}

template <typename DType>
void Tensor::CopyDataFromHostPtr(const DType *src, const size_t num,
                                 const size_t offset) const {
  CHECK_EQ(sizeof(DType), SizeOf(data_type_))
      << "data_type is " << DataType_Name(data_type_)
      << " user given type is of size " << sizeof(DType);
  if (src != nullptr) {
    Device *dev = device_.get();
    const Tensor &thisRef = *this;
    size_t nBytes = sizeof(DType) * num;
    size_t dst_offset = sizeof(DType) * offset;
    device_->Exec(
        [dev, thisRef, src, nBytes, dst_offset](Context *ctx) mutable {
          dev->CopyDataFromHostPtr(thisRef.block(), src, nBytes, dst_offset,
                                   ctx);
        },
        {}, {block()}, "CopyDataFromHostPtr");
  } else {
    LOG(WARNING) << "Copy data from null host ptr";
  }
}
template void Tensor::CopyDataFromHostPtr(const unsigned char *src,
                                          const size_t num,
                                          const size_t offset) const;
template void Tensor::CopyDataFromHostPtr(const half_float::half *src,
                                          const size_t num,
                                          const size_t offset) const;
template void Tensor::CopyDataFromHostPtr(const float *src, const size_t num,
                                          const size_t offset) const;
template void Tensor::CopyDataFromHostPtr(const int *src, const size_t num,
                                          const size_t offset) const;

void Tensor::CopyData(const Tensor &src) {
  CHECK_EQ(Size(), src.Size());
  CHECK(block_ != nullptr);
  CHECK_EQ(src.data_type(), data_type_)
      << "Could not copy data between different data type";
  // Do copy only if the src's block is already initialized.
  if (src.block_ != nullptr) {
    singa::CopyDataToFrom(this, src, Size(), 0, 0);
  }
}

void Tensor::RepeatData(const vector<size_t> &repeats, int axis,
                        int total_repeats, const Tensor &src) {
  if (repeats.size() == 1) {
    CHECK_EQ(Size(), src.Size() * total_repeats);
  } else {
    CHECK_EQ(Size(), src.Size() * total_repeats / src.shape()[axis]);
  }

  CHECK(block_ != nullptr);
  // Do repeat only if the src's block is already initialized.
  if (src.block_ != nullptr) {
    singa::RepeatDataToFrom(false, repeats, axis, this, src, Size());
  }
}

void Tensor::FromProto(const singa::TensorProto &proto) {
  if (block_ != nullptr && block_->DecRefCount() == 0)
    device_->FreeBlock(block_);
  block_ = nullptr;
  for (uint32_t s : proto.shape()) shape_.push_back(s);
  data_type_ = proto.data_type();
  block_ = device_->NewBlock((int)(Product(shape()) * SizeOf(data_type_)));
  // transpose_ = proto.transpose();
  stride_.clear();
  for (int32_t s : proto.stride()) stride_.push_back(s);
  switch (data_type_) {
    case kFloat32: {
      std::unique_ptr<float[]> data_ptr(new float[Product(shape_)]);
      for (size_t i = 0; i < Product(shape_); ++i)
        data_ptr[i] = static_cast<float>(proto.float_data((int)i));
      CopyDataFromHostPtr<float>(data_ptr.get(), Product(shape_));
      break;
    }
    case kDouble: {
      std::unique_ptr<double[]> data(new double[Product(shape_)]);
      for (size_t i = 0; i < Product(shape_); ++i)
        data[i] = proto.double_data((int)i);
      CopyDataFromHostPtr<double>(data.get(), Product(shape_));
      break;
    }
    case kInt: {
      std::unique_ptr<int[]> data(new int[Product(shape_)]);
      for (size_t i = 0; i < Product(shape_); ++i)
        data[i] = proto.int_data((int)i);
      CopyDataFromHostPtr<int>(data.get(), Product(shape_));
      break;
    }
    /// TODO(wangji): Implement to support C++ type char using bytes type in
    /// protobuf
    /// which is equivalent to string type is different from the other cases.
    /// The kchar
    /// and kUChar case is to be implemented.
    /*
    case kChar: {
      std::unique_ptr<char[]> data(new char[Product(shape_)]);
      for (size_t i = 0; i < Product(shape_); ++i)
        data[i] = static_cast<char>(proto.bytes_data(i));
      break;
    }
    case kUChar: {
      std::unique_ptr<unsigned char[]> data(new unsigned char[Product(shape_)]);
      for (size_t i = 0; i < Product(shape_); ++i)
        data[i] = static_cast<unsigned char>(proto.bytes_data(i));
      break;
    }
    */
    default: {
      LOG(FATAL) << "Unsupported Type" << DataType_Name(data_type_);
    }
  }
}

void Tensor::to_proto(singa::TensorProto *proto) const {
  proto->clear_shape();
  for (auto s : shape_) {
    proto->add_shape(s);
  }
  proto->set_data_type(data_type_);
  // proto->set_transpose(transpose_);
  proto->clear_stride();
  for (auto s : stride_) {
    proto->add_stride(s);
  }
  switch (data_type_) {
    case kFloat32: {
      proto->clear_float_data();
      const float *data_ptr = data<float>();
      for (size_t i = 0; i < Product(shape_); ++i)
        proto->add_float_data(data_ptr[i]);
      break;
    }
    case kDouble: {
      proto->clear_double_data();
      const double *data_ptr = data<double>();
      for (size_t i = 0; i < Product(shape_); ++i)
        proto->add_double_data(data_ptr[i]);
      break;
    }
    case kInt: {
      proto->clear_int_data();
      const int *data_ptr = data<int>();
      for (size_t i = 0; i < Product(shape_); ++i)
        proto->add_int_data(data_ptr[i]);
      break;
    }
    /*
    case kChar: {
      proto->clear_bytes_data();
      const char *data = data<char>();
      for (size_t i = 0; i < Product(shape_); ++i)
        proto->add_bytes_data(static_cast<unsigned char>(data[i]));
      break;
    }
    case kUChar: {
      proto->clear_bytes_data();
      const unsigned char *data = data<unsigned char>();
      for (size_t i = 0; i < Product(shape_); ++i)
        proto->add_bytes_data(static_cast<unsigned char>(data[i]));
      break;
    }
    */
    default: {
      LOG(FATAL) << "Unsupported Type" << DataType_Name(data_type_);
    }
  }
}

void Tensor::ToProto(singa::TensorProto *proto) const { to_proto(proto); }

Tensor Tensor::Repeat(const vector<size_t> &repeats, int axis,
                      std::shared_ptr<Device> device) {
  if (device == nullptr) device = device_;
  vector<size_t> tshape;
  int total_repeats = 0;
  if (axis == Noaxis) {
    total_repeats = repeats[0];
    tshape.push_back(Product(shape_) * total_repeats);
  } else {
    if (repeats.size() == 1) {
      total_repeats = repeats[0];
      for (int i = 0; i < static_cast<int>(shape_.size()); i++) {
        if (i == axis) {
          tshape.push_back(shape_[i] * total_repeats);
        } else {
          tshape.push_back(shape_[i]);
        }
      }
    } else {
      if (repeats.size() != shape_[axis]) {
        LOG(FATAL) << "the repeats number doesn't match the axis";
      }
      for (size_t i = 0; i < shape_[axis]; i++) {
        if (repeats[i] < 0) {
          LOG(FATAL) << "the repeats number is less than zero";
        }
        total_repeats += repeats[i];
      }
      for (int i = 0; i < static_cast<int>(shape_.size()); i++) {
        if (i == axis) {
          tshape.push_back(total_repeats);
        } else {
          tshape.push_back(shape_[i]);
        }
      }
    }
  }
  Tensor t(tshape, device_);
  // t.stride_.push_back(1);
  t.RepeatData(repeats, axis, total_repeats, *this);
  return t;
}

Tensor Tensor::Clone(std::shared_ptr<Device> device) const {
  if (device == nullptr) device = device_;
  Tensor t(shape_, device, data_type_);
  // t.transpose_ = transpose_;
  t.stride_ = stride_;
  t.CopyData(*this);
  return t;
}

void Tensor::Clone(Tensor *&other, std::shared_ptr<Device> device) const {
  if (device == nullptr) device = device_;
  other = new Tensor(shape_, device, data_type_);
  other->stride_ = stride_;
  other->CopyData(*this);
  return;
}

Tensor &Tensor::Broadcast(const Shape &shape, const int ignore_last_dim) {
  // TODO(wangwei) do we need to transform the mem layout if the tensor was
  // transposed?
  auto m = shape_.size() - 1, n = shape.size() - 1;
  // ignore_last_dim is useful for mult broadcast
  // e.g. (2,3,4)x(4,5) to (2,3,4)x(2,4,5)
  if (ignore_last_dim < std::min(m, n) + 1) {
    for (size_t i = ignore_last_dim; i <= std::min(m, n); i++) {
      if ((shape.at(n - i) != shape_.at(m - i)) && (shape.at(n - i) != 1)) {
        CHECK_EQ(shape_.at(m - i), 1) << "i= " << i << "\n";  // << Backtrace();
        shape_.at(m - i) = shape.at(n - i);
        stride_.at(m - i) = 0;
      }
    }
  }
  if (m < n) {
    for (size_t i = m + 1; i <= n; i++) {
      shape_.emplace(shape_.begin(), shape.at(n - i));
      stride_.emplace(stride_.begin(), 0);
    }
  }
  return *this;
}

Tensor Broadcast(const Tensor &in, const Shape &shape,
                 const int ignore_last_dim) {
  Tensor out(in);
  return out.Broadcast(shape, ignore_last_dim);
}

Tensor &Tensor::T() {
  // this function only works for 2d tensors
  CHECK_EQ(shape_.size(), 2u);
  Transpose();
  return *this;
}

// normal transpose without axes
Tensor &Tensor::Transpose() {
  std::reverse(shape_.begin(), shape_.end());
  std::reverse(stride_.begin(), stride_.end());
  return *this;
}

// transpose with axes
Tensor &Tensor::Transpose(const vector<size_t> &axes) {
  CHECK_EQ(axes.size(), shape_.size())
      << "Tranpose axes's length should be equal to shape";

  auto shape = shape_;
  auto stride = stride_;
  shape_.clear();
  stride_.clear();
  for (size_t n = 0; n < axes.size(); ++n) {
    shape_.push_back(shape[axes[n]]);
    stride_.push_back(stride[axes[n]]);
  }
  return *this;
}

// normal transpose without axes
Tensor Transpose(const Tensor &in) {
  Tensor out(in);
  out.Transpose();
  return out;
}

// transpose with axes
Tensor Transpose(const Tensor &in, const vector<size_t> &axes) {
  Tensor out(in);
  out.Transpose(axes);
  return out;
}

Tensor &Tensor::operator=(const Tensor &in) {
  if (block_ != nullptr && block_->DecRefCount() == 0)
    device_->FreeBlock(block_);
  stride_ = in.stride_;
  data_type_ = in.data_type_;
  shape_ = in.shape_;
  device_ = in.device_;
  block_ = in.block();
  if (block_ != nullptr) block_->IncRefCount();
  return *this;
}

Tensor &Tensor::operator=(Tensor &&in) {
  if (block_ != nullptr && block_->DecRefCount() == 0)
    device_->FreeBlock(block_);
  stride_ = std::move(in.stride_);
  data_type_ = in.data_type_;
  shape_ = std::move(in.shape_);
  device_ = in.device_;
  block_ = in.block_;
  in.block_ = nullptr;
  return *this;
}

#define GenUnaryTensorArgMemberFn(op, fn) \
  Tensor &Tensor::op(const Tensor &in) {  \
    Tensor out(*this);                    \
    fn(*this, in, &out);                  \
    return *this;                         \
  }

GenUnaryTensorArgMemberFn(operator+=, Add);
GenUnaryTensorArgMemberFn(operator-=, Sub);
GenUnaryTensorArgMemberFn(operator*=, EltwiseMult);
GenUnaryTensorArgMemberFn(operator/=, Div);

#define GenUnaryScalarArgMemberFn(op, fn) \
  template <typename DType>               \
  Tensor &Tensor::op(const DType x) {     \
    Tensor out(*this);                    \
    fn(*this, x, &out);                   \
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

  Device *dev = nullptr;
  CopyDirection direct;
  std::shared_ptr<Device> src_dev = src.device(), dst_dev = dst->device();
  if (dst_dev->lang() != src_dev->lang()) {
    // let the none cpp device conduct copy op
    if (dst_dev->lang() == kCpp) {
      dev = src_dev.get();
      direct = kDeviceToHost;
    } else if (src_dev->lang() == kCpp) {
      dev = dst_dev.get();
      direct = kHostToDevice;
    } else {
      LOG(FATAL) << "Not support mem copy between Cuda and OpenCL device";
    }
  } else {
    dev = src_dev.get();
    direct = src_dev->lang() == kCpp ? kHostToHost : kDeviceToDevice;
  }

  Tensor &dstRef = *dst;
  dev->Exec(
      [dev, dstRef, src, nBytes, direct, d_offset,
       s_offset](Context *ctx) mutable {
        Block *from = src.block(), *to = dstRef.block();
        dev->CopyDataToFrom(to, from, nBytes, direct, (int)d_offset,
                            (int)s_offset, ctx);
      },
      {src.block()}, {dst->block()}, "CopyDataToFrom");
}

void RepeatDataToFrom(bool broadcast_flag, const vector<size_t> &repeats,
                      int axis, Tensor *dst, const Tensor &src,
                      const size_t num) {
  if (repeats.size() == 1) {
    broadcast_flag = true;
  } else if (repeats.size() > 1) {
    if (axis == Noaxis) {
      LOG(FATAL) << "When repeats parameter is sequence, axis cannot be None";
    }
  }
  for (size_t i = 0; i < repeats.size(); i++) {
    CHECK_GE(repeats[i], 0);
  }
  auto width = SizeOf(src.data_type());
  CHECK_EQ(width, SizeOf(dst->data_type()));
  // size_t nBytes = num * width;
  int chunk = width;
  int axis_shape = 1;
  int shape_outer = 1;
  if (axis == Noaxis) {
    axis_shape = 1;
    shape_outer = Product(src.shape());
  } else {
    for (int i = 0; i < axis; i++) {
      shape_outer *= src.shape()[i];
    }
    axis_shape = src.shape()[axis];
    for (int i = axis + 1; i < static_cast<int>(src.nDim()); i++) {
      chunk *= src.shape()[i];
    }
  }

  Device *dev = nullptr;
  CopyDirection direct;
  std::shared_ptr<Device> src_dev = src.device(), dst_dev = dst->device();
  if (dst_dev->lang() != src_dev->lang()) {
    // let the none cpp device conduct copy op
    if (dst_dev->lang() == kCpp) {
      dev = src_dev.get();
      direct = kDeviceToHost;
    } else if (src_dev->lang() == kCpp) {
      dev = dst_dev.get();
      direct = kHostToDevice;
    } else {
      LOG(FATAL)
          << "Not support mem repeat copy between Cuda and OpenCL device";
    }
  } else {
    dev = src_dev.get();
    direct = src_dev->lang() == kCpp ? kHostToHost : kDeviceToDevice;
  }

  int dst_offset = 0;
  int src_offset = 0;
  Tensor &dstRef = *dst;
  for (int i = 0; i < shape_outer; i++) {
    for (int j = 0; j < axis_shape; j++) {
      int temp = broadcast_flag ? repeats[0] : repeats[j];
      for (int k = 0; k < temp; k++) {
        dev->Exec(
            [dev, dstRef, src, chunk, direct, dst_offset,
             src_offset](Context *ctx) mutable {
              Block *from = src.block(), *to = dstRef.block();
              dev->CopyDataToFrom(to, from, chunk, direct, dst_offset,
                                  src_offset, ctx);
            },
            {src.block()}, {dst->block()}, "CopyDataToFrom");
        dst_offset += chunk;
      }
      src_offset += chunk;
    }
  }
}

//============================================================================
/// typedef DType accroding to type value.
/// DType would be used in the code block __VA_ARGS__.
#define TYPE_SWITCH(type, DType, ...)                               \
  do {                                                              \
    switch (type) {                                                 \
      case kFloat16: {                                              \
        typedef half_float::half DType;                             \
        { __VA_ARGS__ }                                             \
        break;                                                      \
      }                                                             \
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
      case kDouble: {                                               \
        typedef double DType;                                       \
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
      case ((kFloat16 << _SwitchShift) + kCpp): {              \
        typedef half_float::half DType;                        \
        typedef lang::Cpp Lang;                                \
        { __VA_ARGS__ }                                        \
        break;                                                 \
      }                                                        \
      case ((kFloat16 << _SwitchShift) + kCuda): {             \
        typedef half_float::half DType;                        \
        typedef lang::Cuda Lang;                               \
        { __VA_ARGS__ }                                        \
        break;                                                 \
      }                                                        \
      case ((kFloat32 << _SwitchShift) + kCuda): {             \
        typedef float DType;                                   \
        typedef lang::Cuda Lang;                               \
        { __VA_ARGS__ }                                        \
        break;                                                 \
      }                                                        \
      case ((kInt << _SwitchShift) + kCuda): {                 \
        typedef int DType;                                     \
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
      case ((kInt << _SwitchShift) + kCpp): {                  \
        typedef int DType;                                     \
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
float Tensor::l1() const {
  float nrm = 0.0f;
  TYPE_LANG_SWITCH(data_type_, DType, device_->lang(), Lang, {
    device_->Exec(
        [&nrm, this](Context *ctx) {
          DType ret = DType(0);
          Asum<DType, Lang>(*this, &ret, ctx);
          nrm = TypeCast<DType, float>(ret);
        },
        {this->block()}, {}, "l1");
  });
  return nrm / Size();
}

// DEPRECATED use l1()
float Tensor::L1() const { return l1(); }

/// L2 norm, Do not use Nrm2 (name conflict).
float Tensor::l2() const {
  float nrm = 0.0f;
  TYPE_LANG_SWITCH(data_type_, DType, device_->lang(), Lang, {
    device_->Exec(
        [&nrm, this](Context *ctx) { Nrm2<DType, Lang>(*this, &nrm, ctx); },
        {this->block()}, {}, "L1");
  });
  return nrm / Size();
}

// DEPRECATED use l2()
float Tensor::L2() const { return l2(); }

template <typename SType>
void Tensor::SetValue(const SType x) {
  // auto size = Size();
  auto ptr = block_;

  TYPE_LANG_SWITCH(data_type_, DType, device_->lang(), Lang, {
    DType tmp = TypeCast<SType, DType>(x);
    Tensor &thisRef = *this;
    device_->Exec(
        [thisRef, tmp](Context *ctx) mutable {
          Set<DType, Lang>(tmp, &thisRef, ctx);
        },
        {}, {ptr}, "SetValue");
  });
}
template void Tensor::SetValue<float>(const float x);
template void Tensor::SetValue<half_float::half>(const half_float::half x);
template void Tensor::SetValue<int>(const int x);

template <typename SType>
void Tensor::get_value(SType *value, const size_t num) const {
  CHECK(device_ == defaultDevice);
  Tensor t(shape_, device_, data_type_);
  // transform function arrange data in memory considering stride
  singa::Transform(*this, &t);
  auto ptr = static_cast<const SType *>(t.block()->data());
  for (size_t i = 0; i < num; i++) value[i] = ptr[i];
}
template void Tensor::get_value<float>(float *value, const size_t num) const;
template void Tensor::get_value<half_float::half>(half_float::half *value,
                                                  const size_t num) const;
template void Tensor::get_value<int>(int *value, const size_t num) const;

// DEPRECATED
template <typename SType>
void Tensor::GetValue(SType *value, const size_t num) const {
  get_value(value, num);
}
template void Tensor::GetValue<float>(float *value, const size_t num) const;
template void Tensor::GetValue<int>(int *value, const size_t num) const;

#define EltwiseUnaryTensorFn(fn, t, ret)                               \
  do {                                                                 \
    TYPE_LANG_SWITCH(t.data_type(), DType, t.device()->lang(), Lang, { \
      Tensor &retRef = *ret;                                           \
      ret->device()->Exec(                                             \
          [t, retRef](Context *ctx) mutable {                          \
            fn<DType, Lang>(t, &retRef, ctx);                          \
          },                                                           \
          {t.block()}, {ret->block()}, #fn);                           \
    });                                                                \
  } while (0)

#define GenUnaryTensorFn(fn)                             \
  Tensor fn(const Tensor &in) {                          \
    Tensor ret(in.shape(), in.device(), in.data_type()); \
    Tensor *retptr = &ret;                               \
    EltwiseUnaryTensorFn(fn, in, retptr);                \
    return ret;                                          \
  }                                                      \
  void fn(const Tensor &in, Tensor *out) { EltwiseUnaryTensorFn(fn, in, out); }

GenUnaryTensorFn(Abs);
GenUnaryTensorFn(Erf);
GenUnaryTensorFn(Ceil);
GenUnaryTensorFn(Floor);
GenUnaryTensorFn(Round);
GenUnaryTensorFn(RoundE);
GenUnaryTensorFn(Exp);
GenUnaryTensorFn(Log);
GenUnaryTensorFn(ReLU);
GenUnaryTensorFn(Sigmoid);
GenUnaryTensorFn(SoftPlus);
GenUnaryTensorFn(SoftSign);
GenUnaryTensorFn(Sign);
GenUnaryTensorFn(Sqrt);
GenUnaryTensorFn(Square);
GenUnaryTensorFn(Transform);
GenUnaryTensorFn(Cos);
GenUnaryTensorFn(Cosh);
GenUnaryTensorFn(Acos);
GenUnaryTensorFn(Acosh);
GenUnaryTensorFn(Sin);
GenUnaryTensorFn(Sinh);
GenUnaryTensorFn(Asin);
GenUnaryTensorFn(Asinh);
GenUnaryTensorFn(Tan);
GenUnaryTensorFn(Tanh);
GenUnaryTensorFn(Atan);
GenUnaryTensorFn(Atanh);
GenUnaryTensorFn(SoftMax);

// add axis to softmax API according to ONNX specification
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
void SoftMax(const Tensor &in, Tensor *out, int axis) {
  // {a_0, a_1, ..., a_k-1, a_k, ... a_n-1}
  // reshape to
  // { a_0 * a_1 * ... a_k-1, a_k * ... a_n-1 }

  // assert axis \in {-r, r-1}
  CHECK_LE(axis, (int)in.shape().size() - 1);
  CHECK_GE(axis, -1 * (int)in.nDim());

  Shape original_shape = in.shape();
  if (axis < 0) axis = in.shape().size() + axis;

  Shape coerced_shape = {1, 1};
  for (std::size_t i = 0, max = in.shape().size(); i != max; ++i) {
    if (i < axis)
      coerced_shape[0] *= in.shape()[i];
    else
      coerced_shape[1] *= in.shape()[i];
  }
  Tensor in_reshaped = Reshape(in, coerced_shape);
  out->Reshape(coerced_shape);

  // optimise by minus x - x.max()
  auto in_max = RowMax(in_reshaped);
  in_max.Reshape({coerced_shape[0], 1});
  in_reshaped = in_reshaped - in_max;

  SoftMax(in_reshaped, out);

  out->Reshape(original_shape);
}

Tensor SoftMax(const Tensor &in, int axis) {
  Tensor ret(in.shape(), in.device(), in.data_type());
  auto *retptr = &ret;
  SoftMax(in, retptr, axis);
  return ret;
}
void SoftMaxBackward(const Tensor &in, Tensor *out, int axis,
                     const Tensor &fdout) {
  // {a_0, a_1, ..., a_k-1, a_k, ... a_n-1}
  // reshape to
  // { a_0 * a_1 * ... a_k-1, a_k * ... a_n-1 }

  // assert axis \in {-r, r-1}
  CHECK_LE(axis, (int)in.shape().size() - 1);
  CHECK_GE(axis, -1 * (int)in.nDim());

  Shape original_shape = in.shape();
  if (axis < 0) axis = in.shape().size() + axis;

  Shape coerced_shape = {1, 1};
  for (std::size_t i = 0, max = in.shape().size(); i != max; ++i) {
    if (i < axis)
      coerced_shape[0] *= in.shape()[i];
    else
      coerced_shape[1] *= in.shape()[i];
  }

  Tensor in_reshaped = Reshape(in, coerced_shape);
  out->Reshape(coerced_shape);

  do {
    TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
      Tensor &outRef = *out;
      out->device()->Exec(
          [in, outRef, fdout](Context *ctx) mutable {
            SoftMaxBackward<DType, Lang>(in, &outRef, fdout, ctx);
          },
          {in.block(), fdout.block()}, {out->block()}, "SoftmaxBackward");
    });
  } while (0);

  out->Reshape(original_shape);
}

Tensor SoftMaxBackward(const Tensor &in, int axis, const Tensor &fdout) {
  Tensor ret(in.shape(), in.device(), in.data_type());
  auto *retptr = &ret;
  SoftMaxBackward(in, retptr, axis, fdout);
  return ret;
}

#define EltwiseBinaryTensorFn(fn, lhs, rhs, ret)                           \
  do {                                                                     \
    TYPE_LANG_SWITCH(lhs.data_type(), DType, lhs.device()->lang(), Lang, { \
      CHECK_EQ(sizeof(DType), SizeOf(rhs.data_type()))                     \
          << "lhs dtype size" << sizeof(DType) << " rhs dtype size"        \
          << SizeOf(rhs.data_type());                                      \
      Tensor &retRef = *ret;                                               \
      ret->device()->Exec(                                                 \
          [lhs, rhs, retRef](Context *ctx) mutable {                       \
            fn<DType, Lang>(lhs, rhs, &retRef, ctx);                       \
          },                                                               \
          {lhs.block(), rhs.block()}, {ret->block()}, #fn);                \
    });                                                                    \
  } while (0)

#define GenBinaryTensorFn(op, fn)                                            \
  Tensor op(const Tensor &lhs, const Tensor &rhs) {                          \
    if (lhs.shape() != rhs.shape()) {                                        \
      if (lhs.data_type() == kFloat32 && rhs.data_type() == kFloat32) {      \
        auto lhs_ = Broadcast(lhs, rhs.shape());                             \
        auto rhs_ = Broadcast(rhs, lhs.shape());                             \
        Tensor ret(lhs_.shape(), lhs.device(), lhs.data_type());             \
        fn(lhs_, rhs_, &ret);                                                \
        return ret;                                                          \
      } else {                                                               \
        /* lhs tensor and rhs tensor are not both in float, cast to float */ \
        Tensor tmp_lhs = lhs.Clone().AsType(kFloat32);                       \
        Tensor tmp_rhs = rhs.Clone().AsType(kFloat32);                       \
        tmp_lhs = Broadcast(tmp_lhs, tmp_rhs.shape());                       \
        tmp_rhs = Broadcast(tmp_rhs, tmp_lhs.shape());                       \
        Tensor ret(tmp_lhs.shape(), tmp_lhs.device(), tmp_lhs.data_type());  \
        fn(tmp_lhs, tmp_rhs, &ret);                                          \
        /* if lhs and rhs are both int, cast back to int */                  \
        if (lhs.data_type() == kInt && rhs.data_type() == kInt)              \
          return ret.Clone().AsType(kInt);                                   \
        return ret;                                                          \
      }                                                                      \
    } else {                                                                 \
      if (lhs.data_type() == kFloat32 && rhs.data_type() == kFloat32) {      \
        Tensor ret(lhs.shape(), lhs.device(), lhs.data_type());              \
        fn(lhs, rhs, &ret);                                                  \
        return ret;                                                          \
      } else {                                                               \
        /* lhs tensor and rhs tensor are not both in float, cast to float */ \
        Tensor tmp_lhs = lhs.Clone().AsType(kFloat32);                       \
        Tensor tmp_rhs = rhs.Clone().AsType(kFloat32);                       \
        Tensor ret(tmp_lhs.shape(), tmp_lhs.device(), tmp_lhs.data_type());  \
        fn(tmp_lhs, tmp_rhs, &ret);                                          \
        /* if lhs and rhs are both int, cast back to int */                  \
        if (lhs.data_type() == kInt && rhs.data_type() == kInt)              \
          return ret.Clone().AsType(kInt);                                   \
        return ret;                                                          \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  void fn(const Tensor &lhs, const Tensor &rhs, Tensor *ret) {               \
    CHECK_EQ(lhs.device(), ret->device());                                   \
    CHECK_EQ(rhs.device(), ret->device());                                   \
    if (lhs.shape() != rhs.shape()) {                                        \
      auto lhs_ = Broadcast(lhs, rhs.shape());                               \
      auto rhs_ = Broadcast(rhs, lhs.shape());                               \
      CHECK(lhs_.shape() == ret->shape());                                   \
      EltwiseBinaryTensorFn(fn, lhs_, rhs_, ret);                            \
    } else {                                                                 \
      CHECK(lhs.shape() == ret->shape());                                    \
      EltwiseBinaryTensorFn(fn, lhs, rhs, ret);                              \
    }                                                                        \
  }

// boradcasting operations:
// https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
GenBinaryTensorFn(operator+, Add);
GenBinaryTensorFn(operator-, Sub);
GenBinaryTensorFn(operator*, EltwiseMult);
GenBinaryTensorFn(operator/, Div);
GenBinaryTensorFn(Pow, Pow);
GenBinaryTensorFn(operator<, LT);
GenBinaryTensorFn(operator<=, LE);
GenBinaryTensorFn(operator>, GT);
GenBinaryTensorFn(operator>=, GE);
GenBinaryTensorFn(operator==, EQ);
GenBinaryTensorFn(ReLUBackward, ReLUBackward);

#define EltwiseTensorScalarFn(fn, t, x, ret)                           \
  do {                                                                 \
    TYPE_LANG_SWITCH(t.data_type(), DType, t.device()->lang(), Lang, { \
      DType tmp_x = TypeCast<SType, DType>(x);                         \
      Tensor &retRef = *ret;                                           \
      ret->device()->Exec(                                             \
          [t, tmp_x, retRef](Context *ctx) mutable {                   \
            fn<DType, Lang>(t, tmp_x, &retRef, ctx);                   \
          },                                                           \
          {t.block()}, {ret->block()}, #fn);                           \
    });                                                                \
  } while (0)

#define GenTensorScalarFn(op, fn)                                          \
  template <typename SType>                                                \
  Tensor op(const Tensor &in, const SType x) {                             \
    if (in.data_type() == kFloat32 && std::is_same<SType, float>::value) { \
      Tensor ret(in.shape(), in.device(), in.data_type());                 \
      fn(in, x, &ret);                                                     \
      return ret;                                                          \
    } else if (in.data_type() == kFloat32) {                               \
      Tensor ret(in.shape(), in.device(), in.data_type());                 \
      float tmp_x = x;                                                     \
      fn(in, tmp_x, &ret);                                                 \
      return ret;                                                          \
    } else {                                                               \
      /* tensor and scalar are not both in float, cast to float */         \
      Tensor tmp_in = in.Clone().AsType(kFloat32);                         \
      float tmp_x = x;                                                     \
      Tensor ret(tmp_in.shape(), tmp_in.device(), tmp_in.data_type());     \
      fn(tmp_in, tmp_x, &ret);                                             \
      /* if tensor and scalar are both int, cast back to int */            \
      if (in.data_type() == kInt && std::is_same<SType, int>::value)       \
        return ret.Clone().AsType(kInt);                                   \
      return ret;                                                          \
    }                                                                      \
  }                                                                        \
  template <typename SType>                                                \
  void fn(const Tensor &in, const SType x, Tensor *ret) {                  \
    EltwiseTensorScalarFn(fn, in, x, ret);                                 \
  }                                                                        \
  template Tensor op<float>(const Tensor &in, const float x);              \
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
GenTensorScalarFn(operator==, EQ);

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
    DType tmp_alpha = TypeCast<SType, DType>(alpha);
    Tensor &outRef = *out;
    in.device()->Exec(
        [tmp_alpha, in, outRef](Context *ctx) mutable {
          Div<DType, Lang>(tmp_alpha, in, &outRef, ctx);
        },
        {in.block()}, {out->block()}, "Div");
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
  } else if (axis == 1) {
    return Sum(M, 1) / (1.0f * M.shape(1));
  } else {
    LOG(FATAL) << "Not currently support Sum over axis = " << axis;
  }
}
// TODO(wangwei) conside async exec
template <>
float Sum<float>(const Tensor &in) {
  float s = 0.0f;
  Tensor one(in.shape(), in.device(), in.data_type());
  one.SetValue(1.0f);
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    one.device()->Exec(
        // cannot use this sum function in computational graph
        [in, one, &s](Context *ctx) mutable {
          DType ret = DType(0);
          Dot<DType, Lang>(in, one, &ret, ctx);
          s = ret;
        },
        {in.block(), one.block()}, {}, "Sum");
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
    Tensor out = Tensor(Shape{M.shape(0)}, M.device(), M.data_type());
    SumColumns(M, &out);
    return out;
  }
}

Tensor SumAll(const Tensor &in) {
  Tensor out({(size_t)1}, in.device(), in.data_type());
  Tensor one(in.shape(), in.device(), in.data_type());
  one.SetValue(1.0f);
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    one.device()->Exec(
        [in, one, out](Context *ctx) mutable {
          Dot<DType, Lang>(in, one, &out, ctx);
        },
        {in.block(), one.block()}, {out.block()}, "SumAll");
  });
  return out;
}

Tensor RowMax(const Tensor &in) {
  Tensor ret({in.shape(0)}, in.device(), in.data_type());
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    in.device()->Exec(
        [in, ret](Context *ctx) mutable {
          // size_t nrow = 1;
          // if (in.nDim() > 1) nrow = in.shape(0);
          // size_t ncol = in.Size() / nrow;
          RowMax<DType, Lang>(in, &ret, ctx);
        },
        {in.block()}, {ret.block()}, "RowMax");
  });
  return ret;
}

void AddColumn(const Tensor &v, Tensor *M) { AddColumn(1, 1, v, M); }
/// Add column 'v' onto each column of matrix M;
template <typename SType>
void AddColumn(const SType alpha, const SType beta, const Tensor &v,
               Tensor *M) {
  if (M->transpose()) {
    Tensor X(Transpose(*M));
    AddRow(v, &X);
  } else {
    CHECK_EQ(M->nDim(), 2u);
    // CHECK_EQ(v.nDim(), 1u); (chonho) shape of v is 2-element tuple
    size_t nb_row = M->shape(0), nb_col = M->shape(1);
    CHECK_EQ(nb_row, v.Size());

    Tensor one(Shape{1, nb_col}, M->device(), M->data_type());
    one.SetValue(1.0f);  // TODO(wangwei) cast type
    Tensor vmat(Reshape(v, Shape{nb_row, 1}));
    Mult(alpha, vmat, one, beta, M);
  }
}
template void AddColumn(const float alpha, const float beta, const Tensor &v,
                        Tensor *M);

void AddRow(const Tensor &v, Tensor *M) { AddRow(1, 1, v, M); }

/// Add row 'v' by each column of matrix M; write results into 'out'
template <typename SType>
void AddRow(const SType alpha, const SType beta, const Tensor &v, Tensor *M) {
  if (M->transpose()) {
    Tensor X(Transpose(*M));
    AddColumn(v, &X);
  } else {
    CHECK_EQ(M->nDim(), 2u);
    // CHECK_EQ(v.nDim(), 1u); (chonho) shape of v is 2-element tuple
    size_t nb_row = M->shape(0), nb_col = M->shape(1);
    CHECK_EQ(nb_col, v.Size());

    Tensor one(Shape{nb_row, 1}, M->device(), M->data_type());
    one.SetValue(1.0f);
    Tensor vmat(Reshape(v, Shape{1, nb_col}));
    Mult(alpha, one, vmat, beta, M);
  }
}
template void AddRow(const float alpha, const float beta, const Tensor &v,
                     Tensor *M);

/// Divide column 'v' by each column of matrix M; write results into 'out'
void DivColumn(const Tensor &v, Tensor *M) {
  Tensor inv;
  TYPE_SWITCH(v.data_type(), DType, { inv = Div(DType(1), v); });
  MultColumn(inv, M);
}

Tensor ConcatOn(const std::vector<Tensor> &in, int axis) {
  vector<Tensor> tmp;
  Shape out_shape = in[0].shape();
  size_t dim = in[0].shape().size();
  // CHECK_GE(dim, 2u) << " Only work for tensor of dim >=2 ";
  size_t size = in[0].Size() / in[0].shape(axis);
  size_t new_size = 0u;
  for (const auto &t : in) {
    CHECK_EQ(dim, t.shape().size()) << "All tensors should have the same dim";
    CHECK_EQ(size, t.Size() / t.shape(axis))
        << "The size of all axis should "
        << " be the same except the concatenated axis";
    new_size += t.shape(axis);
  }
  out_shape[axis] = new_size;
  if (axis == 0) {
    size_t nrow = 0;
    for (const auto &t : in) {
      nrow += t.shape(0);
      tmp.push_back(Reshape(t, {t.shape(0), t.Size() / t.shape(0)}));
    }
    auto ret = ConcatenateRows(tmp);
    ret.Reshape(out_shape);
    return ret;
  } else {
    for (const auto &t : in) {
      size_t nrow = 1;
      for (int i = 0; i < axis; i++) nrow *= t.shape(i);
      tmp.push_back(Reshape(t, {nrow, t.Size() / nrow}));
    }
    auto ret = ConcatenateColumns(tmp);
    ret.Reshape(out_shape);
    return ret;
  }
}

Tensor ConcatenateRows(const vector<Tensor> &in) {
  size_t nrow = 0, ncol = 0;
  CHECK(in.size());
  for (const auto &x : in) {
    CHECK(!x.transpose());
    CHECK_EQ(x.nDim(), 2u);
    nrow += x.shape(0);
    if (ncol == 0)
      ncol = x.shape(1);
    else
      CHECK_EQ(ncol, x.shape(1));
  }
  Tensor out(Shape{nrow, ncol}, in.at(0).device(), in.at(0).data_type());
  size_t dst_offset = 0;
  for (const auto &x : in) {
    CopyDataToFrom(&out, x, x.Size(), dst_offset, 0);
    dst_offset += x.Size();
  }
  return out;
}
Tensor ConcatRows(const vector<Tensor> &in) { return ConcatenateRows(in); }
// TODO(wangwei) add a copypatch function for improve the efficiency on GPU.
Tensor ConcatenateColumns(const vector<Tensor> &in) {
  size_t nrow = 0, ncol = 0;
  CHECK(in.size());
  for (const auto &x : in) {
    CHECK(!x.transpose());
    CHECK_EQ(x.nDim(), 2u);
    ncol += x.shape(1);
    if (nrow == 0)
      nrow = x.shape(0);
    else
      CHECK_EQ(nrow, x.shape(0));
  }
  Tensor out(Shape{nrow, ncol}, in.at(0).device(), in.at(0).data_type());
  for (size_t row = 0; row < nrow; row++) {
    size_t dst_offset = row * ncol;
    for (const auto &x : in) {
      size_t src_offset = row * x.shape(1);
      CopyDataToFrom(&out, x, x.shape(1), dst_offset, src_offset);
      dst_offset += x.shape(1);
    }
    CHECK_EQ(dst_offset, row * ncol + ncol);
  }
  return out;
}
Tensor ConcatColumns(const vector<Tensor> &in) {
  return ConcatenateColumns(in);
}

Tensor CopyRows(const Tensor &in, const size_t start, const size_t end) {
  CHECK_LT(start, end);
  CHECK_GE(in.shape(0), end) << "Tensor size must >= end";
  Shape s = in.shape();
  s[0] = end - start;
  size_t sample_size = in.Size() / in.shape(0);
  Tensor out(s, in.device(), in.data_type());
  CopyDataToFrom(&out, in, out.Size(), 0, start * sample_size);
  return out;
}

Tensor SliceOn(const Tensor &in, const size_t start, const size_t end,
               int axis) {
  Shape out_shape = in.shape();
  out_shape[axis] = end - start;
  if (axis == 0) {
    auto ret = SliceRows(Reshape(in, {in.shape(0), in.Size() / in.shape(0)}),
                         start, end);
    ret.Reshape(out_shape);
    return ret;
  } else {
    size_t nrow = 1;
    for (int i = 0; i < axis; i++) nrow *= in.shape(i);
    auto suffix = in.Size() / nrow / in.shape(axis);
    auto ret = SliceColumns(Reshape(in, {nrow, in.Size() / nrow}),
                            start * suffix, end * suffix);
    ret.Reshape(out_shape);
    return ret;
  }
}

Tensor SliceRows(const Tensor &in, const size_t start, const size_t end) {
  return CopyRows(in, start, end);
}

Tensor CopyColumns(const Tensor &in, const size_t start, const size_t end) {
  CHECK_EQ(in.nDim(), 2u);
  CHECK_LT(start, end);
  CHECK_GE(in.shape(1), end);
  Shape s{in.shape(0), end - start};
  Tensor out(s, in.device(), in.data_type());
  for (size_t row = 0; row < out.shape(0); row++) {
    size_t src_offset = row * in.shape(1) + start;
    size_t dst_offset = row * out.shape(1);
    CopyDataToFrom(&out, in, end - start, dst_offset, src_offset);
  }
  return out;
}

Tensor SliceColumns(const Tensor &in, const size_t start, const size_t end) {
  return CopyColumns(in, start, end);
}

/// Divide row 'v' by each row of matrix M; write results into 'out'
void DivRow(const Tensor &v, Tensor *M) {
  Tensor inv;
  TYPE_SWITCH(v.data_type(), DType, { inv = Div(DType(1), v); });
  MultRow(inv, M);
}

/// Multiply column 'v' and each column of matrix M; write results into 'out'
void MultColumn(const Tensor &v, Tensor *M) {
  // CHECK(!M->transpose()) << "Not supported yet";
  CHECK_EQ(M->nDim(), 2u);
  // CHECK_EQ(v.nDim(), 1u); (chonho) shape of v is 2-element tuple
  CHECK_EQ(v.Size(), M->shape(0));
  CheckDataTypeAndLang(*M, v);
  TYPE_LANG_SWITCH(v.data_type(), DType, v.device()->lang(), Lang, {
    Tensor &MRef = *M;
    v.device()->Exec(
        [MRef, v](Context *ctx) mutable {
          DGMM<DType, Lang>(false, MRef, v, &MRef, ctx);
        },
        {M->block(), v.block()}, {M->block()}, "MultColumn");
  });
}

/// Multiply row 'v' with each row of matrix M; write results into 'out'
void MultRow(const Tensor &v, Tensor *M) {
  // CHECK(!M->transpose()) << "Not supported yet";
  CHECK_EQ(M->nDim(), 2u);
  // CHECK_EQ(v.nDim(), 1u); (chonho) shape of v is 2-element tuple
  CHECK_EQ(v.Size(), M->shape(1));
  CheckDataTypeAndLang(*M, v);
  TYPE_LANG_SWITCH(v.data_type(), DType, v.device()->lang(), Lang, {
    Tensor &MRef = *M;
    v.device()->Exec(
        [MRef, v](Context *ctx) mutable {
          DGMM<DType, Lang>(true, MRef, v, &MRef, ctx);
        },
        {M->block(), v.block()}, {M->block()}, "MultRow");
  });
}

void SubColumn(const Tensor &v, Tensor *M) { AddColumn(-1, 1, v, M); }

void SubRow(const Tensor &v, Tensor *M) { AddRow(-1, 1, v, M); }

void SumColumns(const Tensor &M, Tensor *v) {
  if (M.transpose()) {
    Tensor X = Transpose(M);
    SumRows(X, v);
  } else {
    CHECK_EQ(M.nDim(), 2u);
    // CHECK_EQ(v->nDim(), 1u); (chonho) shape of v is 2-element tuple
    size_t nb_row = M.shape().at(0), nb_col = M.shape().at(1);
    CHECK_EQ(nb_row, v->Size());

    Tensor one(Shape{nb_col}, M.device(), M.data_type());
    one.SetValue(1.0f);  // TODO(wangwei) cast type
    Mult(M, one, v);
  }
}
void SumRows(const Tensor &M, Tensor *v) {
  if (M.transpose()) {
    Tensor X = Transpose(M);
    SumColumns(X, v);
  } else {
    CHECK_EQ(M.nDim(), 2u);
    // CHECK_EQ(v->nDim(), 1u); (chonho) shape of v is 2-element tuple
    size_t nb_row = M.shape(0), nb_col = M.shape(1);
    CHECK_EQ(nb_col, v->Size());

    Tensor one(Shape{nb_row}, M.device(), M.data_type());
    one.SetValue(1.0f);  // TODO(wangwei) cast type
    Tensor X = Transpose(M);
    Mult(X, one, v);
  }
}
// ====================Random operations=====================================
template <typename SType>
void Bernoulli(const SType p, Tensor *out) {
  TYPE_LANG_SWITCH(out->data_type(), DType, out->device()->lang(), Lang, {
    auto prob = TypeCast<SType, DType>(p);
    Tensor &outRef = *out;
    out->device()->Exec(
        [prob, outRef](Context *ctx) mutable {
          Bernoulli<DType, Lang>(prob, &outRef, ctx);
        },
        {}, {out->block()}, "Bernoulli", true);
  });
}

template void Bernoulli<float>(const float p, Tensor *out);

template <typename SType>
void Uniform(const SType low, const SType high, Tensor *out) {
  TYPE_LANG_SWITCH(out->data_type(), DType, out->device()->lang(), Lang, {
    auto l = TypeCast<SType, DType>(low);
    auto h = TypeCast<SType, DType>(high);
    Tensor &outRef = *out;
    out->device()->Exec(
        [l, h, outRef](Context *ctx) mutable {
          Uniform<DType, Lang>(l, h, &outRef, ctx);
        },
        {}, {out->block()}, "Uniform", true);
  });
}

template void Uniform<float>(const float low, const float high, Tensor *out);

template <typename SType>
void Gaussian(const SType mean, const SType std, Tensor *out) {
  TYPE_LANG_SWITCH(out->data_type(), DType, out->device()->lang(), Lang, {
    auto m = TypeCast<SType, DType>(mean);
    auto s = TypeCast<SType, DType>(std);
    Tensor &outRef = *out;
    out->device()->Exec(
        [m, s, outRef](Context *ctx) mutable {
          Gaussian<DType, Lang>(m, s, &outRef, ctx);
        },
        {}, {out->block()}, "Gaussian", true);
  });
}
template void Gaussian<float>(const float mean, const float std, Tensor *out);

// ================Blas operations============================================

template <typename SType>
void Axpy(const SType alpha, const Tensor &in, Tensor *out) {
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    auto a = TypeCast<SType, DType>(alpha);
    Tensor &outRef = *out;
    Tensor fake(*out);
    out->device()->Exec(
        [a, in, outRef, fake](Context *ctx) mutable {
          Axpy<DType, Lang>(a, in, &outRef, ctx);
        },
        {in.block(), out->block()}, {out->block()}, "Axpy");
  });
}

template void Axpy<float>(const float alpha, const Tensor &in, Tensor *out);

void Axpy(const Tensor &alpha, const Tensor &in, Tensor *out) {
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    Tensor fake(*out);
    Tensor &outRef = *out;
    out->device()->Exec(
        [alpha, in, outRef, fake](Context *ctx) mutable {
          Axpy<DType, Lang>(alpha, in, &outRef, ctx);
        },
        {alpha.block(), in.block(), out->block()}, {out->block()}, "Axpy");
  });
}

Tensor Mult(const Tensor &A, const Tensor &B) {
  auto A_ = Broadcast(A, B.shape(), 2);
  auto B_ = Broadcast(B, A.shape(), 2);

  Shape s = A_.shape();
  s.pop_back();
  s.push_back(B.shape(B.nDim() - 1));

  Tensor out(s, A.device(), A.data_type());
  Mult(A_, B_, &out);
  return out;
}

void Mult(const Tensor &A, const Tensor &B, Tensor *out) {
  Mult(1.0f, A, B, 0.0f, out);
}

template <typename SType>
void Mult(const SType alpha, const Tensor &A, const Tensor &B, const SType beta,
          Tensor *C) {
  Tensor fakeC;
  vector<Block *> read_blocks = {A.block(), B.block()};
  if (beta) {
    fakeC = *C;
    read_blocks.push_back(C->block());
  }
  if (B.nDim() == 1u) {
    CHECK_EQ(A.shape().size(), 2u);
    TYPE_LANG_SWITCH(A.data_type(), DType, A.device()->lang(), Lang, {
      auto a = TypeCast<SType, DType>(alpha);
      auto b = TypeCast<SType, DType>(beta);
      Tensor &CRef = *C;
      C->device()->Exec(
          [a, A, b, B, CRef, fakeC](Context *ctx) mutable {
            GEMV<DType, Lang>(a, A, B, b, &CRef, ctx);
          },
          read_blocks, {C->block()}, "GEMV");
    });
  } else if (B.nDim() == 2u) {
    CHECK_EQ(A.shape().size(), 2u);
    CHECK(!C->transpose());
    TYPE_LANG_SWITCH(A.data_type(), DType, A.device()->lang(), Lang, {
      auto a = TypeCast<SType, DType>(alpha);
      auto b = TypeCast<SType, DType>(beta);
      Tensor &CRef = *C;
      C->device()->Exec(
          [a, A, b, B, CRef, fakeC](Context *ctx) mutable {
            GEMM<DType, Lang>(a, A, B, b, &CRef, ctx);
          },
          read_blocks, {C->block()}, "GEMM");
    });
  } else if (B.nDim() == 3u || B.nDim() == 4u) {
    CHECK_EQ(A.shape().size(), B.shape().size());
    CHECK(!C->transpose());
    TYPE_LANG_SWITCH(A.data_type(), DType, A.device()->lang(), Lang, {
      auto a = TypeCast<SType, DType>(alpha);
      auto b = TypeCast<SType, DType>(beta);

      Tensor A_tmp;
      Tensor B_tmp;

      if (A.transpose() || A.broadcasted()) {
        A_tmp = Tensor(A.shape(), A.device(), A.data_type());
        singa::Transform(A, &A_tmp);
      } else {
        A_tmp = A;
      }

      if (B.transpose() || B.broadcasted()) {
        B_tmp = Tensor(B.shape(), B.device(), B.data_type());
        singa::Transform(B, &B_tmp);
      } else {
        B_tmp = B;
      }

      // batch GEMM should have same batch size
      CHECK_EQ(A_tmp.shape(0), B_tmp.shape(0));
      if (B.nDim() == 4u) CHECK_EQ(A_tmp.shape(1), B_tmp.shape(1));

      Tensor &CRef = *C;
      C->device()->Exec(
          [a, A_tmp, b, B_tmp, CRef, fakeC](Context *ctx) mutable {
            GEMMBatched<DType, Lang>(a, A_tmp, B_tmp, b, &CRef, ctx);
          },
          read_blocks, {C->block()}, "GEMMBatched");
    });
  } else {
    LOG(FATAL) << "Un-supported tensor dimentions " << A.nDim() << "d matmul "
               << B.nDim() << "d\n";
  }
}

// ************************
// Misc.
// ************************
Tensor CrossEntropyFwd(const Tensor &p, const Tensor &t) {
  Tensor loss({p.shape(0)}, p.device(), p.data_type());
  ComputeCrossEntropy(p, t, &loss);
  return loss;
}

Tensor SoftmaxCrossEntropyBwd(const Tensor &p, const Tensor &t) {
  Tensor g = p.Clone();
  SoftmaxCrossEntropyBwd(t, &g);
  return g;
}

void ComputeCrossEntropy(const Tensor &p, const Tensor &t, Tensor *loss) {
  CHECK_LE(p.nDim(), 2u);
  CHECK_LE(t.nDim(), 2u);
  size_t batchsize = 1;
  if (p.nDim() == 2u) batchsize = p.shape(0);
  size_t dim = p.Size() / batchsize;
  TYPE_LANG_SWITCH(p.data_type(), DType, p.device()->lang(), Lang, {
    Tensor &lossRef = *loss;
    p.device()->Exec(
        [batchsize, dim, t, p, lossRef](Context *ctx) mutable {
          bool int_target = t.Size() == batchsize;
          ComputeCrossEntropy<DType, Lang>(int_target, batchsize, dim, p, t,
                                           &lossRef, ctx);
        },
        {p.block(), t.block()}, {loss->block()}, "ComputeCrossEntropy");
  });
}

void SoftmaxCrossEntropyBwd(const Tensor &t, Tensor *p) {
  CHECK_LE(p->nDim(), 2u);
  CHECK_LE(t.nDim(), 2u);
  size_t batchsize = 1;
  if (p->nDim() == 2u) batchsize = p->shape(0);
  size_t dim = p->Size() / batchsize;
  TYPE_LANG_SWITCH(p->data_type(), DType, p->device()->lang(), Lang, {
    Tensor &pRef = *p;
    Tensor pFake(*p);  // just add a ref count
    p->device()->Exec(
        [batchsize, dim, t, pRef, pFake, p](Context *ctx) mutable {
          bool int_target = t.Size() == batchsize;
          SoftmaxCrossEntropyBwd<DType, Lang>(int_target, batchsize, dim, pRef,
                                              t, &pRef, ctx);
        },
        {p->block(), t.block()}, {p->block()}, "SoftmaxCrossEntropyBackward");
  });
}

Tensor &Tensor::Contiguous() {
  if (transpose()) {
    Tensor t(shape_, device_, data_type_);
    singa::Transform(*this, &t);
    std::swap(t.block_, block_);
  }
  return *this;
}

Tensor Contiguous(const Tensor &in) {
  Tensor out(in);
  return out.Contiguous();
}

// if tensor is not transposed yet, we change the shape and generate new stride
// if tensor is already transposed, we reallocate the memory and generate stride
Tensor &Tensor::Reshape(const Shape &shape) {
  // Check original volumn with the new one
  // do not use Product(shape_) due to stride 0 from broadcasting.
  // printf("reshape loc b\n");
  CHECK_EQ(Product(shape), Size());
  if (transpose()) {
    Tensor t(shape_, device_, data_type_);
    singa::Transform(*this, &t);
    std::swap(t.block_, block_);
    shape_ = shape;
  } else {
    shape_ = shape;
  }
  generate_stride();
  // printf("reshape loc c\n");
  return *this;
}

Tensor Reshape(const Tensor &in, const Shape &s) {
  // printf("reshape loc a\n");
  Tensor out(in);
  return out.Reshape(s);
}

}  // namespace singa
