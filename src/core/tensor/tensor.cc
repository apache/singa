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

#define Noaxis 9999

namespace singa {

Tensor::~Tensor() {
  if (block_ != nullptr && block_->DecRefCount() == 0)
    device_->FreeBlock(block_);
  block_ = nullptr;
}

Tensor::Tensor() {
  device_ = defaultDevice;
  strides_ = {1};
}

//non-strided constructors
Tensor::Tensor(const Shape &shape, DataType dtype)
  : data_type_(dtype), device_(defaultDevice), shape_(shape) {
  size_t size = Product(shape_) * SizeOf(data_type_);
  if (size)
    block_ = device_->NewBlock((int)size);
  generate_strides();
}
Tensor::Tensor(Shape &&shape, DataType dtype)
  : data_type_(dtype), device_(defaultDevice), shape_(shape) {
  size_t size = Product(shape_) * SizeOf(data_type_);
  if (size)
    block_ = device_->NewBlock((int)size);
  generate_strides();
}

//non-strided constructors with device
Tensor::Tensor(const Shape &shape, std::shared_ptr<Device> device,
               DataType dtype)
  : data_type_(dtype), device_(device), shape_(shape) {
  size_t size = Product(shape_) * SizeOf(data_type_);
  if (size)
    block_ = device_->NewBlock((int)size);
  generate_strides();
}
Tensor::Tensor(Shape &&shape, std::shared_ptr<Device> device, DataType dtype)
  : data_type_(dtype), device_(device), shape_(shape) {
  size_t size = Product(shape_) * SizeOf(data_type_);
  if (size)
    block_ = device_->NewBlock((int)size);
  generate_strides();
}


Tensor::Tensor(const Tensor &in)
  : //transpose_(in.transpose_),
    data_type_(in.data_type_),
    device_(in.device_),
    block_(in.block()),
    shape_(in.shape_),
    strides_(in.strides_) {
  if (block_ != nullptr)
    block_->IncRefCount();
}

//strided constructor taking in a tensor, shape and strides
Tensor::Tensor(const Tensor &in, Shape &new_shape, vector<int> &new_strides)
  : //transpose_(in.transpose_),
    data_type_(in.data_type_),
    device_(in.device_),
    block_(in.block()),
    shape_(new_shape),
    strides_(new_strides) {
  if (block_ != nullptr)
    block_->IncRefCount();
}

Tensor::Tensor(Tensor &&in)
  : //transpose_(in.transpose_),
    data_type_(in.data_type_),
    device_(in.device_),
    shape_(std::move(in.shape_)),
    strides_(in.strides_) {
  block_ = in.block_;
  in.block_ = nullptr;
}


void Tensor::SetBlock(Block *block) {
  LOG(WARNING) << "Pls avoid using this function, which may have side-effect.";
  if (block_ != nullptr)
    if (block_->DecRefCount()) device_->FreeBlock(block_);
  block_ = block;
}

void Tensor::ResetLike(const Tensor &in) {
  if (block_ == nullptr || device_ != in.device_ || MemSize() != in.MemSize()) {
    if (block_ != nullptr && block_->DecRefCount() == 0)
      device_->FreeBlock(block_);
    device_ = in.device_;
    data_type_ = in.data_type_;
    block_ = device_->NewBlock((int)in.MemSize());
  }
  shape_ = in.shape_;
  strides_ = in.strides_;
}

Tensor Tensor::Reshape(const Shape &shape) {
  if (strides_.size() == 0)
    strides_.push_back(1);

  if (Product(shape_) != Product(shape)) {
    if (block_ != nullptr && block_->DecRefCount() == 0)
      device_->FreeBlock(block_);
    block_ = device_->NewBlock((int)(Product(shape) * SizeOf(data_type_)));
    shape_ = shape;
    generate_strides();
    return *this;

  } else if (transpose()) {
    Tensor t(shape_, device_, data_type_);
    t.block_ = t.device()->NewBlock((int)(Product(shape) * SizeOf(data_type_)));
    singa::Transform(*this, &t);
    t.shape_ = shape;
    return t;
 }

  shape_ = shape;
  generate_strides();
  Tensor t(shape, device_, data_type_);
  t.block_ = block_;
  t.block_->IncRefCount();
  return t;
}

Tensor Tensor::Reshape(Shape &&shape) {
  if (strides_.size() == 0)
    strides_.push_back(1);

  if (Product(shape_) != Product(shape)) {
    if (block_ != nullptr && block_->DecRefCount() == 0)
      device_->FreeBlock(block_);
    block_ = device_->NewBlock((int)(Product(shape) * SizeOf(data_type_)));
    shape_ = std::move(shape);
    generate_strides();
    return *this;

  } else if (transpose()) {
    Tensor t(shape_, device_, data_type_);
    t.block_ = t.device()->NewBlock((int)(Product(shape) * SizeOf(data_type_)));
    singa::Transform(*this, &t);
    t.shape_ = shape;
    return t;
 }

  shape_ = shape;
  generate_strides();
  Tensor t(shape, device_, data_type_);
  t.block_ = block_;
  t.block_->IncRefCount();
  return t;
}

void Tensor::AsType(const DataType type) {
  if (data_type_ != type) {
    if (block_ != nullptr && block_->DecRefCount() == 0)
      device_->FreeBlock(block_);
    block_ = device_->NewBlock((int)(Product(shape_) * SizeOf(type)));
    data_type_ = type;
  }
}

void Tensor::ToDevice(std::shared_ptr<Device> dst) {
  // TODO(wangwei) the comparison is restricted. May compare against device ID?
  if (device_ != dst) {
    Tensor tmp(shape_, dst, data_type_);
    if (block_ != nullptr && Size() && block_->initialized())
      tmp.CopyData(*this);
    if (block_ != nullptr && block_->DecRefCount() == 0)
      device_->FreeBlock(block_);
    block_ = tmp.block_;
    tmp.block_ = nullptr;
    device_ = dst;
  }
}

void Tensor::ToHost() {
  if (device_ != defaultDevice) ToDevice(device_->host());
}

template <typename DType>
void Tensor::CopyDataFromHostPtr(const DType *src, const size_t num,
                                 const size_t offset) {
  CHECK_EQ(sizeof(DType), SizeOf(data_type_))
      << "data_type is " << DataType_Name(data_type_)
      << " user given type is of size " << sizeof(DType);
  if (src != nullptr) {
    device_->CopyDataFromHostPtr(block(), src, sizeof(DType) * num,
                                 sizeof(DType) * offset);
  } else {
    LOG(WARNING) << "Copy data from null host ptr";
  }
}
template void Tensor::CopyDataFromHostPtr(const unsigned char *src,
    const size_t num,
    const size_t offset);
template void Tensor::CopyDataFromHostPtr(const float *src, const size_t num,
    const size_t offset);
template void Tensor::CopyDataFromHostPtr(const int *src, const size_t num,
    const size_t offset);

void Tensor::CopyData(const Tensor &src) {
  CHECK_EQ(Size(), src.Size());
  CHECK(block_ != nullptr);
  // Do copy only if the src's block is already initialized.
  if (src.block_ != nullptr) {
    singa::CopyDataToFrom(this, src, Size(), 0, 0);
  }
}

void Tensor::RepeatData(vector<size_t> repeats, int axis, int total_repeats, const Tensor &src) {
  if(repeats.size() == 1) {
    CHECK_EQ(Size(), src.Size()*total_repeats);
  } else {
    CHECK_EQ(Size(), src.Size()*total_repeats/src.shape()[axis]);
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
  //transpose_ = proto.transpose();
  strides_.clear();
  for (int32_t s : proto.strides()) strides_.push_back(s);
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
    for (size_t i = 0; i < Product(shape_); ++i) data[i] = proto.int_data((int)i);
    CopyDataFromHostPtr<int>(data.get(), Product(shape_));
    break;
  }
  ///TODO(wangji): Implement to support C++ type char using bytes type in protobuf
  /// which is equivalent to string type is different from the other cases. The kchar
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
  default: { LOG(FATAL) << "Unsupported Type" << DataType_Name(data_type_); }
  }
}

void Tensor::ToProto(singa::TensorProto *proto) const {
  proto->clear_shape();
  for (auto s : shape_) {
    proto->add_shape(s);
  }
  proto->set_data_type(data_type_);
  //proto->set_transpose(transpose_);
  proto->clear_strides();
  for (auto s : strides_) {
    proto->add_strides(s);
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
  default: { LOG(FATAL) << "Unsupported Type" << DataType_Name(data_type_); }
  }
}

Tensor Tensor::Clone(std::shared_ptr<Device> device) const {
  if (device == nullptr) device = device_;
  Tensor t(shape_, device_, data_type_);
  //t.transpose_ = transpose_;
  t.strides_ = strides_;
  t.CopyData(*this);
  return t;
}

Tensor Tensor::Repeat(vector<size_t> repeats, int axis, std::shared_ptr<Device> device) {
  if (device == nullptr) device = device_;
  vector<size_t> tshape;
  int total_repeats = 0;
  if (axis == Noaxis) {
    total_repeats = repeats[0];
    tshape.push_back(Product(shape_)*total_repeats);
  } else {
    if (repeats.size() == 1){
      total_repeats = repeats[0];
      for (int i = 0; i < shape_.size(); i++) {
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
        if(repeats[i] < 0) {
          LOG(FATAL) << "the repeats number is less than zero";
        }
        total_repeats += repeats[i];
      }
      for (int i = 0; i < shape_.size(); i++){
        if (i == axis) {
          tshape.push_back(total_repeats);
        } else{
          tshape.push_back(shape_[i]);
        }
      }
    }
  }
  Tensor t(tshape, device_);
  //t.strides_.push_back(1);
  t.RepeatData(repeats, axis, total_repeats, *this);
  return t;
}

//yisen todo
Tensor Tensor::T() const {
  // this function only works for 2d tensors
  CHECK_EQ(shape_.size(), 2u);
  Tensor t;
  t.device_ = device_;
  t.data_type_ = data_type_;
  t.shape_.push_back(shape_[1]);
  t.shape_.push_back(shape_[0]);
  t.strides_.clear();
  t.strides_.push_back(strides_[1]);
  t.strides_.push_back(strides_[0]);
  t.block_ = block_;
  block_->IncRefCount();
  return t;
}

//normal transpose without axes
Tensor Tensor::Transpose() const {
  // if(shape_.size() != strides_.size())
  //   generate_strides();

  Tensor t;
  t.device_ = device_;
  t.data_type_ = data_type_;
  t.strides_.clear();
  for (size_t n = 0; n < shape_.size(); ++n) {
    t.shape_.push_back(shape_[shape_.size() - n - 1]);
    t.strides_.push_back(strides_[shape_.size() - n - 1]);
  }
  t.block_ = block_;
  block_->IncRefCount();
  return t;
}

//transpose with axes
// TODO(wangwei) the shape and axes should match
Tensor Tensor::Transpose(const vector<size_t> &axes) const {
  // if(axes.size() != shape_.size()){
  //   std::cout << "Warning: Size of input axes doesn't match size of shape" << std::endl;
  //   return void();
  // }
  // if(shape_.size() != strides_.size())
  //   generate_strides();

  Tensor t;
  t.device_ = device_;
  t.data_type_ = data_type_;
  t.strides_.clear();
  for (size_t n = 0; n < axes.size(); ++n) {
    t.shape_.push_back(shape_[axes[n]]);
    t.strides_.push_back(strides_[axes[n]]);
  }
  t.block_ = block_;
  block_->IncRefCount();
  return t;
}

Tensor &Tensor::operator=(const Tensor &in) {
  // LOG(ERROR) << "= const &";
  if (block_ != nullptr && block_->DecRefCount() == 0)
    device_->FreeBlock(block_);
  //transpose_ = in.transpose_;
  strides_ = in.strides_;
  data_type_ = in.data_type_;
  shape_ = in.shape_;
  device_ = in.device_;
  block_ = in.block();
  if (block_ != nullptr)
    block_->IncRefCount();
  return *this;
}

Tensor &Tensor::operator=(Tensor &&in) {
  // LOG(ERROR) << "= &&";
  if (block_ != nullptr && block_->DecRefCount() == 0)
    device_->FreeBlock(block_);
  //transpose_ = in.transpose_;
  strides_ = std::move(in.strides_);
  data_type_ = in.data_type_;
  shape_ = std::move(in.shape_);
  device_ = in.device_;
  block_ = in.block_;
  in.block_ = nullptr;
  return *this;
}

//yisen todo
Tensor Reshape(const Tensor &in, const Shape &s) {
  Tensor out(in);
  out = out.Reshape(s);
  return out;
}

Tensor Reshape(const Tensor &in, Shape &&s) {
  Tensor out(in);
  out = out.Reshape(std::move(s));
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

  std::shared_ptr<Device> src_dev = src.device(), dst_dev = dst->device();
  Block *from = src.block(), *to = dst->block();
  if (dst_dev->lang() != src_dev->lang()) {
    // let the none cpp device conduct copy op
    if (dst_dev->lang() == kCpp) {
      src_dev->CopyDataToFrom(to, from, nBytes, kDeviceToHost, (int)d_offset,
                              (int)s_offset);
    } else if (src_dev->lang() == kCpp) {
      dst_dev->CopyDataToFrom(to, from, nBytes, kHostToDevice, (int)d_offset,
                              (int)s_offset);
    } else {
      LOG(FATAL) << "Not support mem copy betwee Cuda and OpenCL device";
    }
  } else {
    auto direct = src_dev->lang() == kCpp ? kHostToHost : kDeviceToDevice;
    src_dev->CopyDataToFrom(to, from, nBytes, direct, (int)d_offset, (int)s_offset);
  }
}

void RepeatDataToFrom(bool broadcast_flag, vector<size_t> repeats, int axis, 
                      Tensor *dst, const Tensor &src, const size_t num) {
  if (repeats.size() == 1) {
    broadcast_flag = true;
  } else if (repeats.size() > 1) {
    if (axis == Noaxis) {
      LOG(FATAL) << "When repeats parameter is sequence, axis cannot be None";
    }
  }
  for (size_t i = 0; i < repeats.size(); i++){
    CHECK_GE(repeats[i], 0);
  }
  auto width = SizeOf(src.data_type());
  CHECK_EQ(width, SizeOf(dst->data_type()));
  // size_t nBytes = num * width;
  int chunk = width;
  int axis_shape = 1;
  int shape_outer = 1;
  if (axis == Noaxis){
    axis_shape = 1;
    shape_outer = Product(src.shape());
  } else {
    for (size_t i = 0; i < axis; i++) {
      shape_outer *= src.shape()[i];
    }
    axis_shape = src.shape()[axis];
    for(size_t i = axis + 1; i < src.nDim(); i++) {
      chunk *= src.shape()[i];
    }
  }
  int dst_offset = 0;
  int src_offset = 0;
  std::shared_ptr<Device> src_dev = src.device(), dst_dev = dst->device();
  Block *from = src.block(), *to = dst->block();
  for (int i = 0; i < shape_outer; i++) {
    for (int j = 0; j < axis_shape; j++) {
      int temp = broadcast_flag ? repeats[0] : repeats[j];
      for (int k = 0; k < temp; k++) {
        if (dst_dev->lang() != src_dev->lang()) {
          // let the none cpp device conduct copy op
          if (dst_dev->lang() == kCpp) {
            src_dev->CopyDataToFrom(to, from, chunk, kDeviceToHost, dst_offset, src_offset);
          } else if (src_dev->lang() == kCpp) {
            dst_dev->CopyDataToFrom(to, from, chunk, kHostToDevice, dst_offset, src_offset);
          } else {
            LOG(FATAL) << "Not support mem repeat copy betwee Cuda and OpenCL device";
          }
        } else {
          auto direct = src_dev->lang() == kCpp ? kHostToHost : kDeviceToDevice;
          src_dev->CopyDataToFrom(to, from, chunk, direct, dst_offset, src_offset);
        }
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
float Tensor::L1() const {
  float nrm = 0.0f;
  TYPE_LANG_SWITCH(data_type_, DType, device_->lang(), Lang, {
    device_->Exec([&nrm, this](Context * ctx) {
      DType ret = DType(0);
      Asum<DType, Lang>(*this, &ret, ctx);
      nrm = TypeCast<DType, float>(ret);
    }, {this->block()}, {});
  });
  return nrm / Size();
}

/// L2 norm, Do not use Nrm2 (name conflict).
float Tensor::L2() const {
  float nrm = 0.0f;
  TYPE_LANG_SWITCH(data_type_, DType, device_->lang(), Lang, {
    device_->Exec([&nrm, this](Context * ctx) {
      DType ret = DType(0);
      Nrm2<DType, Lang>(*this, &ret, ctx);
      nrm = TypeCast<DType, float>(ret);
    }, {this->block()}, {});
  });
  return nrm / Size();
}

template <typename SType>
void Tensor::SetValue(const SType x) {
  CHECK_EQ(sizeof(SType), SizeOf(data_type_));
  //auto size = Size();
  auto ptr = block_;
  
  TYPE_LANG_SWITCH(data_type_, DType, device_->lang(), Lang, {
    // TODO(wangwei) cast x to DType
    device_->Exec([this, x, ptr](Context * ctx) {
      Set<DType, Lang>(x, this, ctx);
    }, {}, {ptr});
  });
}
template void Tensor::SetValue<float>(const float x);
template void Tensor::SetValue<int>(const int x);

#define EltwiseUnaryTensorFn(fn, t, ret)                               \
  do {                                                                 \
    TYPE_LANG_SWITCH(t.data_type(), DType, t.device()->lang(), Lang, { \
      ret->device()->Exec([t, ret](Context * ctx) {                    \
        fn<DType, Lang>(t, ret, ctx);       \
      }, {t.block()}, {ret->block()});                                 \
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
GenUnaryTensorFn(Transform);

#define EltwiseBinaryTensorFn(fn, lhs, rhs, ret)                            \
  do {                                                                      \
    TYPE_LANG_SWITCH(lhs.data_type(), DType, lhs.device()->lang(), Lang, {  \
      CHECK_EQ(sizeof(DType), SizeOf(rhs.data_type()));                     \
      ret->device()->Exec([lhs, rhs, ret](Context * ctx) {                  \
        fn<DType, Lang>(lhs, rhs, ret, \
                        ctx);                                               \
      }, {lhs.block(), rhs.block()}, {ret->block()});                       \
    });                                                                     \
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
GenBinaryTensorFn(operator<, LT);
GenBinaryTensorFn(operator<=, LE);
GenBinaryTensorFn(operator>, GT);
GenBinaryTensorFn(operator>=, GE);
#define EltwiseTensorScalarFn(fn, t, x, ret)                            \
  do {                                                                  \
    TYPE_LANG_SWITCH(t.data_type(), DType, t.device()->lang(), Lang, {  \
      static_assert(std::is_same<SType, DType>::value,                  \
                    "The Scalar type must match the Tensor data type"); \
      ret->device()->Exec([t, x, ret](Context * ctx) {                  \
        fn<DType, Lang>(t, x, ret, ctx);     \
      }, {t.block()}, {ret->block()});                                  \
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
  template Tensor op <float>(const Tensor &in, const float x); \
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
    in.device()->Exec([alpha, in, out](Context * ctx) {
      Div<DType, Lang>(alpha, in, out, ctx);
    }, {in.block()}, {out->block()});
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
  Tensor one(in.shape(), in.device(), in.data_type());
  one.SetValue(1.0f);
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    one.device()->Exec([in, one, &s](Context * ctx) {
      DType ret = DType(0);
      Dot<DType, Lang>(in, one, &ret, ctx);
      s = ret;
    }, {in.block(), one.block()}, {});
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

Tensor SoftMax(const Tensor &in) {
  Tensor out(in.shape(), in.device(), in.data_type());
  SoftMax(in, &out);
  return out;
}

Tensor RowMax(const Tensor &in) {
  Tensor ret({in.shape(0)}, in.device(), in.data_type());
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    in.device()->Exec([&in, &ret](Context * ctx) {
      //size_t nrow = 1;
      //if (in.nDim() > 1) nrow = in.shape(0);
      //size_t ncol = in.Size() / nrow;
      RowMax<DType, Lang>(in, &ret, ctx);
    }, {in.block()}, {ret.block()});
  });
  return ret;
}

void SoftMax(const Tensor &in, Tensor *out) {
  CHECK_LE(in.nDim(), 2u);
  out->CopyData(in);
  size_t nrow = 1, ncol = in.Size(), size = ncol;
  if (in.nDim() == 2u) {
    nrow = in.shape(0);
    ncol = size / nrow;
    out->Reshape(Shape{nrow, ncol});
  }
  Tensor tmp = RowMax(*out);
  SubColumn(tmp, out);
  Exp(*out, out);

  SumColumns(*out, &tmp);
  DivColumn(tmp, out);
  out->Reshape(in.shape());
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
    // CHECK_EQ(v.nDim(), 1u); (chonho) shape of v is 2-element tuple
    size_t nb_row = M->shape(0), nb_col = M->shape(1);
    CHECK_EQ(nb_row, v.Size());

    Tensor one(Shape{1, nb_col}, M->device(), M->data_type());
    one.SetValue(1.0f);  // TODO(wangwei) cast type
    Tensor vmat = Reshape(v, Shape{nb_row, 1});
    Mult(alpha, vmat, one, beta, M);
  }
}
template
void AddColumn(const float alpha, const float beta, const Tensor &v, Tensor *M);

void AddRow(const Tensor &v, Tensor *M) { AddRow(1, 1, v, M); }

/// Add row 'v' by each column of matrix M; write results into 'out'
template <typename SType>
void AddRow(const SType alpha, const SType beta, const Tensor &v, Tensor *M) {
  if (M->transpose()) {
    Tensor X = M->T();
    AddColumn(v, &X);
  } else {
    CHECK_EQ(M->nDim(), 2u);
    // CHECK_EQ(v.nDim(), 1u); (chonho) shape of v is 2-element tuple
    size_t nb_row = M->shape(0), nb_col = M->shape(1);
    CHECK_EQ(nb_col, v.Size());

    Tensor one(Shape{nb_row, 1}, M->device(), M->data_type());
    one.SetValue(1.0f);
    Tensor vmat = Reshape(v, Shape{1, nb_col});
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

Tensor ConcatOn(const vector<Tensor> &in, int axis) {
  vector<Tensor> tmp;
  Shape out_shape = in[0].shape();
  size_t dim = in[0].shape().size();
  CHECK_GE(dim, 2u) << " Only work for tensor of dim >=2 ";
  size_t size = in[0].Size() / in[0].shape(axis);
  size_t new_size = 0u;
  for (const auto& t : in) {
    CHECK_EQ(dim, t.shape().size()) << "All tensors should have the same dim";
    CHECK_EQ(size, t.Size() / t.shape(axis)) << "The size of all axis should "
        << " be the same except the concatenated axis";
    new_size += t.shape(axis);
  }
  out_shape[axis] = new_size;
  if (axis == 0) {
    size_t nrow = 0;
    for (const auto& t : in) {
      nrow += t.shape(0);
      tmp.push_back(Reshape(t, {t.shape(0), t.Size() / t.shape(0)}));
    }
    auto ret = ConcatenateRows(tmp);
    ret = ret.Reshape(out_shape);
    return ret;
  } else {
    for (const auto& t : in) {
      size_t nrow = 1;
      for (int i = 0; i < axis; i++)
        nrow *= t.shape(i);
      tmp.push_back(Reshape(t, {nrow, t.Size() / nrow}));
    }
    auto ret = ConcatenateColumns(tmp);
    ret = ret.Reshape(out_shape);
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
Tensor ConcatRows(const vector<Tensor> &in) {
  return ConcatenateRows(in);
}
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


Tensor SliceOn(const Tensor&in, const size_t start, const size_t end, int axis) {
  Shape out_shape = in.shape();
  out_shape[axis] = end - start;
  if (axis == 0) {
    auto ret = SliceRows(Reshape(in, {in.shape(0), in.Size() / in.shape(0)}),
                         start, end);
    ret.Reshape(out_shape);
    return ret;
  } else {
    size_t nrow = 1;
    for (int i = 0; i < axis; i++)
      nrow *= in.shape(i);
    auto suffix = in.Size() / nrow / in.shape(axis);
    auto ret = SliceColumns(Reshape(in, {nrow, in.Size() / nrow}),
                            start * suffix, end * suffix);
    ret = ret.Reshape(out_shape);
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
  //CHECK(!M->transpose()) << "Not supported yet";
  CHECK_EQ(M->nDim(), 2u);
  // CHECK_EQ(v.nDim(), 1u); (chonho) shape of v is 2-element tuple
  CHECK_EQ(v.Size(), M->shape(0));
  CheckDataTypeAndLang(*M, v);
  TYPE_LANG_SWITCH(v.data_type(), DType, v.device()->lang(), Lang, {
    v.device()->Exec([M, v](Context * ctx) {
      DGMM<DType, Lang>(false, *M, v,
      M, ctx);
    }, {M->block(), v.block()}, {M->block()});
  });
}

/// Multiply row 'v' with each row of matrix M; write results into 'out'
void MultRow(const Tensor &v, Tensor *M) {
  //CHECK(!M->transpose()) << "Not supported yet";
  CHECK_EQ(M->nDim(), 2u);
  // CHECK_EQ(v.nDim(), 1u); (chonho) shape of v is 2-element tuple
  CHECK_EQ(v.Size(), M->shape(1));
  CheckDataTypeAndLang(*M, v);
  TYPE_LANG_SWITCH(v.data_type(), DType, v.device()->lang(), Lang, {
    v.device()->Exec([M, v](Context * ctx) {
      DGMM<DType, Lang>(true, *M, v,
      M, ctx);
    }, {M->block(), v.block()}, {M->block()});
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
    Tensor X = M.T();
    SumColumns(X, v);
  } else {
    CHECK_EQ(M.nDim(), 2u);
    // CHECK_EQ(v->nDim(), 1u); (chonho) shape of v is 2-element tuple
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
    out->device()->Exec([prob, out](Context * ctx) {
      Bernoulli<DType, Lang>(prob, out, ctx);
    }, {}, {out->block()}, true);
  });
}

template void Bernoulli<float>(const float p, Tensor *out);

template <typename SType>
void Uniform(const SType low, const SType high, Tensor *out) {
  TYPE_LANG_SWITCH(out->data_type(), DType, out->device()->lang(), Lang, {
    auto l = TypeCast<SType, DType>(low);
    auto h = TypeCast<SType, DType>(high);
    out->device()->Exec([l, h, out](Context * ctx) {
      Uniform<DType, Lang>(l, h, out, ctx);
    }, {}, {out->block()}, true);
  });
}

template void Uniform<float>(const float low, const float high, Tensor *out);

template <typename SType>
void Gaussian(const SType mean, const SType std, Tensor *out) {
  TYPE_LANG_SWITCH(out->data_type(), DType, out->device()->lang(), Lang, {
    auto m = TypeCast<SType, DType>(mean);
    auto s = TypeCast<SType, DType>(std);
    out->device()->Exec([m, s, out](Context * ctx) {
      Gaussian<DType, Lang>(m, s, out, ctx);
    }, {}, {out->block()}, true);
  });
}
template void Gaussian<float>(const float mean, const float std, Tensor *out);

// ================Blas operations============================================

template <typename SType>
void Axpy(const SType alpha, const Tensor &in, Tensor *out) {
  TYPE_LANG_SWITCH(in.data_type(), DType, in.device()->lang(), Lang, {
    auto a = TypeCast<SType, DType>(alpha);
    out->device()->Exec([a, in, out](Context * ctx) {
      Axpy<DType, Lang>(a, in, out, ctx);
    }, {in.block(), out->block()}, {out->block()});
  });
}

template
void Axpy<float>(const float alpha, const Tensor &in, Tensor *out);

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
      C->device()->Exec([a, A, b, B, C](Context * ctx) {
        GEMV<DType, Lang>(a, A, B, b, C, ctx);
      }, {A.block(), B.block()}, {C->block()});
    });
  } else {
    CHECK(!C->transpose());
    TYPE_LANG_SWITCH(A.data_type(), DType, A.device()->lang(), Lang, {
      auto a = TypeCast<SType, DType>(alpha);
      auto b = TypeCast<SType, DType>(beta);
      C->device()->Exec([a, A, b, B, C](Context * ctx) {
        GEMM<DType, Lang>(a, A, B, b, C,
        ctx);
      }, {A.block(), B.block()}, {C->block()});
    });
  }
}

// ************************
// Misc.
// ************************
void ComputeCrossEntropy(const Tensor &p, const Tensor &t, Tensor *loss) {
  CHECK_LE(p.nDim(), 2u);
  CHECK_LE(t.nDim(), 2u);
  size_t batchsize = 1;
  if (p.nDim() == 2u) batchsize = p.shape(0);
  size_t dim = p.Size() / batchsize;
  TYPE_LANG_SWITCH(p.data_type(), DType, p.device()->lang(), Lang, {
    p.device()->Exec([batchsize, dim, t, p, loss](Context * ctx) {
      bool int_target = t.Size() == batchsize;
      ComputeCrossEntropy<DType, Lang>(int_target, batchsize, dim, p.block(),
      t.block(), loss->block(), ctx);
    }, {p.block(), t.block()}, {loss->block()});
  });
}

void SoftmaxCrossEntropyBwd(const Tensor &t, Tensor *p) {
  CHECK_LE(p->nDim(), 2u);
  CHECK_LE(t.nDim(), 2u);
  size_t batchsize = 1;
  if (p->nDim() == 2u) batchsize = p->shape(0);
  size_t dim = p->Size() / batchsize;
  TYPE_LANG_SWITCH(p->data_type(), DType, p->device()->lang(), Lang, {
    p->device()->Exec([batchsize, dim, t, p](Context * ctx) {
      bool int_target = t.Size() == batchsize;
      SoftmaxCrossEntropyBwd<DType, Lang>(int_target, batchsize, dim,
      p->block(), t.block(), p->block(), ctx);
    }, {p->block(), t.block()}, {p->block()});
  });
}

}  // namespace singa
