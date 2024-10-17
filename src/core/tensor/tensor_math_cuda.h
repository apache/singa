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

#ifndef SINGA_CORE_TENSOR_TENSOR_MATH_CUDA_H_
#define SINGA_CORE_TENSOR_TENSOR_MATH_CUDA_H_
#include "singa/singa_config.h"
#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "../../model/layer/cudnn_utils.h"
#include "./math_kernel.h"
#include "./tensor_math.h"
#include "singa/core/common.h"
#include "singa/core/tensor.h"
#include "singa/utils/cuda_utils.h"

#define check_cudnn(expression)                          \
  {                                                      \
    cudnnStatus_t status = (expression);                 \
    if (status != CUDNN_STATUS_SUCCESS) {                \
      LOG(FATAL) << "Error on line " << __LINE__ << ": " \
                 << cudnnGetErrorString(status) << " ";  \
    }                                                    \
  }

namespace singa {

// ===================== Helper Functions =============================

/*
cudnn requires tensor dimensions to fulfill 1 requirement:
  1.) Dimensions to be set to a minimum of 4 for 4d and lower dimensional
tensors
      if input tensor is 5d, cudnn will take a 5d tensor as input. Beyond 5d,
certain operations are not supported.
      (cudnnOp supports up to 5d, cudnnReduce supports up to 8d)

  for e.g. Tensor A has shape {3,3}, cudnn requires shape of {1,1,3,3} to be the
input
           Tensor B has shape (2,3,4), cudnn requires shape of {1,2,3,4} to be
the input
*/
vector<int> generate_shape_cuda(const Tensor& x) {
  Shape shape = x.shape();
  // maximum dimension allowed defined in cudnn.h, variable CUDNN_DIM_MAX
  // TODO: check other side effects
  CHECK_LE(shape.size(), CUDNN_DIM_MAX)
      << "Dimensions (shape) beyond " << CUDNN_DIM_MAX
      << " are currently not supported";
  vector<int> shape_arr;
  if (shape.size() < 4) {
    for (int n = 0; n < 4 - int(shape.size()); ++n) {
      shape_arr.push_back(1);
    }
  }
  for (auto x : shape) {
    shape_arr.push_back(static_cast<int>(x));
  }
  return shape_arr;
}

int generate_dim_cuda(const Tensor& x) {
  // maximum dimension allowed defined in cudnn.h, variable CUDNN_DIM_MAX
  CHECK_LE(x.nDim(), CUDNN_DIM_MAX)
      << "Dimensions (shape) beyond " << CUDNN_DIM_MAX
      << " are currently not supported";
  if (x.shape().size() <= 4) {
    return 4;
  } else {
    return x.nDim();
  }
}

/*
  cudnn requires stride dimensions to conform to the format of the shape input
  as well
    1.) Stride dimensions to be set to a minimum of 4 for 4d and lower
  dimensional tensors
        If input tensor is 5d, cudnn will take a 5d tensor as input. Beyond 5d,
  certain operations are not supported.
        (cudnnOp supports up to 5d, cudnnReduce supports up to 8d)

    for e.g. Tensor A has shape {3,3}, stride {3,1}, cudnn requires shape
  {1,1,3,3}
    and stride {9, 9, 3, 1} or {9, 9, 1, 3} to be the inputs
  */
vector<int> generate_strides_cuda(const Tensor& x) {
  Shape shape = x.shape();
  auto& strides = x.stride();
  vector<int> strides_arr;
  int product = Product(shape);
  if (shape.size() < 4) {
    for (int n = 0; n < 4 - int(shape.size()); ++n) {
      strides_arr.push_back(product);
    }
  }
  for (auto x : strides) strides_arr.push_back(static_cast<int>(x));
  return strides_arr;
}

cudnnTensorDescriptor_t generate_tensor_nd_desc(const Tensor& x) {
  cudnnTensorDescriptor_t x_desc;
  check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
  // LOG(INFO) << vec2str(x.shape());
  // LOG(INFO) << vec2str(x.stride());
  auto st = x.stride();
  std::vector<size_t> sh;
  bool reshape = false;
  for (size_t i = 0; i < st.size(); i++) {
    if (st[i] == 0) {
      sh.push_back(1);
      reshape = true;
    } else {
      sh.push_back(x.shape(i));
    }
  }
  auto y = x;
  if (reshape) y = Reshape(x, sh);

  auto shape = generate_shape_cuda(y);
  auto stride = generate_strides_cuda(y);

  // LOG(INFO) << vec2str(shape);
  // LOG(INFO) << vec2str(stride);
  // LOG(INFO) << "";
  check_cudnn(cudnnSetTensorNdDescriptor(
      x_desc, GetCudnnDataType(x.data_type()), generate_dim_cuda(y),
      shape.data(), stride.data()));

  return x_desc;
}

cudnnOpTensorDescriptor_t generate_op_desc(cudnnOpTensorOp_t op) {
  cudnnOpTensorDescriptor_t op_desc;
  check_cudnn(cudnnCreateOpTensorDescriptor(&op_desc));
  check_cudnn(cudnnSetOpTensorDescriptor(op_desc, op, CUDNN_DATA_FLOAT,
                                         CUDNN_PROPAGATE_NAN));

  return op_desc;
}

// ===================== CUDA Functions =============================

/// out[i] = |in[i]|
template <>
void Abs<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());

  float alpha1 = 1.0;
  float alpha2 = -1.0;
  float beta = 0.0;
  cudnnTensorDescriptor_t in_desc = generate_tensor_nd_desc(in);
  check_cudnn(cudnnOpTensor(
      ctx->cudnn_handle, generate_op_desc(CUDNN_OP_TENSOR_MAX),
      (void*)(&alpha1), in_desc, inPtr, (void*)(&alpha2), in_desc, inPtr,
      (void*)(&beta), generate_tensor_nd_desc(*out), outPtr));
  cudnnDestroyTensorDescriptor(in_desc);
}

template <>
void CastCopy<float, int, lang::Cuda>(const Tensor* src, Tensor* dst,
                                      Context* ctx) {
  const float* srcPtr = static_cast<const float*>(src->block()->data());
  int* dstPtr = static_cast<int*>(dst->block()->mutable_data());
  cuda::cast_float_2_int(dst->Size(), srcPtr, dstPtr, ctx->stream);
}

template <>
void CastCopy<int, float, lang::Cuda>(const Tensor* src, Tensor* dst,
                                      Context* ctx) {
  const int* srcPtr = static_cast<const int*>(src->block()->data());
  float* dstPtr = static_cast<float*>(dst->block()->mutable_data());
  cuda::cast_int_2_float(dst->Size(), srcPtr, dstPtr, ctx->stream);
}

template <>
void CastCopy<float, half_float::half, lang::Cuda>(const Tensor* src,
                                                   Tensor* dst, Context* ctx) {
  /* cpp half is for labeling only, cuda requires __half */
  const float* srcPtr = static_cast<const float*>(src->block()->data());
  __half* dstPtr = static_cast<__half*>(dst->block()->mutable_data());
  cuda::float2half(dst->Size(), srcPtr, dstPtr, ctx->stream);
}

template <>
void CastCopy<half_float::half, float, lang::Cuda>(const Tensor* src,
                                                   Tensor* dst, Context* ctx) {
  /* cpp half is for labeling only, cuda requires __half */
  const __half* srcPtr = static_cast<const __half*>(src->block()->data());
  float* dstPtr = static_cast<float*>(dst->block()->mutable_data());
  cuda::half2float(dst->Size(), srcPtr, dstPtr, ctx->stream);
}

template <>
void Set<float, lang::Cuda>(const float x, Tensor* out, Context* ctx) {
  float* outPtr = static_cast<float*>(out->block()->mutable_data());

  check_cudnn(cudnnSetTensor(ctx->cudnn_handle, generate_tensor_nd_desc(*out),
                             outPtr, (void*)(&x)));
}

template <>
void Set<half_float::half, lang::Cuda>(const half_float::half x, Tensor* out,
                                       Context* ctx) {
  vector<half_float::half> data_src(out->size(), x);
  out->CopyDataFromHostPtr(data_src.data(), out->size(), 0);
}

template <>
void Add<float, lang::Cuda>(const Tensor& in, const float x, Tensor* out,
                            Context* ctx) {
  Set<float, lang::Cuda>(x, out, ctx);
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());

  float alpha = 1.0, beta = 1.0;
  check_cudnn(cudnnAddTensor(ctx->cudnn_handle, (void*)(&alpha),
                             generate_tensor_nd_desc(in), inPtr, (void*)(&beta),
                             generate_tensor_nd_desc(*out), outPtr));
}

template <typename T>
void TraverseUnaryTransformImpl(const Tensor& in1, Tensor* in1Bc,
                                Context* ctx) {
  Tensor shape(Shape{in1.nDim()}, in1.device(), kInt);
  Tensor stride(Shape{in1.nDim()}, in1.device(), kInt);
  const vector<int> strideVec(in1.stride().begin(), in1.stride().end());
  const vector<int> shapeVec(in1.shape().begin(), in1.shape().end());
  shape.CopyDataFromHostPtr(shapeVec.data(), in1.nDim());
  stride.CopyDataFromHostPtr(strideVec.data(), in1.nDim());
  const int* shapePtr = static_cast<const int*>(shape.block()->data());
  const int* stridePtr = static_cast<const int*>(stride.block()->data());

  const T* inPtr1 = static_cast<const T*>(in1.block()->data());
  T* inBcPtr1 = static_cast<T*>(in1Bc->block()->mutable_data());

  const size_t n = Product(in1Bc->shape());

  cuda::traverse_unary_transform(n, in1.nDim(), inPtr1, shapePtr, stridePtr,
                                 inBcPtr1, ctx->stream);
}
template void TraverseUnaryTransformImpl<float>(const Tensor& in1,
                                                Tensor* in1Bc, Context* ctx);
template void TraverseUnaryTransformImpl<__half>(const Tensor& in1,
                                                 Tensor* in1Bc, Context* ctx);

template <typename T>
void TransformImpl(const Tensor& in, Tensor* out, Context* ctx) {
  if (in.broadcasted()) {
    TraverseUnaryTransformImpl<T>(in, out, ctx);
  } else {
    const void* inPtr = in.block()->data();
    void* outPtr = out->block()->mutable_data();

    float alpha = 1.0;
    float beta = 0.0;

    check_cudnn(cudnnTransformTensor(
        ctx->cudnn_handle, (void*)(&alpha), generate_tensor_nd_desc(in), inPtr,
        (void*)(&beta), generate_tensor_nd_desc(*out), outPtr));
  }
}
template void TransformImpl<__half>(const Tensor& in, Tensor* out,
                                    Context* ctx);
template void TransformImpl<float>(const Tensor& in, Tensor* out, Context* ctx);

template <>
void Transform<half_float::half, lang::Cuda>(const Tensor& in, Tensor* out,
                                             Context* ctx) {
  TransformImpl<__half>(in, out, ctx);
}

template <>
void Transform<__half, lang::Cuda>(const Tensor& in, Tensor* out,
                                   Context* ctx) {
  TransformImpl<__half>(in, out, ctx);
}

template <>
void Transform<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  TransformImpl<float>(in, out, ctx);
}

/// add sub div mul pow on two tensors
#define GenBinaryMathFn(fn, fn_impl, kernel)                                  \
  template <typename T>                                                       \
  void fn_impl(const Tensor& in1, const Tensor& in2, Tensor* out,             \
               Context* ctx) {                                                \
    const T* inPtr1 = static_cast<const T*>(in1.block()->data());             \
    const T* inPtr2 = static_cast<const T*>(in2.block()->data());             \
    T* outPtr = static_cast<T*>(out->block()->mutable_data());                \
    const size_t num = out->Size();                                           \
                                                                              \
    if (!in1.broadcasted() && !in2.broadcasted()) {                           \
      if (!in1.transpose() && !in2.transpose() &&                             \
          (in1.stride() == in2.stride())) {                                   \
        kernel(num, inPtr1, inPtr2, outPtr, ctx->stream);                     \
      } else {                                                                \
        if (in1.transpose() && in2.transpose()) {                             \
          Tensor t(in1.shape(), in1.device(), in1.data_type());               \
          Transform<T, lang::Cuda>(in1, &t, ctx);                             \
          Transform<T, lang::Cuda>(in2, out, ctx);                            \
                                                                              \
          T* tPtr = static_cast<T*>(t.block()->mutable_data());               \
          kernel(num, tPtr, outPtr, outPtr, ctx->stream);                     \
        } else if (in1.transpose()) {                                         \
          Transform<T, lang::Cuda>(in1, out, ctx);                            \
          kernel(num, outPtr, inPtr2, outPtr, ctx->stream);                   \
        } else if (in2.transpose()) {                                         \
          Transform<T, lang::Cuda>(in2, out, ctx);                            \
          kernel(num, inPtr1, outPtr, outPtr, ctx->stream);                   \
        }                                                                     \
      }                                                                       \
    } else {                                                                  \
      Tensor in1bc, in2bc;                                                    \
      if (in1.broadcasted()) {                                                \
        in1bc = Tensor(in1.shape(), in1.device(), in1.data_type());           \
        Transform<T, lang::Cuda>(in1, &in1bc, ctx);                           \
        inPtr1 = static_cast<const T*>(in1bc.block()->data());                \
      }                                                                       \
      if (in2.broadcasted()) {                                                \
        in2bc = Tensor(in2.shape(), in2.device(), in2.data_type());           \
        Transform<T, lang::Cuda>(in2, &in2bc, ctx);                           \
        inPtr2 = static_cast<const T*>(in2bc.block()->data());                \
      }                                                                       \
      kernel(num, inPtr1, inPtr2, outPtr, ctx->stream);                       \
    }                                                                         \
  }                                                                           \
  template void fn_impl<__half>(const Tensor& in1, const Tensor& in2,         \
                                Tensor* out, Context* ctx);                   \
  template void fn_impl<float>(const Tensor& in1, const Tensor& in2,          \
                               Tensor* out, Context* ctx);                    \
                                                                              \
  template <>                                                                 \
  void fn<float, lang::Cuda>(const Tensor& in1, const Tensor& in2,            \
                             Tensor* out, Context* ctx) {                     \
    fn_impl<float>(in1, in2, out, ctx);                                       \
  }                                                                           \
  template <>                                                                 \
  void fn<half_float::half, lang::Cuda>(const Tensor& in1, const Tensor& in2, \
                                        Tensor* out, Context* ctx) {          \
    fn_impl<__half>(in1, in2, out, ctx);                                      \
  }

/// out = in1 * in2
GenBinaryMathFn(EltwiseMult, EltwiseMultImpl, cuda::mult);
/// out = in1 + in2
GenBinaryMathFn(Add, AddImpl, cuda::add);
/// out = in1 - in2
GenBinaryMathFn(Sub, SubImpl, cuda::sub);
/// out = in1 / in2
GenBinaryMathFn(Div, DivImpl, cuda::div);
/// out = in1 ^ in2
GenBinaryMathFn(Pow, PowImpl, cuda::pow);

/// Element-wise operation, clamp every element into [low, high]
/// if x>high, then x=high; if x<low, then x=low.
template <>
void Clamp<float, lang::Cuda>(const float low, const float high,
                              const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();
  // if both in and out strides are the same, we proceed to normal cuda::clamp
  if (in.stride() == out->stride()) {
    cuda::clamp(num, low, high, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::clamp(num, low, high, outPtr, outPtr, ctx->stream);
  }
}

template <>
void Div<float, lang::Cuda>(const float x, const Tensor& in, Tensor* out,
                            Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::div(num, x, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::div(num, x, outPtr, outPtr, ctx->stream);
  }
}

/// out = in * x
template <>
void EltwiseMult<float, lang::Cuda>(const Tensor& in, const float x,
                                    Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();
  cuda::mult(num, inPtr, x, outPtr, ctx->stream);
}

template <>
void EltwiseMult<half_float::half, lang::Cuda>(const Tensor& in,
                                               const half_float::half x,
                                               Tensor* out, Context* ctx) {
  const __half* inPtr = static_cast<const __half*>(in.block()->data());
  __half* outPtr = static_cast<__half*>(out->block()->mutable_data());
  const size_t num = in.Size();
  cuda::mult(num, inPtr, static_cast<__half>(x), outPtr, ctx->stream);
}

/// Base is e. out[i]=e^in[i]
template <>
void Exp<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::exp(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::exp(num, outPtr, outPtr, ctx->stream);
  }
}

template <>
void Erf<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::erf(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::erf(num, outPtr, outPtr, ctx->stream);
  }
}

template <>
void Ceil<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::ceil2(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::ceil2(num, outPtr, outPtr, ctx->stream);
  }
}

template <>
void Floor<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::floor(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::floor(num, outPtr, outPtr, ctx->stream);
  }
}

template <>
void Round<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::round(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::round(num, outPtr, outPtr, ctx->stream);
  }
}

template <>
void RoundE<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::rounde(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::rounde(num, outPtr, outPtr, ctx->stream);
  }
}

template <>
void GE<float, lang::Cuda>(const Tensor& in, const float x, Tensor* out,
                           Context* ctx) {
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const float* inPtr = static_cast<const float*>(in.block()->data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::ge(num, inPtr, x, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::ge(num, outPtr, x, outPtr, ctx->stream);
  }
}
template <>
void GE<float, lang::Cuda>(const Tensor& in1, const Tensor& in2, Tensor* out,
                           Context* ctx) {
  Sub<float, lang::Cuda>(in1, in2, out, ctx);
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in1.Size();
  cuda::ge(num, outPtr, 0.0, outPtr, ctx->stream);
}

template <>
void GT<float, lang::Cuda>(const Tensor& in, const float x, Tensor* out,
                           Context* ctx) {
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const float* inPtr = static_cast<const float*>(in.block()->data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::gt(num, inPtr, x, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::gt(num, outPtr, x, outPtr, ctx->stream);
  }
}
template <>
void GT<float, lang::Cuda>(const Tensor& in1, const Tensor& in2, Tensor* out,
                           Context* ctx) {
  Sub<float, lang::Cuda>(in1, in2, out, ctx);
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in1.Size();
  cuda::gt(num, outPtr, 0.0, outPtr, ctx->stream);
}

template <>
void LE<float, lang::Cuda>(const Tensor& in, const float x, Tensor* out,
                           Context* ctx) {
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const float* inPtr = static_cast<const float*>(in.block()->data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::le(num, inPtr, x, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::le(num, outPtr, x, outPtr, ctx->stream);
  }
}
template <>
void LE<float, lang::Cuda>(const Tensor& in1, const Tensor& in2, Tensor* out,
                           Context* ctx) {
  Sub<float, lang::Cuda>(in1, in2, out, ctx);
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in1.Size();
  cuda::le(num, outPtr, 0.0, outPtr, ctx->stream);
}

template <>
void EQ<float, lang::Cuda>(const Tensor& in, const float x, Tensor* out,
                           Context* ctx) {
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const float* inPtr = static_cast<const float*>(in.block()->data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::eq(num, inPtr, x, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::eq(num, outPtr, x, outPtr, ctx->stream);
  }
}
template <>
void EQ<float, lang::Cuda>(const Tensor& in1, const Tensor& in2, Tensor* out,
                           Context* ctx) {
  Sub<float, lang::Cuda>(in1, in2, out, ctx);
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in1.Size();
  cuda::eq(num, outPtr, 0.0, outPtr, ctx->stream);
}

/// Natual logarithm, the base is e, Neper number out[i]=ln(in[i]).
template <>
void Log<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::log(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::log(num, outPtr, outPtr, ctx->stream);
  }
}

template <>
void LT<float, lang::Cuda>(const Tensor& in, const float x, Tensor* out,
                           Context* ctx) {
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const float* inPtr = static_cast<const float*>(in.block()->data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::lt(num, inPtr, x, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::lt(num, outPtr, x, outPtr, ctx->stream);
  }
}
template <>
void LT<float, lang::Cuda>(const Tensor& in1, const Tensor& in2, Tensor* out,
                           Context* ctx) {
  Sub<float, lang::Cuda>(in1, in2, out, ctx);
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in1.Size();
  cuda::lt(num, outPtr, 0.0, outPtr, ctx->stream);
}

/// Element-wise operation, out[i] = in[i]^x
template <>
void Pow<float, lang::Cuda>(const Tensor& in, const float x, Tensor* out,
                            Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::pow(num, inPtr, x, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::pow(num, outPtr, x, outPtr, ctx->stream);
  }
}

template <>
void ReLUBackward<float, lang::Cuda>(const Tensor& in1, const Tensor& in2,
                                     Tensor* out, Context* ctx) {
  const float* in1Ptr = static_cast<const float*>(in1.block()->data());
  const float* in2Ptr = static_cast<const float*>(in2.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in1.Size();
  cuda::relubackward(num, in1Ptr, in2Ptr, outPtr, ctx->stream);
}
template <>
void ReLUBackward<half_float::half, lang::Cuda>(const Tensor& in1,
                                                const Tensor& in2, Tensor* out,
                                                Context* ctx) {
  const __half* in1Ptr = static_cast<const __half*>(in1.block()->data());
  const __half* in2Ptr = static_cast<const __half*>(in2.block()->data());
  __half* outPtr = static_cast<__half*>(out->block()->mutable_data());
  const size_t num = in1.Size();
  cuda::relubackward(num, in1Ptr, in2Ptr, outPtr, ctx->stream);
}

/// Element-wise operation, out[i]=max(0, in[i])
// template <>
// void ReLU<float, lang::Cuda>(const Tensor& in, Tensor* out,
//                              Context* ctx) {
//   const float* inPtr = static_cast<const float*>(in.block()->data());
//   float* outPtr = static_cast<float*>(out->block()->mutable_data());

//   cudnnActivationDescriptor_t act_desc;
//   cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
//   cudnnNanPropagation_t cudnn_propagation = CUDNN_PROPAGATE_NAN;
//   double coef = 0.0; //only used for CLIPPED_RELU or ELU
//   cudnnCreateActivationDescriptor(&act_desc);
//   cudnnSetActivationDescriptor(act_desc, mode, cudnn_propagation, coef);

//   float alpha[1] = {1.0};
//   float beta[1] = {0.0};
//   cudnnDataType_t cudnn_dtype = CUDNN_DATA_FLOAT;
//   cudnnTensorDescriptor_t in_desc, out_desc;
//   cudnnCreateTensorDescriptor(&in_desc);
//   cudnnCreateTensorDescriptor(&out_desc);
//   cudnnSetTensorNdDescriptor(in_desc, cudnn_dtype, in.generate_dim_cuda(),
// in.generate_shape_cuda().data(), in.generate_strides_cuda().data());
//   cudnnSetTensorNdDescriptor(out_desc, cudnn_dtype, out->generate_dim_cuda(),
// out->generate_shape_cuda().data(), out->generate_strides_cuda().data());
//   cudnnActivationForward(ctx->cudnn_handle, act_desc, (void*)(&alpha),
//   in_desc, inPtr,
//                         (void*)(&beta), out_desc, outPtr);

//   cudnnDestroyTensorDescriptor(in_desc);
//   cudnnDestroyTensorDescriptor(out_desc);
//   cudnnDestroyActivationDescriptor(act_desc);
// }

template <>
void ReLU<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::relu(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::relu(num, outPtr, outPtr, ctx->stream);
  }
}
template <>
void ReLU<half_float::half, lang::Cuda>(const Tensor& in, Tensor* out,
                                        Context* ctx) {
  const __half* inPtr = static_cast<const __half*>(in.block()->data());
  __half* outPtr = static_cast<__half*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::relu(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<half_float::half, lang::Cuda>(in, out, ctx);
    cuda::relu(num, outPtr, outPtr, ctx->stream);
  }
}

// /// Element-wise operation, out[i]=sigmoid([in[i])
// template <>
// void Sigmoid<float, lang::Cuda>(const Tensor& in, Tensor* out,
//                                 Context* ctx) {
//   const float* inPtr = static_cast<const float*>(in.block()->data());
//   float* outPtr = static_cast<float*>(out->block()->mutable_data());

//   cudnnActivationDescriptor_t act_desc;
//   cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
//   cudnnNanPropagation_t cudnn_propagation = CUDNN_PROPAGATE_NAN;
//   double coef = 0.0; //only used for CLIPPED_RELU or ELU
//   cudnnCreateActivationDescriptor(&act_desc);
//   cudnnSetActivationDescriptor(act_desc, mode, cudnn_propagation, coef);

//   float alpha[1] = {1.0};
//   float beta[1] = {0.0};
//   cudnnDataType_t cudnn_dtype = CUDNN_DATA_FLOAT;
//   cudnnTensorDescriptor_t in_desc, out_desc;
//   cudnnCreateTensorDescriptor(&in_desc);
//   cudnnCreateTensorDescriptor(&out_desc);
//   cudnnSetTensorNdDescriptor(in_desc, cudnn_dtype, in.generate_dim_cuda(),
// in.generate_shape_cuda().data(), in.generate_strides_cuda().data());
//   cudnnSetTensorNdDescriptor(out_desc, cudnn_dtype, out->generate_dim_cuda(),
// out->generate_shape_cuda().data(), out->generate_strides_cuda().data());
//   cudnnActivationForward(ctx->cudnn_handle, act_desc, (void*)(&alpha),
//   in_desc, inPtr,
//                         (void*)(&beta), out_desc, outPtr);

//   cudnnDestroyTensorDescriptor(in_desc);
//   cudnnDestroyTensorDescriptor(out_desc);
//   cudnnDestroyActivationDescriptor(act_desc);
// }

/// Element-wise operation, out[i]=sigmoid([in[i])
template <>
void Sigmoid<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::sigmoid(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::sigmoid(num, outPtr, outPtr, ctx->stream);
  }
}

// out[i] = sign(in[i])
template <>
void Sign<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::sign(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::sign(num, outPtr, outPtr, ctx->stream);
  }
}

// out[i] = softplus(in[i])
template <>
void SoftPlus<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::softplus(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::softplus(num, outPtr, outPtr, ctx->stream);
  }
}

// out[i] = softsign(in[i])
template <>
void SoftSign<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::softsign(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::softsign(num, outPtr, outPtr, ctx->stream);
  }
}

// Element-wise operation, out[i]=sqrt([in[i])
template <>
void Sqrt<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  float* outPtr = static_cast<float*>(out->block()->mutable_data());

#if CUDNN_MAJOR < 7
  Transform<float, lang::Cuda>(in, out, ctx);
  size_t num = in.Size();
  cuda::sqrt(num, outPtr, outPtr, ctx->stream);
#else
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float alpha1 = 1.0;
  float alpha2 = 0.0;
  float beta = 0.0;
  cudnnTensorDescriptor_t in_desc = generate_tensor_nd_desc(in);
  check_cudnn(cudnnOpTensor(
      ctx->cudnn_handle, generate_op_desc(CUDNN_OP_TENSOR_SQRT),
      (void*)(&alpha1), in_desc, inPtr, (void*)(&alpha2), in_desc, inPtr,
      (void*)(&beta), generate_tensor_nd_desc(*out), outPtr));
#endif  // CUDNN_MAJOR < 7
}

/// Element-wise operation, out[i]=in[i]^2
template <>
void Square<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = in.Size();

  if (in.stride() == out->stride()) {
    cuda::square(num, inPtr, outPtr, ctx->stream);
  } else {  // else we transform in to out to store first
    Transform<float, lang::Cuda>(in, out, ctx);
    cuda::square(num, outPtr, outPtr, ctx->stream);
  }
}

// template <>
// void Sum<float, lang::Cuda>(const size_t num, const Block* in, float* out,
//                             Context* ctx) {
//   LOG(FATAL) << "Cuda Sum is not implemented!";
//   // const float* inPtr = static_cast<const float*>(in.data());
//   // cuda::sum(num, inPtr, out, ctx->stream);
// }

/// Element-wise operation, out[i]=tanh([in[i])
// template <>
// void Tanh<float, lang::Cuda>(const Tensor& in, Tensor* out,
//                              Context* ctx) {
//   const float* inPtr = static_cast<const float*>(in.block()->data());
//   float* outPtr = static_cast<float*>(out->block()->mutable_data());

//   cudnnActivationDescriptor_t act_desc;
//   cudnnActivationMode_t mode = CUDNN_ACTIVATION_TANH;
//   cudnnNanPropagation_t cudnn_propagation = CUDNN_PROPAGATE_NAN;
//   double coef = 0.0; //only used for CLIPPED_RELU or ELU
//   cudnnCreateActivationDescriptor(&act_desc);
//   cudnnSetActivationDescriptor(act_desc, mode, cudnn_propagation, coef);

//   float alpha[1] = {1.0};
//   float beta[1] = {0.0};
//   cudnnDataType_t cudnn_dtype = CUDNN_DATA_FLOAT;
//   cudnnTensorDescriptor_t in_desc, out_desc;
//   cudnnCreateTensorDescriptor(&in_desc);
//   cudnnCreateTensorDescriptor(&out_desc);
//   cudnnSetTensorNdDescriptor(in_desc, cudnn_dtype, in.generate_dim_cuda(),
// in.generate_shape_cuda().data(), in.generate_strides_cuda().data());
//   cudnnSetTensorNdDescriptor(out_desc, cudnn_dtype, out->generate_dim_cuda(),
// out->generate_shape_cuda().data(), out->generate_strides_cuda().data());
//   cudnnActivationForward(ctx->cudnn_handle, act_desc, (void*)(&alpha),
//   in_desc, inPtr,
//                         (void*)(&beta), out_desc, outPtr);

//   cudnnDestroyTensorDescriptor(in_desc);
//   cudnnDestroyTensorDescriptor(out_desc);
//   cudnnDestroyActivationDescriptor(act_desc);
// }

#define GenUnaryTensorCudaFn(fn, cudafn)                                    \
  template <>                                                               \
  void fn<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) { \
    const float* inPtr = static_cast<const float*>(in.block()->data());     \
    float* outPtr = static_cast<float*>(out->block()->mutable_data());      \
    const size_t num = in.Size();                                           \
    if (in.stride() == out->stride()) {                                     \
      cuda::cudafn(num, inPtr, outPtr, ctx->stream);                        \
    } else {                                                                \
      Transform<float, lang::Cuda>(in, out, ctx);                           \
      cuda::cudafn(num, outPtr, outPtr, ctx->stream);                       \
    }                                                                       \
  }

GenUnaryTensorCudaFn(Cos, cos);
GenUnaryTensorCudaFn(Cosh, cosh);
GenUnaryTensorCudaFn(Acos, acos);
GenUnaryTensorCudaFn(Acosh, acosh);
GenUnaryTensorCudaFn(Sin, sin);
GenUnaryTensorCudaFn(Sinh, sinh);
GenUnaryTensorCudaFn(Asin, asin);
GenUnaryTensorCudaFn(Asinh, asinh);
GenUnaryTensorCudaFn(Tan, tan);
GenUnaryTensorCudaFn(Tanh, tanh);
GenUnaryTensorCudaFn(Atan, atan);
GenUnaryTensorCudaFn(Atanh, atanh);

// ================Random functions===========================================
/// Each element of out would be 1 with prob p and 0 with 1-p. 0<= p <= 1
// Get the random generator from 'ctx'
// If DType is not float, then convert the threshold to DType
template <>
void Bernoulli<float, lang::Cuda>(const float p, Tensor* out, Context* ctx) {
  auto rgen = ctx->curand_generator;
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = out->Size();
  CURAND_CHECK(curandGenerateUniform(rgen, outPtr, num));
  cuda::threshold(num, p, outPtr, outPtr, ctx->stream);
}

// The random generator should be extracted from ctx.
// If DType is not float, then convert the low and high to DType
template <>
void Uniform<float, lang::Cuda>(const float low, const float high, Tensor* out,
                                Context* ctx) {
  auto rgen = ctx->curand_generator;
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = out->Size();
  CURAND_CHECK(curandGenerateUniform(rgen, outPtr, num));
  cuda::mult(num, outPtr, high - low, outPtr, ctx->stream);
  cuda::add(num, outPtr, low, outPtr, ctx->stream);
}
template <>
void Uniform<half_float::half, lang::Cuda>(const half_float::half low,
                                           const half_float::half high,
                                           Tensor* out, Context* ctx) {
  Tensor tmp(out->shape(), out->device(), kFloat32);
  Uniform<float, lang::Cuda>(static_cast<float>(low), static_cast<float>(high),
                             &tmp, ctx);
  CastCopy<float, half_float::half, lang::Cuda>(&tmp, out, ctx);
}

// The random generator should be extracted from ctx.
// If DType is not float, then convert the mean and delta to DType
template <>
void Gaussian<float, lang::Cuda>(const float mean, const float std, Tensor* out,
                                 Context* ctx) {
  auto rgen = ctx->curand_generator;
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = out->Size();

  // CURAND_STATUS_LENGTH_NOT_MULTIPLE
  if (num % 2 != 0) {
    Tensor tmp(Shape{num + 1}, out->device());
    float* outPtr_tmp = static_cast<float*>(tmp.block()->mutable_data());
    CURAND_CHECK(curandGenerateNormal(rgen, outPtr_tmp, num + 1, mean, std));
    CopyDataToFrom(out, tmp, num, 0, 0);
  } else {
    CURAND_CHECK(curandGenerateNormal(rgen, outPtr, num, mean, std));
  }
}
template <>
void Gaussian<half_float::half, lang::Cuda>(const half_float::half mean,
                                            const half_float::half std,
                                            Tensor* out, Context* ctx) {
  Tensor tmp(out->shape(), out->device(), kFloat32);
  Gaussian<float, lang::Cuda>(static_cast<float>(mean), static_cast<float>(std),
                              &tmp, ctx);
  CastCopy<float, half_float::half, lang::Cuda>(&tmp, out, ctx);
}

// =========================Blas operations==================================
// ref to http://docs.nvidia.com/cuda/cublas
template <>
void Amax<float, lang::Cuda>(const Tensor& in, size_t* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  int idx = 1;
  const size_t num = in.Size();
  CUBLAS_CHECK(cublasIsamax(handle, num, inPtr, 1, &idx));
  *out = idx - 1;  // cublas index starts from 1
}

/// return the index of the element with the min value.
template <>
void Amin<float, lang::Cuda>(const Tensor& in, size_t* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  int idx = 1;
  const size_t num = in.Size();
  CUBLAS_CHECK(cublasIsamin(handle, num, inPtr, 1, &idx));
  *out = idx - 1;
}

/// out = sum |x| for all x in in
template <>
void Asum<float, lang::Cuda>(const Tensor& in, float* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  const size_t num = in.Size();
  CUBLAS_CHECK(cublasSasum(handle, num, inPtr, 1, out));
}

/// out = alpha * in + out
template <>
void Axpy<float, lang::Cuda>(const float alpha, const Tensor& in, Tensor* out,
                             Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  const size_t num = in.Size();
  CUBLAS_CHECK(cublasSaxpy(handle, num, &alpha, inPtr, 1, outPtr, 1));
}

/// out = alpha * in + out
template <>
void Axpy<float, lang::Cuda>(const Tensor& alpha, const Tensor& in, Tensor* out,
                             Context* ctx) {
  auto handle = ctx->cublas_handle;
  const size_t num = in.Size();
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK(cublasAxpyEx(
      handle, num, alpha.block()->data(), CUDA_R_32F, in.block()->data(),
      CUDA_R_32F, 1, out->block()->mutable_data(), CUDA_R_32F, 1, CUDA_R_32F));
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

template <>
void Axpy<half_float::half, lang::Cuda>(const Tensor& alpha, const Tensor& in,
                                        Tensor* out, Context* ctx) {
  auto handle = ctx->cublas_handle;
  const size_t num = in.Size();

  auto _alpha = alpha.AsType(kFloat32);

  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK(cublasAxpyEx(
      handle, num, _alpha.block()->data(), CUDA_R_32F, in.block()->data(),
      CUDA_R_16F, 1, out->block()->mutable_data(), CUDA_R_16F, 1, CUDA_R_32F));
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

template <>
void Axpy<half_float::half, lang::Cuda>(const half_float::half alpha,
                                        const Tensor& in, Tensor* out,
                                        Context* ctx) {
  auto handle = ctx->cublas_handle;
  const size_t num = in.Size();
  const float _alpha = static_cast<const float>(alpha);
  CUBLAS_CHECK(cublasAxpyEx(handle, num, &alpha, CUDA_R_32F, in.block()->data(),
                            CUDA_R_16F, 1, out->block()->mutable_data(),
                            CUDA_R_16F, 1, CUDA_R_32F));
}

/// out = \sum_i in1[i] * in2[i]
template <>
void Dot<float, lang::Cuda>(const Tensor& in1, const Tensor& in2, float* out,
                            Context* ctx) {
  const float* inPtr1 = static_cast<const float*>(in1.block()->data());
  const float* inPtr2 = static_cast<const float*>(in2.block()->data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  const size_t num = in1.Size();
  CUBLAS_CHECK(cublasSdot(handle, num, inPtr1, 1, inPtr2, 1, out));
}
template <>
void Dot<float, lang::Cuda>(const Tensor& in1, const Tensor& in2, Tensor* out,
                            Context* ctx) {
  const float* inPtr1 = static_cast<const float*>(in1.block()->data());
  const float* inPtr2 = static_cast<const float*>(in2.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  auto handle = ctx->cublas_handle;
  const size_t num = in1.Size();
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK(cublasSdot(handle, num, inPtr1, 1, inPtr2, 1, outPtr));
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

template <>
void Nrm2<float, lang::Cuda>(const Tensor& in, float* out, Context* ctx) {
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  const float* inPtr = static_cast<const float*>(in.block()->data());
  const size_t num = in.Size();
  cublasSnrm2(handle, num, inPtr, 1, out);
}
template <>
void Scale<float, lang::Cuda>(const float x, Tensor* out, Context* ctx) {
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t num = out->Size();
  CUBLAS_CHECK(cublasSscal(handle, num, &x, outPtr, 1));
}
// NOTE: cublas uses column major order.
// http://peterwittek.com/cublas-matrix-c-style.html
template <>
void DGMM<float, lang::Cuda>(const bool side_right, const Tensor& M,
                             const Tensor& v, Tensor* out, Context* ctx) {
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  const float* MPtr = static_cast<const float*>(M.block()->data());
  const float* vPtr = static_cast<const float*>(v.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t nrow = M.shape(0);
  const size_t ncol = M.shape(1);
  if (side_right) {
    CUBLAS_CHECK(cublasSdgmm(handle, CUBLAS_SIDE_LEFT, ncol, nrow, MPtr, ncol,
                             vPtr, 1, outPtr, ncol));
  } else {
    CUBLAS_CHECK(cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, ncol, nrow, MPtr, ncol,
                             vPtr, 1, outPtr, ncol));
  }
}
template <>
void GEMV<float, lang::Cuda>(const float alpha, const Tensor& A,
                             const Tensor& v, const float beta, Tensor* out,
                             Context* ctx) {
  const float* APtr = static_cast<const float*>(A.block()->data());
  const float* vPtr = static_cast<const float*>(v.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t m = A.shape()[0];
  const size_t n = A.shape()[1];

  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  if (!(A.transpose()))
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, APtr, n, vPtr,
                             1, &beta, outPtr, 1));
  else
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, APtr, m, vPtr,
                             1, &beta, outPtr, 1));
}

template <>
void GEMV<half_float::half, lang::Cuda>(const half_float::half alpha,
                                        const Tensor& A, const Tensor& v,
                                        const half_float::half beta,
                                        Tensor* out, Context* ctx) {
  // Fp16 not supported
  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  auto _A = A.AsType(kFloat32);
  auto _v = v.AsType(kFloat32);
  Tensor _out = Tensor(out->shape(), out->device(), kFloat32);
  GEMV<float, lang::Cuda>(static_cast<float>(alpha), _A, _v,
                          static_cast<float>(beta), &_out, ctx);
  CastCopy<float, half_float::half, lang::Cuda>(&_out, out, ctx);
}

template <>
void GEMM<half_float::half, lang::Cuda>(const half_float::half alpha,
                                        const Tensor& A, const Tensor& B,
                                        const half_float::half beta, Tensor* C,
                                        Context* ctx) {
  auto transA = A.transpose();
  auto transa = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto transB = B.transpose();
  auto transb = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  const size_t nrowA = A.shape()[0];
  const size_t ncolA = A.shape()[1];
  const size_t ncolB = B.shape()[1];
  int lda = transA ? nrowA : ncolA;
  int ldb = transB ? ncolA : ncolB;
  int ldc = ncolB;
  const __half* APtr = static_cast<const __half*>(A.block()->data());
  const __half* BPtr = static_cast<const __half*>(B.block()->data());
  __half* CPtr = static_cast<__half*>(C->block()->mutable_data());
  const __half* alphaPtr =
      static_cast<const __half*>(static_cast<const void*>(&alpha));
  const __half* betaPtr =
      static_cast<const __half*>(static_cast<const void*>(&beta));
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  CUBLAS_CHECK(cublasHgemm(handle, transb, transa, ncolB, nrowA, ncolA,
                           alphaPtr, BPtr, ldb, APtr, lda, betaPtr, CPtr, ldc));
}

template <>
void Dot<half_float::half, lang::Cuda>(const Tensor& in1, const Tensor& in2,
                                       Tensor* out, Context* ctx) {
  const __half* inPtr1 = static_cast<const __half*>(in1.block()->data());
  const __half* inPtr2 = static_cast<const __half*>(in2.block()->data());
  __half* outPtr = static_cast<__half*>(out->block()->mutable_data());
  auto handle = ctx->cublas_handle;
  const size_t num = in1.Size();
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK(cublasDotEx(handle, num, inPtr1, CUDA_R_16F, 1, inPtr2,
                           CUDA_R_16F, 1, outPtr, CUDA_R_16F, CUDA_R_32F));
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

// http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
template <>
void GEMM<float, lang::Cuda>(const float alpha, const Tensor& A,
                             const Tensor& B, const float beta, Tensor* C,
                             Context* ctx) {
  auto transA = A.transpose();
  auto transa = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto transB = B.transpose();
  auto transb = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  const size_t nrowA = A.shape()[0];
  const size_t ncolA = A.shape()[1];
  const size_t ncolB = B.shape()[1];
  int lda = transA ? nrowA : ncolA;
  int ldb = transB ? ncolA : ncolB;
  int ldc = ncolB;
  const float* APtr = static_cast<const float*>(A.block()->data());
  const float* BPtr = static_cast<const float*>(B.block()->data());
  float* CPtr = static_cast<float*>(C->block()->mutable_data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  CUBLAS_CHECK(cublasSgemm(handle, transb, transa, ncolB, nrowA, ncolA, &alpha,
                           BPtr, ldb, APtr, lda, &beta, CPtr, ldc));
}

/* pseudocode for GEMM Strided Batched:
 * for (int p = 0; p < batchCount; ++p) {
 *   for (int m = 0; m < M; ++m) {
 *     for (int n = 0; n < N; ++n) {
 *       T c_mnp = 0;
 *       for (int k = 0; k < K, ++k)
 *         c_mnp += A[m + k*ldA + p*strideA] * B[k + n*ldB + p*strideB];
 *       C[m + n*ldC + p*strideC] =
 *         (*alpha)*c_mnp + (*beta)*C[m + n*ldC + p*strideC];
 *     }
 *   }
 * }
 */
template <>
void GEMMBatched<float, lang::Cuda>(const float alpha, const Tensor& A,
                                    const Tensor& B, const float beta,
                                    Tensor* C, Context* ctx) {
  auto handle = ctx->cublas_handle;

  auto transA = A.transpose();
  auto transa = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto transB = B.transpose();
  auto transb = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  const size_t ncolB = B.shape().end()[-1];
  const size_t nrowA = A.shape().end()[-2];
  const size_t ncolA = A.shape().end()[-1];

  size_t batchCount = A.shape()[0];
  if (A.nDim() == 4u) batchCount *= A.shape()[1];

  const size_t strideA = A.shape().end()[-1] * A.shape().end()[-2];
  const size_t strideB = B.shape().end()[-1] * B.shape().end()[-2];
  const size_t strideC = C->shape().end()[-1] * C->shape().end()[-2];

  int lda = transA ? nrowA : ncolA;
  int ldb = transB ? ncolA : ncolB;
  int ldc = ncolB;

  const float* APtr = static_cast<const float*>(A.block()->data());
  const float* BPtr = static_cast<const float*>(B.block()->data());
  float* CPtr = static_cast<float*>(C->block()->mutable_data());
  CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, transa, transb, ncolB, nrowA, ncolA, &alpha, BPtr, ldb, strideB,
      APtr, lda, strideA, &beta, CPtr, ldc, strideC, batchCount));
}

template <>
void SoftMax<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE;
  cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;

  /*
   * tensor tmp is for generating cudnn descriptor
   *   as for cudnn softmax, it required shape of {N, C, 1, 1}
   *   while helper func `generate_shape_cuda` generate shape of {1, 1, N, C}
   *   Thus this part serve similar purpose as `generate_shape_cuda` but in
   * reverse manner
   */
  CHECK_LE(in.shape().size(), 5)
      << "Dimensions (shape) beyond 5 are currently not supported";
  auto tmp = in;
  while (tmp.shape().size() < 4) {
    auto s = tmp.shape();
    s.push_back(1);
    tmp.Reshape(s);
  }

  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());

  float alpha = 1.0;
  float beta = 0.0;

  check_cudnn(cudnnSoftmaxForward(ctx->cudnn_handle, algorithm, mode,
                                  (void*)(&alpha), generate_tensor_nd_desc(tmp),
                                  inPtr, (void*)(&beta),
                                  generate_tensor_nd_desc(tmp), outPtr));
}
template <>
void SoftMax<half_float::half, lang::Cuda>(const Tensor& in, Tensor* out,
                                           Context* ctx) {
  cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE;
  cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;

  /*
   * tensor tmp is for generating cudnn descriptor
   *   as for cudnn softmax, it required shape of {N, C, 1, 1}
   *   while helper func `generate_shape_cuda` generate shape of {1, 1, N, C}
   *   Thus this part serve similar purpose as `generate_shape_cuda` but in
   * reverse manner
   */
  CHECK_LE(in.shape().size(), 5)
      << "Dimensions (shape) beyond 5 are currently not supported";
  auto tmp = in;
  while (tmp.shape().size() < 4) {
    auto s = tmp.shape();
    s.push_back(1);
    tmp.Reshape(s);
  }

  const __half* inPtr = static_cast<const __half*>(in.block()->data());
  __half* outPtr = static_cast<__half*>(out->block()->mutable_data());

  float alpha = 1.0f;
  float beta = 0.0f;

  check_cudnn(cudnnSoftmaxForward(
      ctx->cudnn_handle, algorithm, mode, static_cast<void*>(&alpha),
      generate_tensor_nd_desc(tmp), inPtr, static_cast<void*>(&beta),
      generate_tensor_nd_desc(tmp), outPtr));
}

template <>
void SoftMaxBackward<float, lang::Cuda>(const Tensor& in, Tensor* out,
                                        const Tensor& fdout, Context* ctx) {
  cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE;
  cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;

  /*
   * tensor tmp is for generating cudnn descriptor
   *   as for cudnn softmax, it required shape of {N, C, 1, 1}
   *   while helper func `generate_shape_cuda` generate shape of {1, 1, N, C}
   *   Thus this part serve similar purpose as `generate_shape_cuda` but in
   * reverse manner
   */
  CHECK_LE(in.shape().size(), 5)
      << "Dimensions (shape) beyond 5 are currently not supported";
  auto tmp = in;
  while (tmp.shape().size() < 4) {
    auto s = tmp.shape();
    s.push_back(1);
    tmp.Reshape(s);
  }

  const float* inPtr = static_cast<const float*>(in.block()->data());
  const float* fdoutPtr = static_cast<const float*>(fdout.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());

  float alpha = 1.0;
  float beta = 0.0;

  check_cudnn(cudnnSoftmaxBackward(
      ctx->cudnn_handle, algorithm, mode, (void*)(&alpha),
      generate_tensor_nd_desc(tmp), fdoutPtr, generate_tensor_nd_desc(tmp),
      inPtr, (void*)(&beta), generate_tensor_nd_desc(tmp), outPtr));
}

template <>
void ComputeCrossEntropy<float, lang::Cuda>(bool int_target,
                                            const size_t batchsize,
                                            const size_t dim, const Tensor& p,
                                            const Tensor& t, Tensor* loss,
                                            Context* ctx) {
  const float* pPtr = static_cast<const float*>(p.block()->data());
  const int* tPtr = static_cast<const int*>(t.block()->data());
  float* lossPtr = static_cast<float*>(loss->block()->mutable_data());
  cuda::ComputeCrossEntropy(int_target, batchsize, dim, pPtr, tPtr, lossPtr,
                            ctx->stream);
}

template <>
void ComputeCrossEntropy<half_float::half, lang::Cuda>(
    bool int_target, const size_t batchsize, const size_t dim, const Tensor& p,
    const Tensor& t, Tensor* loss, Context* ctx) {
  const __half* pPtr = static_cast<const __half*>(p.block()->data());
  const int* tPtr = static_cast<const int*>(t.block()->data());
  __half* lossPtr = static_cast<__half*>(loss->block()->mutable_data());
  cuda::ComputeCrossEntropy(int_target, batchsize, dim, pPtr, tPtr, lossPtr,
                            ctx->stream);
}

template <>
void SoftmaxCrossEntropyBwd<float, lang::Cuda>(bool int_target,
                                               const size_t batchsize,
                                               const size_t dim,
                                               const Tensor& p, const Tensor& t,
                                               Tensor* grad, Context* ctx) {
  CHECK_EQ(p.block(), grad->block())
      << "Use the same pointer to optimize performance";
  const float* pPtr = static_cast<const float*>(p.block()->data());
  const int* tPtr = static_cast<const int*>(t.block()->data());
  float* gradPtr = static_cast<float*>(grad->block()->mutable_data());
  cuda::SoftmaxCrossEntropyBwd(int_target, batchsize, dim, pPtr, tPtr, gradPtr,
                               ctx->stream);
}

template <>
void SoftmaxCrossEntropyBwd<half_float::half, lang::Cuda>(
    bool int_target, const size_t batchsize, const size_t dim, const Tensor& p,
    const Tensor& t, Tensor* grad, Context* ctx) {
  CHECK_EQ(p.block(), grad->block())
      << "Use the same pointer to optimize performance";
  const __half* pPtr = static_cast<const __half*>(p.block()->data());
  const int* tPtr = static_cast<const int*>(t.block()->data());
  __half* gradPtr = static_cast<__half*>(grad->block()->mutable_data());
  cuda::SoftmaxCrossEntropyBwd(int_target, batchsize, dim, pPtr, tPtr, gradPtr,
                               ctx->stream);
}

// template <>
// void RowMax<float, lang::Cuda>(const Tensor& in, Tensor* out,
//                                Context* ctx) {
//   const float* inPtr = static_cast<const float*>(in.block()->data());
//   float* outPtr = static_cast<float*>(out->block()->mutable_data());
//   // const size_t nrow = in.shape()[0];
//   // const size_t ncol = in.shape()[1];
//   // cuda::RowMax(nrow, ncol, inPtr, outPtr, ctx->stream);

//   //vector<int> reduce_row_axes_shape = in.generate_shape_cuda();
//   //reduce_row_axes_shape.back() = 1; //reduce axis 1, so we set last element
//   d in shape {a,b,c,d} to 1

//   vector<int> reduce_row_axes_shape = {1,1,1,1};
//   vector<int> reduced_strides = {1,1,1,1};

//   //reduce_desc
//   cudnnReduceTensorDescriptor_t reduce_desc;
//   cudnnReduceTensorOp_t reduce_op = CUDNN_REDUCE_TENSOR_ADD;
//   cudnnDataType_t cudnn_dtype = CUDNN_DATA_FLOAT;
//   cudnnNanPropagation_t cudnn_propagation = CUDNN_PROPAGATE_NAN;
//   cudnnReduceTensorIndices_t cudnn_indices = CUDNN_REDUCE_TENSOR_NO_INDICES;
//   //cudnnReduceTensorIndices_t cudnn_indices =
//   CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
//   cudnnIndicesType_t cudnn_indices_type = CUDNN_32BIT_INDICES;
//   cudnnCreateReduceTensorDescriptor(&reduce_desc);
//   cudnnSetReduceTensorDescriptor(reduce_desc, reduce_op, cudnn_dtype,
//                                  cudnn_propagation, cudnn_indices,
//                                  cudnn_indices_type);

//   //instantiate new tensor to use new blocks as memory instead of cudaMalloc
//   //create 2 tensors of same size as input tensor
//   Shape reduction_size = {1000};
//   Tensor indices(reduction_size, in.device(), in.data_type());
//   Tensor workspace(reduction_size, in.device(), in.data_type());
//   size_t indices_bytes = indices.block()->size()*1000;
//   size_t workspace_bytes = workspace.block()->size()*1000;
//   size_t* indicesPtr = static_cast<size_t*>(indices.block()->mutable_data());
//   float* workspacePtr =
//   static_cast<float*>(workspace.block()->mutable_data());
//   //void* indicesPtr{nullptr}; void* workspacePtr{nullptr};
//   //cudaMalloc(&indicesPtr, indices_bytes); cudaMalloc(&workspacePtr,
//   workspace_bytes);

//   float alpha[1] = {1.0};
//   float beta[1] = {0.0};
//   cudnnTensorDescriptor_t in_desc, out_desc;
//   cudnnCreateTensorDescriptor(&in_desc);
//   cudnnCreateTensorDescriptor(&out_desc);
//   cudnnSetTensorNdDescriptor(in_desc, cudnn_dtype, in.generate_dim_cuda(),
// in.generate_shape_cuda().data(), in.generate_strides_cuda().data());
//   //cudnnSetTensorNdDescriptor(out_desc, cudnn_dtype,
//   out->generate_dim_cuda(),
// out->generate_shape_cuda().data(), out->generate_strides_cuda().data());
//   cudnnSetTensorNdDescriptor(out_desc, cudnn_dtype, out->generate_dim_cuda(),
// reduce_row_axes_shape.data(), reduced_strides.data());
//   cudnnReduceTensor(ctx->cudnn_handle, reduce_desc,
//                     indicesPtr, indices_bytes, workspacePtr, workspace_bytes,
//                     (void*)(&alpha), in_desc, inPtr, (void*)(&beta),
//                     out_desc, outPtr);

//   cudnnDestroyTensorDescriptor(in_desc);
//   cudnnDestroyTensorDescriptor(out_desc);
// }

template <>
void RowMax<float, lang::Cuda>(const Tensor& in, Tensor* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in.block()->data());
  float* outPtr = static_cast<float*>(out->block()->mutable_data());
  const size_t nrow = in.shape()[0];
  const size_t ncol = in.shape()[1];

  if (in.transpose()) {
    Tensor t(in.shape(), in.device(), in.data_type());
    Transform<float, lang::Cuda>(in, &t, ctx);
    const float* tPtr_const = static_cast<const float*>(t.block()->data());
    cuda::RowMax(nrow, ncol, tPtr_const, outPtr, ctx->stream);
  } else {
    cuda::RowMax(nrow, ncol, inPtr, outPtr, ctx->stream);
  }
}

// must put this function after Set and Dot functions due to the error from
// instantiation before specialization
template <>
void Sum<float, lang::Cuda>(const Tensor& in, float* out, Context* ctx) {
#if CUDNN_MAJOR < 7
  Tensor one(in.shape(), in.device(), in.data_type());
  Set<float, lang::Cuda>(float(1), &one, ctx);
  Dot<float, lang::Cuda>(in, one, out, ctx);
#else
  const float* inPtr = static_cast<const float*>(in.block()->data());
  // reduce all axes to 1 for cudnnReduce, e.g. Tensor A with shape (2,4) will
  // be reduced to (1)
  Shape reduced_shape = {1};
  Tensor t(reduced_shape, in.device(), in.data_type());
  float* tPtr = static_cast<float*>(t.block()->mutable_data());
  vector<int> reduce_all_axes = generate_shape_cuda(in);
  for (size_t n = 0; n < reduce_all_axes.size(); ++n) {
    reduce_all_axes[n] = 1;
  }

  // reduce_desc
  cudnnReduceTensorDescriptor_t reduce_desc;
  cudnnReduceTensorOp_t reduce_op = CUDNN_REDUCE_TENSOR_ADD;
  cudnnDataType_t cudnn_dtype = CUDNN_DATA_FLOAT;
  cudnnNanPropagation_t cudnn_propagation = CUDNN_PROPAGATE_NAN;
  cudnnReduceTensorIndices_t cudnn_indices = CUDNN_REDUCE_TENSOR_NO_INDICES;
  cudnnIndicesType_t cudnn_indices_type = CUDNN_32BIT_INDICES;
  check_cudnn(cudnnCreateReduceTensorDescriptor(&reduce_desc));
  check_cudnn(cudnnSetReduceTensorDescriptor(
      reduce_desc, reduce_op, cudnn_dtype, cudnn_propagation, cudnn_indices,
      cudnn_indices_type));

  // instantiate 2 new tensors to use new blocks as memory instead of cudaMalloc
  size_t reduction_size_int = Product(in.shape());
  Shape reduction_size = {reduction_size_int * 100};
  Tensor indices(reduction_size, in.device(), in.data_type());
  Tensor workspace(reduction_size, in.device(), in.data_type());
  size_t indices_bytes = indices.block()->size() * 100;
  size_t workspace_bytes = workspace.block()->size() * 100;
  size_t* indicesPtr = static_cast<size_t*>(indices.block()->mutable_data());
  float* workspacePtr = static_cast<float*>(workspace.block()->mutable_data());
  // void* indicesPtr{nullptr}; void* workspacePtr{nullptr};
  // cudaMalloc(&indicesPtr, indices_bytes); cudaMalloc(&workspacePtr,
  // workspace_bytes);

  float alpha = 1.0;
  float beta = 0.0;
  check_cudnn(cudnnReduceTensor(
      ctx->cudnn_handle, reduce_desc, indicesPtr, indices_bytes, workspacePtr,
      workspace_bytes, (void*)(&alpha), generate_tensor_nd_desc(in), inPtr,
      (void*)(&beta), generate_tensor_nd_desc(t), tPtr));

  *out = tPtr[0];
#endif  // CUDNN_MAJOR < 7
}

}  // namespace singa

#endif  // USE_CUDA
#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CUDA_H_
