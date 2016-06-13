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

#ifndef  SINGA_CORE_TENSOR_TENSOR_MATH_CUDA_H_
#define  SINGA_CORE_TENSOR_TENSOR_MATH_CUDA_H_
#include "singa_config.h"
#ifdef USE_CUDA
#include "./tensor_math.h"
#include "./math_kernel.h"
#include "singa/utils/cuda_utils.h"
#include "singa/core/common.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "singa/utils/cuda_utils.h"

namespace singa {

/// out[i] = |in[i]|
template <>
void Abs<float, lang::Cuda>(const size_t num, const Blob* in, Blob* out,
                            Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::abs(num, inPtr, outPtr, ctx->stream);
}
/// out = in + x
template <>
void Add<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                            Blob* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::add(num, inPtr, x, outPtr, ctx->stream);
}
/// out = in1 + in2
template <>
void Add<float, lang::Cuda>(const size_t num, const Blob* in1, const Blob* in2,
                            Blob* out, Context* ctx) {
  const float* inPtr1 = static_cast<const float*>(in1->data());
  const float* inPtr2 = static_cast<const float*>(in2->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::add(num, inPtr1, inPtr2, outPtr, ctx->stream);
}
/// Element-wise operation, clamp every element into [low, high]
/// if x>high, then x=high; if x<low, then x=low.
template <>
void Clamp<float, lang::Cuda>(const size_t num, const float low,
                              const float high, const Blob* in, Blob* out,
                              Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::clamp(num, low, high, inPtr, outPtr, ctx->stream);
}
/// out = in1 / in2
template <>
void Div<float, lang::Cuda>(const size_t num, const Blob* in1, const Blob* in2,
                            Blob* out, Context* ctx) {
  const float* inPtr1 = static_cast<const float*>(in1->data());
  const float* inPtr2 = static_cast<const float*>(in2->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::div(num, inPtr1, inPtr2, outPtr, ctx->stream);
}

template <>
void Div<float, lang::Cuda>(const size_t num, const float x, const Blob* in,
                            Blob* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::div(num, x, inPtr, outPtr, ctx->stream);
}

/// out = in * x
template <>
void EltwiseMult<float, lang::Cuda>(const size_t num, const Blob* in,
                                    const float x, Blob* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::mult(num, inPtr, x, outPtr, ctx->stream);
}
/// out = in1 * in2
template <>
void EltwiseMult<float, lang::Cuda>(const size_t num, const Blob* in1,
                                    const Blob* in2, Blob* out, Context* ctx) {
  const float* inPtr1 = static_cast<const float*>(in1->data());
  const float* inPtr2 = static_cast<const float*>(in2->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::mult(num, inPtr1, inPtr2, outPtr, ctx->stream);
}
/// Base is e. out[i]=e^in[i]
template <>
void Exp<float, lang::Cuda>(const size_t num, const Blob* in, Blob* out,
                            Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::exp(num, inPtr, outPtr, ctx->stream);
}

template <>
void GE<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                           Blob* out, Context* ctx) {
  float* outPtr = static_cast<float*>(out->mutable_data());
  const float* inPtr = static_cast<const float*>(in->data());
  cuda::ge(num, inPtr, x, outPtr, ctx->stream);
}

template <>
void GT<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                           Blob* out, Context* ctx) {
  float* outPtr = static_cast<float*>(out->mutable_data());
  const float* inPtr = static_cast<const float*>(in->data());
  cuda::gt(num, inPtr, x, outPtr, ctx->stream);
}

template <>
void LE<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                           Blob* out, Context* ctx) {
  float* outPtr = static_cast<float*>(out->mutable_data());
  const float* inPtr = static_cast<const float*>(in->data());
  cuda::le(num, inPtr, x, outPtr, ctx->stream);
}

/// Natual logarithm, the base is e, Neper number out[i]=ln(in[i]).
template <>
void Log<float, lang::Cuda>(const size_t num, const Blob* in, Blob* out,
                            Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::log(num, inPtr, outPtr, ctx->stream);
}
template <>
void LT<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                           Blob* out, Context* ctx) {
  float* outPtr = static_cast<float*>(out->mutable_data());
  const float* inPtr = static_cast<const float*>(in->data());
  cuda::lt(num, inPtr, x, outPtr, ctx->stream);
}

/// Element-wise operation, out[i] = in[i]^x
template <>
void Pow<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                            Blob* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::pow(num, inPtr, x, outPtr, ctx->stream);
}
/// Element-wise operation, out[i] = in1[i]^in2[i]
template <>
void Pow<float, lang::Cuda>(const size_t num, const Blob* in1, const Blob* in2,
                            Blob* out, Context* ctx) {
  const float* inPtr1 = static_cast<const float*>(in1->data());
  const float* inPtr2 = static_cast<const float*>(in2->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::pow(num, inPtr1, inPtr2, outPtr, ctx->stream);
}

/// Element-wise operation, out[i]=max(0, in[i])
template <>
void ReLU<float, lang::Cuda>(const size_t num, const Blob* in, Blob* out,
                             Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::relu(num, inPtr, outPtr, ctx->stream);
}

/// out[i] = x
template <>
void Set<float, lang::Cuda>(const size_t num, const float x, Blob* out,
                            Context* ctx) {
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::set(num, x, outPtr, ctx->stream);
}
/// Element-wise operation, out[i]=sigmoid([in[i])
template <>
void Sigmoid<float, lang::Cuda>(const size_t num, const Blob* in, Blob* out,
                                Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::sigmoid(num, inPtr, outPtr, ctx->stream);
}
// out[i] = sign(in[i])
template <>
void Sign<float, lang::Cuda>(const size_t num, const Blob* in, Blob* out,
                             Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::sign(num, inPtr, outPtr, ctx->stream);
}

/// Element-wise operation, out[i]=sqrt([in[i])
template <>
void Sqrt<float, lang::Cuda>(const size_t num, const Blob* in, Blob* out,
                             Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::sqrt(num, inPtr, outPtr, ctx->stream);
}

/// Element-wise operation, out[i]=in[i]^2
template <>
void Square<float, lang::Cuda>(const size_t num, const Blob* in, Blob* out,
                               Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::square(num, inPtr, outPtr, ctx->stream);
}
/// out = in1 - in2
template <>
void Sub<float, lang::Cuda>(const size_t num, const Blob* in1, const Blob* in2,
                            Blob* out, Context* ctx) {
  const float* inPtr1 = static_cast<const float*>(in1->data());
  const float* inPtr2 = static_cast<const float*>(in2->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::sub(num, inPtr1, inPtr2, outPtr, ctx->stream);
}

/// sum all elements of input into out
template <>
void Sum<float, lang::Cuda>(const size_t num, const Blob* in, float* out,
                            Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  cuda::sum(num, inPtr, out, ctx->stream);
}

/// Element-wise operation, out[i]=tanh([in[i])
template <>
void Tanh<float, lang::Cuda>(const size_t num, const Blob* in, Blob* out,
                             Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  cuda::tanh(num, inPtr, outPtr, ctx->stream);
}

// ================Random functions===========================================
/// Each element of out would be 1 with prob p and 0 with 1-p. 0<= p <= 1
// Get the random generator from 'ctx'
// If DType is not float, then convert the threshold to DType
template <>
void Bernoulli<float, lang::Cuda>(const size_t num, const float p, Blob* out,
                                  Context* ctx) {
  auto rgen = ctx->curand_generator;
  float* outPtr = static_cast<float*>(out->mutable_data());
  CURAND_CHECK(curandGenerateUniform(rgen, outPtr, num));
  cuda::threshold(num, p, outPtr, outPtr, ctx->stream);
}

// The random generator should be extracted from ctx.
// If DType is not float, then convert the low and high to DType
template <>
void Uniform<float, lang::Cuda>(const size_t num, const float low,
                                const float high, Blob* out, Context* ctx) {
  auto rgen = ctx->curand_generator;
  float* outPtr = static_cast<float*>(out->mutable_data());
  CURAND_CHECK(curandGenerateUniform(rgen, outPtr, num));
  cuda::mult(num, outPtr, high - low, outPtr, ctx->stream);
  cuda::add(num, outPtr, low, outPtr, ctx->stream);
}

// The random generator should be extracted from ctx.
// If DType is not float, then convert the mean and delta to DType
template <>
void Gaussian<float, lang::Cuda>(const size_t num, const float mean,
                                 const float std, Blob* out, Context* ctx) {
  auto rgen = ctx->curand_generator;
  float* outPtr = static_cast<float*>(out->mutable_data());
  CURAND_CHECK(curandGenerateNormal(rgen, outPtr, num, mean, std));
}

// =========================Blas operations==================================
// ref to http://docs.nvidia.com/cuda/cublas
template <>
void Amax<float, lang::Cuda>(const size_t num, const Blob* in, size_t* out,
                             Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  int idx = 1;
  CUBLAS_CHECK(cublasIsamax(handle, num, inPtr, 1, &idx));
  *out = idx - 1;  // cublas index starts from 1
}

/// return the index of the element with the min value.
template <>
void Amin<float, lang::Cuda>(const size_t num, const Blob* in, size_t* out,
                             Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  int idx = 1;
  CUBLAS_CHECK(cublasIsamin(handle, num, inPtr, 1, &idx));
  *out = idx - 1;
}

/// out = sum |x| for all x in in
template <>
void Asum<float, lang::Cuda>(const size_t num, const Blob* in, float* out,
                             Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  CUBLAS_CHECK(cublasSasum(handle, num, inPtr, 1, out));
}

/// out = alpha * in + out
template <>
void Axpy<float, lang::Cuda>(const size_t num, const float alpha,
                             const Blob* in, Blob* out, Context* ctx) {
  const float* inPtr = static_cast<const float*>(in->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  CUBLAS_CHECK(cublasSaxpy(handle, num, &alpha, inPtr, 1, outPtr, 1));
}

/// out = \sum_i in1[i] * in2[i]
template <>
void Dot<float, lang::Cuda>(const size_t num, const Blob* in1, const Blob* in2,
                            float* out, Context* ctx) {
  const float* inPtr1 = static_cast<const float*>(in1->data());
  const float* inPtr2 = static_cast<const float*>(in2->data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  CUBLAS_CHECK(cublasSdot(handle, num, inPtr1, 1, inPtr2, 1, out));
}
template <>
void Nrm2<float, lang::Cuda>(const size_t num, const Blob* in, float* out,
                             Context* ctx) {
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  const float* inPtr = static_cast<const float*>(in->data());
  cublasSnrm2(handle, num, inPtr, 1, out);
}
template <>
void Scale<float, lang::Cuda>(const size_t num, const float x, Blob* out,
                              Context* ctx) {
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  float* outPtr = static_cast<float*>(out->mutable_data());
  CUBLAS_CHECK(cublasSscal(handle, num, &x, outPtr, 1));
}
// NOTE: cublas uses column major order.
// http://peterwittek.com/cublas-matrix-c-style.html
template <>
void DGMM<float, lang::Cuda>(const bool side_right, const size_t nrow,
                             const size_t ncol, const Blob* M, const Blob* v,
                             Blob* out, Context* ctx) {
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  const float* MPtr = static_cast<const float*>(M->data());
  const float* vPtr = static_cast<const float*>(v->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  if (side_right) {
    CUBLAS_CHECK(cublasSdgmm(handle, CUBLAS_SIDE_LEFT, ncol, nrow, MPtr, ncol,
                             vPtr, 1, outPtr, ncol));
  } else {
    CUBLAS_CHECK(cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, ncol, nrow, MPtr, ncol,
                             vPtr, 1, outPtr, ncol));
  }
}
template <>
void GEMV<float, lang::Cuda>(bool trans, const size_t m, const size_t n,
                             const float alpha, const Blob* A, const Blob* v,
                             const float beta, Blob* out, Context* ctx) {
  const float* APtr = static_cast<const float*>(A->data());
  const float* vPtr = static_cast<const float*>(v->data());
  float* outPtr = static_cast<float*>(out->mutable_data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  if (!trans)
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, APtr, n, vPtr,
                             1, &beta, outPtr, 1));
  else
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, APtr, m, vPtr,
                             1, &beta, outPtr, 1));
}

// http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
template <>
void GEMM<float, lang::Cuda>(const bool transA, const bool transB,
                             const size_t nrowA, const size_t ncolB,
                             const size_t ncolA, const float alpha,
                             const Blob* A, const Blob* B, const float beta,
                             Blob* C, Context* ctx) {
  auto transa = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto transb = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = transA ? nrowA : ncolA;
  int ldb = transB ? ncolA : ncolB;
  int ldc = ncolB;
  const float* APtr = static_cast<const float*>(A->data());
  const float* BPtr = static_cast<const float*>(B->data());
  float* CPtr = static_cast<float*>(C->mutable_data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  CUBLAS_CHECK(cublasSgemm(handle, transb, transa, ncolB, nrowA, ncolA, &alpha,
                           BPtr, ldb, APtr, lda, &beta, CPtr, ldc));
}

template <>
void ComputeCrossEntropy<float, lang::Cuda>(const size_t batchsize,
                                            const size_t dim, const Blob *p,
                                            const Blob *t, Blob *loss,
                                            Context *ctx) {
  const float *pPtr = static_cast<const float *>(p->data());
  const int *tPtr = static_cast<const int *>(t->data());
  float *lossPtr = static_cast<float *>(loss->mutable_data());
  cuda::ComputeCrossEntropy(batchsize, dim, pPtr, tPtr, lossPtr, ctx->stream);
}
template <>
void SoftmaxCrossEntropyBwd<float, lang::Cuda>(const size_t batchsize,
                                               const size_t dim, const Blob *p,
                                               const Blob *t, Blob *grad,
                                               Context *ctx) {
  CHECK_EQ(p, grad) << "Use the same pointer to optimize performance";
  const float *pPtr = static_cast<const float *>(p->data());
  const int *tPtr = static_cast<const int *>(t->data());
  float *gradPtr = static_cast<float *>(grad->mutable_data());
  cuda::SoftmaxCrossEntropyBwd(batchsize, dim, pPtr, tPtr, gradPtr,
                               ctx->stream);
}
}  // namespace singa

#endif  // USE_CUDA
#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CUDA_H_
