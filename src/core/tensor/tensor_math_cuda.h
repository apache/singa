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

namespace singa {

// TODO(wangwei) Clean implementations following comments in tensor_math_cpp.h.
// TODO(wangwei) optimize using stream
template <>
void Add<float, lang::Cuda>(int count, const Blob *lhs, const Blob *rhs,
                            Blob *ret, Context *ctx) {
  const float *a = static_cast<const float *>(lhs->data());
  const float *b = static_cast<const float *>(rhs->data());
  float *c = static_cast<float *>(ret->mutable_data());
  cuda::add(count, a, b, c);
}

// TODO(wangwei) optimize using stream
template <>
void Sub<float, lang::Cuda>(int count, const Blob *lhs, const Blob *rhs,
                            Blob *ret, Context *ctx) {
  const float *a = static_cast<const float *>(lhs->data());
  const float *b = static_cast<const float *>(rhs->data());
  float *c = static_cast<float *>(ret->mutable_data());
  cuda::sub(count, a, b, c);
}

template <>
void EltwiseMult<float, lang::Cuda>(int count, const Blob *input, float x,
                                    Blob *ret, Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(input->data());
  cuda::mult(count, lptr, x, dptr);
}
// TODO(wangwei) optimize using stream
template <>
void Square<float, lang::Cuda>(int count, const Blob *input, Blob *ret,
                               Context *ctx) {
  const float *in = static_cast<const float *>(input->data());
  float *out = static_cast<float *>(ret->mutable_data());
  cuda::square(count, in, out);
}

// sum all elements of input into ret
// TODO(wangwei) optimize using stream
template <>
void Sum<float, lang::Cuda>(int count, const Blob *input, float *ret,
                            Context *ctx) {
  const float *in = static_cast<const float *>(input->data());
  cuda::sum(count, in, ret);
}

// follow the consistency guide of math API
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
void Div<float, lang::Cuda>(const size_t num, const float alpha, const Blob *in,
                            Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  cuda::Div(num, alpha, inPtr, outPtr, ctx->stream);
}

// NOTE: cublas uses column major order.
// http://peterwittek.com/cublas-matrix-c-style.html
template <>
void DGMM<float, lang::Cuda>(const bool side_right, const size_t nrow,
                             const size_t ncol, const Blob *M, const Blob *v,
                             Blob *out, Context *ctx) {
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  const float *MPtr = static_cast<const float *>(M->data());
  const float *vPtr = static_cast<const float *>(v->data());
  float *outPtr = static_cast<float *>(out->mutable_data());
  if (side_right) {
    CUBLAS_CHECK(cublasSdgmm(handle, CUBLAS_SIDE_LEFT, ncol, nrow, MPtr, ncol,
                             vPtr, 1, outPtr, ncol));
  } else {
    CUBLAS_CHECK(cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, ncol, nrow, MPtr, ncol,
                             vPtr, 1, outPtr, ncol));
  }
}
// http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
template <>
void GEMM<float, lang::Cuda>(const bool transA, const bool transB,
                             const size_t nrowA, const size_t ncolB,
                             const size_t ncolA, const float alpha,
                             const Blob *A, const Blob *B, const float beta,
                             Blob *C, Context *ctx) {
  auto transa = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto transb = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = transA ? nrowA : ncolA;
  int ldb = transB ? ncolA : ncolB;
  int ldc = ncolB;
  const float *APtr = static_cast<const float *>(A->data());
  const float *BPtr = static_cast<const float *>(B->data());
  float *CPtr = static_cast<float *>(C->mutable_data());
  auto handle = ctx->cublas_handle;  // TODO(wangwei) set cudastream
  CUBLAS_CHECK(cublasSgemm(handle, transb, transa, ncolB, nrowA, ncolA, &alpha,
                           BPtr, ldb, APtr, lda, &beta, CPtr, ldc));
}

template <>
void GE<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                                   Blob* out, Context *ctx) {
  float* outPtr = static_cast<float*>(out->mutable_data());
  const float* inPtr = static_cast<const float*>(in->data());
  cuda::GE(num, inPtr, x, outPtr, ctx->stream);
}
template <>
void GT<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                                   Blob* out,  Context *ctx) {
  float* outPtr = static_cast<float*>(out->mutable_data());
  const float* inPtr = static_cast<const float*>(in->data());
  cuda::GT(num, inPtr, x, outPtr, ctx->stream);
}
template <>
void LE<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                                   Blob* out, Context *ctx) {
  float* outPtr = static_cast<float*>(out->mutable_data());
  const float* inPtr = static_cast<const float*>(in->data());
  cuda::LE(num, inPtr, x, outPtr, ctx->stream);
}
template <>
void LT<float, lang::Cuda>(const size_t num, const Blob* in, const float x,
                                   Blob* out,  Context *ctx) {
  float* outPtr = static_cast<float*>(out->mutable_data());
  const float* inPtr = static_cast<const float*>(in->data());
  cuda::LT(num, inPtr, x, outPtr, ctx->stream);
}

template<>
void Set<float, lang::Cuda>(const size_t num, const float x, Blob *out,
                            Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  cuda::Set(num, x, outPtr, ctx->stream);
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

