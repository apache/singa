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
#ifndef SINGA_CORE_TENSOR_TENSOR_MATH_CPP_H_
#define SINGA_CORE_TENSOR_TENSOR_MATH_CPP_H_
#include "./tensor_math.h"
#include "singa/core/common.h"
#include <math.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

/// TODO(wangwei) Clean the implementations following the comments in
/// tensor_math.h.
/// For Blob argument xxx, name its pointer as xxxPtr.
namespace singa {
template <>
void Square<float, lang::Cpp>(int count, const Blob *input, Blob *ret,
                              Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *in = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = in[i] * in[i];
  }
}

template <>
void Add<float, lang::Cpp>(int count, const Blob *lhs, const Blob *rhs,
                           Blob *ret, Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(lhs->data());
  const float *rptr = static_cast<const float *>(rhs->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = lptr[i] + rptr[i];
  }
}

template <>
void Add<float, lang::Cpp>(int count, const Blob *input, float x, Blob *ret,
                           Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = lptr[i] + x;
  }
}

template <>
void Sub<float, lang::Cpp>(int count, const Blob *lhs, const Blob *rhs,
                           Blob *ret, Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(lhs->data());
  const float *rptr = static_cast<const float *>(rhs->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = lptr[i] - rptr[i];
  }
}

// sum all elements of input into ret
// TODO(wangwei) optimize using omp
template <>
void Sum<float, lang::Cpp>(int count, const Blob *input, float *ret,
                           Context *ctx) {
  float s = 0.f;
  const float *in = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    s += in[i];
  }
  *ret = s;
}

template <>
void EltwiseMult<float, lang::Cpp>(int count, const Blob *input, float x,
                                   Blob *ret, Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = lptr[i] * x;
  }
}

template <>
void EltwiseMult<float, lang::Cpp>(int count, const Blob *lhs, const Blob *rhs,
                                   Blob *ret, Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(lhs->data());
  const float *rptr = static_cast<const float *>(rhs->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = lptr[i] * rptr[i];
  }
}

template <>
void Exp<float, lang::Cpp>(int count, const Blob *input, Blob *ret,
                           Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = exp(lptr[i]);
  }
}

template <>
void Log<float, lang::Cpp>(int count, const Blob *input, Blob *ret,
                           Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    CHECK_GT(lptr[i], 0.f);
    dptr[i] = log(lptr[i]);
  }
}

template <>
void Tanh<float, lang::Cpp>(int count, const Blob *input, Blob *ret,
                            Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = tanh(lptr[i]);
  }
}

template <>
void ReLU<float, lang::Cpp>(int count, const Blob *input, Blob *ret,
                            Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = (lptr[i] >= 0.f) ? lptr[i] : 0.f;
  }
}

template <>
void Sigmoid<float, lang::Cpp>(int count, const Blob *input, Blob *ret,
                               Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = 1.f / (1.f + exp(-lptr[i]));
  }
}

template <>
void Pow<float, lang::Cpp>(int count, const Blob *input, float x, Blob *ret,
                           Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = pow(lptr[i], x);
  }
}

template <>
void Pow<float, lang::Cpp>(int count, const Blob *lhs, const Blob *rhs,
                           Blob *ret, Context *ctx) {
  float *dptr = static_cast<float *>(ret->mutable_data());
  const float *lptr = static_cast<const float *>(lhs->data());
  const float *rptr = static_cast<const float *>(rhs->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = pow(lptr[i], rptr[i]);
  }
}

template <>
void Bernoulli<float, lang::Cpp>(int count, float p, Blob *ret, Context *ctx) {
  std::bernoulli_distribution distribution(p);
  float *ptr = static_cast<float *>(ret->mutable_data());
  for (int i = 0; i < count; i++) {
    ptr[i] = distribution(ctx->random_generator) ? 1.0f : 0.0f;
  }
}

template <>
void Uniform<float, lang::Cpp>(int count, float low, float high, Blob *ret,
                               Context *ctx) {
  std::uniform_real_distribution<float> distribution(low, high);
  float *ptr = static_cast<float *>(ret->mutable_data());
  for (int i = 0; i < count; i++) {
    ptr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}

template <>
void Gaussian<float, lang::Cpp>(int count, float mean, float std, Blob *ret,
                                Context *ctx) {
  std::normal_distribution<float> distribution(mean, std);
  float *ptr = static_cast<float *>(ret->mutable_data());
  for (int i = 0; i < count; i++) {
    ptr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}

// follow the consistency guide of math API
template <>
void Div<float, lang::Cpp>(const size_t num, const float alpha, const Blob *in,
                           Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) outPtr[i] = alpha / inPtr[i];
}
template <>
void LT<float, lang::Cpp>(const size_t num, const Blob *in, const float x,
                          Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = (inPtr[i] < x) ? 1.f : 0.f;
  }
}

template <>
void DGMM<float, lang::Cpp>(const bool side_right, const size_t nrow,
                            const size_t ncol, const Blob *M, const Blob *v,
                            Blob *out, Context *ctx) {
  const float *MPtr = static_cast<const float *>(M->data());
  const float *vPtr = static_cast<const float *>(v->data());
  float *outPtr = static_cast<float *>(out->mutable_data());
  if (side_right) {
    for (size_t r = 0; r < nrow; r++) {
      size_t offset = r * ncol;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[offset + c] = MPtr[offset + c] * vPtr[c];
      }
    }
  } else {
    for (size_t r = 0; r < nrow; r++) {
      size_t offset = r * ncol;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[offset + c] = MPtr[offset + c] * vPtr[r];
      }
    }
  }
}

template <>
void Set<float, lang::Cpp>(const size_t num, const float x, Blob *out,
                           Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  for (size_t i = 0; i < num; i++) outPtr[i] = x;
}
template <>
void LE<float, lang::Cpp>(const size_t num, const Blob *in, const float x,
                          Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = (inPtr[i] <= x) ? 1.f : 0.f;
  }
}

template <>
void GT<float, lang::Cpp>(const size_t num, const Blob *in, const float x,
                          Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = (inPtr[i] > x) ? 1.f : 0.f;
  }
}

template <>
void GE<float, lang::Cpp>(const size_t num, const Blob *in, const float x,
                          Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = (inPtr[i] >= x) ? 1.f : 0.f;
  }
}

#ifdef USE_CBLAS
template <>
void Dot<float, lang::Cpp>(const size_t num, const Blob *in1, const Blob *in2,
                           float *out, Context *ctx) {
  const float *in1Ptr = static_cast<const float *>(in1->data());
  const float *in2Ptr = static_cast<const float *>(in2->data());
  *out = cblas_sdot(num, in1Ptr, 1, in2Ptr, 1);
}

template <>
void GEMM<float, lang::Cpp>(const bool transA, const bool transB,
                            const size_t nrowA, const size_t ncolB,
                            const size_t ncolA, const float alpha,
                            const Blob *A, const Blob *B, const float beta,
                            Blob *C, Context *ctx) {
  auto transa = transA ? CblasTrans : CblasNoTrans;
  auto transb = transB ? CblasTrans : CblasNoTrans;
  auto lda = transA ? nrowA : ncolA;
  auto ldb = transB ? ncolA : ncolB;
  auto ldc = ncolB;
  const float *APtr = static_cast<const float *>(A->data());
  const float *BPtr = static_cast<const float *>(B->data());
  float *CPtr = static_cast<float *>(C->mutable_data());
  cblas_sgemm(CblasRowMajor, transa, transb, nrowA, ncolB, ncolA, alpha, APtr,
              lda, BPtr, ldb, beta, CPtr, ldc);
}

#endif  // USE_CBLAS

}  // namespace singa

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CPP_H_
