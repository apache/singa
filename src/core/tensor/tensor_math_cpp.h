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

#ifdef USE_CBLAS
#include <cblas.h>
#endif

/// TODO(wangwei) Clean the implementations following the comments in
/// tensor_math.h.
/// For Blob argument xxx, name its pointer as xxxPtr.
namespace singa {
template <>
void Square<float, lang::Cpp>(int count, const Blob* input,
                           Blob* ret, Context* ctx) {
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* in = static_cast<const float*>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = in[i] * in[i];
  }
}

template <>
void Add<float, lang::Cpp>(int count, const Blob* lhs, const Blob* rhs,
                           Blob* ret, Context* ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* lptr = static_cast<const float*>(lhs->data());
  const float* rptr = static_cast<const float*>(rhs->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = lptr[i] + rptr[i];
  }
}

template <>
void Sub<float, lang::Cpp>(int count, const Blob* lhs, const Blob* rhs,
                           Blob* ret, Context* ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* lptr = static_cast<const float*>(lhs->data());
  const float* rptr = static_cast<const float*>(rhs->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = lptr[i] - rptr[i];
  }
}
// sum all elements of input into ret
// TODO(wangwei) optimize using omp
template <>
void Sum<float, lang::Cpp>(int count, const Blob* input, float* ret,
    Context* ctx) {
  float s = 0.f;
  const float* in = static_cast<const float*>(input->data());
  for (int i = 0; i < count; i++) {
    s += in[i];
  }
  *ret = s;
}

// TODO(wangwei) optimize using omp
template <>
void SumRows<float, lang::Cpp>(int nrow, int ncol, const Blob* input, Blob* ret,
    Context* ctx) {
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* in = static_cast<const float*>(input->data());
  memset(dptr, 0, ncol * sizeof(float));
  for (int r = 0; r < nrow; r++) {
    for (int c = 0; c < ncol; c++) {
      dptr[c] += in[r * ncol + c];
    }
  }
}

// Sum the rows of the input matrix into a vector
// TODO(wangwei) optimize using omp
template <>
void SumColumns<float, lang::Cpp>(int nrow, int ncol, const Blob* input, Blob* ret,
    Context* ctx) {
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* in = static_cast<const float*>(input->data());
  memset(dptr, 0, ncol * sizeof(float));
  for (int r = 0; r < nrow; r++) {
    for (int c = 0; c < ncol; c++) {
      dptr[r] += in[r * ncol + c];
    }
  }
}

template <>
void EltwiseMult<float, lang::Cpp>(int count, const Blob* input, float x,
                                   Blob* ret, Context* ctx) {
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* lptr = static_cast<const float*>(input->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = lptr[i] * x;
  }
}

template <>
void EltwiseMult<float, lang::Cpp>(int count, const Blob* lhs, const Blob* rhs,
                                   Blob* ret, Context* ctx) {
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* lptr = static_cast<const float*>(lhs->data());
  const float* rptr = static_cast<const float*>(rhs->data());
  for (int i = 0; i < count; i++) {
    dptr[i] = lptr[i] * rptr[i];
  }
}

template <>
void Bernoulli<float, lang::Cpp>(int count, float p, Blob* ret, Context* ctx) {
  std::bernoulli_distribution distribution(p);
  float* ptr = static_cast<float*>(ret->mutable_data());
  for (int i = 0; i < count; i++) {
    ptr[i] = distribution(ctx->random_generator) ? 1.0f : 0.0f;
  }
}

template <>
void Uniform<float, lang::Cpp>(int count, float low, float high, Blob* ret,
                               Context* ctx) {
  std::uniform_real_distribution<float> distribution(low, high);
  float* ptr = static_cast<float*>(ret->mutable_data());
  for (int i = 0; i < count; i++) {
    ptr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}

template <>
void Gaussian<float, lang::Cpp>(int count, float mean, float std, Blob* ret,
                                Context* ctx) {
  std::normal_distribution<float> distribution(mean, std);
  float* ptr = static_cast<float*>(ret->mutable_data());
  for (int i = 0; i < count; i++) {
    ptr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}

#ifdef USE_CBLAS
template <>
void Dot<float, lang::Cpp>(int count, const Blob* lhs, const Blob* rhs,
                           float* ret, Context* ctx) {
  float dptr = ret->mutable_data(), lptr = lhs->data(), rptr = rhs->data();
  *ret = cblas_sdot(count, lptr, 1, rptr, 1);
}

#endif
}

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CPP_H_
