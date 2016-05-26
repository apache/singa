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
#include "singa/core/common.h"

namespace singa {

// TODO(wangwei) optimize using stream
template<>
void Add<float, lang::Cuda>(int count, const Blob* lhs, const Blob* rhs,
                        Blob* ret, Context* ctx) {
  const float* a = static_cast<const float*> (lhs->data());
  const float* b = static_cast<const float*> (rhs->data());
  float* c = static_cast<float*> (ret->mutable_data());
  cuda::add(count, a, b, c);
}

// TODO(wangwei) optimize using stream
template<>
void Sub<float, lang::Cuda>(int count, const Blob* lhs, const Blob* rhs,
                        Blob* ret, Context* ctx) {
  const float* a = static_cast<const float*> (lhs->data());
  const float* b = static_cast<const float*> (rhs->data());
  float* c = static_cast<float*> (ret->mutable_data());
  cuda::sub(count, a, b, c);
}

template <>
void EltwiseMult<float, lang::Cuda>(int count, const Blob* input, float x,
    Blob* ret, Context* ctx)
{
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* lptr = static_cast<const float*>(input->data());
  cuda::mult(count, lptr, x, dptr);
}
// TODO(wangwei) optimize using stream
template <>
void Square<float, lang::Cuda>(int count, const Blob* input, Blob* ret,
                            Context* ctx) {
  const float* in = static_cast<const float*>(input->data());
  float* out = static_cast<float*>(ret->mutable_data());
  cuda::square(count, in, out);
}
// sum all elements of input into ret
// TODO(wangwei) optimize using stream
template <>
void Sum<float, lang::Cuda>(int count, const Blob* input, float* ret,
                            Context* ctx) {
  const float* in = static_cast<const float*>(input->data());
  cuda::sum(count, in, ret);
}

// TODO(wangwei) optimize using stream
template <>
void SumRows<float, lang::Cuda>(int nrow, int ncol, const Blob* input,
                                Blob* ret, Context* ctx) {
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* in = static_cast<const float*>(input->data());
  cuda::sum_row(nrow, ncol, ncol, in, dptr);
}

// Sum the rows of the input matrix into a vector
// TODO(wangwei) optimize using stream
template <>
void SumColumns<float, lang::Cuda>(int nrow, int ncol, const Blob* input,
                                   Blob* ret, Context* ctx) {
  float* dptr = static_cast<float*>(ret->mutable_data());
  const float* in = static_cast<const float*>(input->data());
  cuda::sum_col(nrow, ncol, ncol, in, dptr);
}
}


#endif  // USE_CUDA
#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CUDA_H_
