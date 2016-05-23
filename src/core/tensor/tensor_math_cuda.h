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
#include "./tensor_math.h"
#include "singa_config.h"
#include "singa/core/common.h"


namespace singa {

#ifdef USE_CUDA
template<>
void Add<float, lang::Cuda>(int count, const Blob* lhs, const Blob* rhs,
                        Blob* ret, Context* ctx) {
  /*
  cublasSetStream(ctx->cublas_handle, ctx->stream);
  const float* lptr = static_cast<const float*>(lhs->data());
  const float* rptr = static_cast<const float*>(rhs->data());
  float* ptr = static_cast<float*>(ret->mutable_data());
  cublasScopy(ctx->cublas_handle, count, lptr, 1, ptr, 1);
  cublasSaxpy(ctx->cublas_handle, 1.0f, rptr, 1, ptr, 1);
  */
}

#endif
}


#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CUDA_H_
