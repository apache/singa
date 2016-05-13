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
#include "singa/core/common.h"


namespace singa {

#ifdef USE_CUDA
template<>
void Add<float, lib::Cuda>(int count, const Blob* lhs, const Blob* rhs,
                        Blob* ret, Context* ctx) {
  cublasSetStream(ctx->handle, ctx->stream);
  cublasScopy(ctx->handle, count, lhs->data(), 1, ret->mutable_data(), 1);
  cublasSaxpy(ctx->handle, 1.0f, rhs->data(), 1, ret->mutable_data(), 1);
}

#ifdef USE_CUDNN
template<>
void Conv<float, lib::Cudnn>(const OpConf *conf,
          const Blob* input,
          const Blob* W,
          const Blob* b,
          Blob* ret,
          Context* ctx) {
  // auto conv_conf = conf->CastTo<ConvConf>();
  // conv op
}

#endif
#endif
}


#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CUDA_H_
