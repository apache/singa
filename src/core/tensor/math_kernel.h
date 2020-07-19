/************************************************************
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 *************************************************************/
#ifndef SRC_CORE_TENSOR__MATH_KERNEL_H_
#define SRC_CORE_TENSOR__MATH_KERNEL_H_

#include "singa/singa_config.h"
#ifdef USE_CUDA

#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "cuda_fp16.h"

/// TODO(wangwei) Clean the function APIs as commented in tensor_math.h
///  Add 'Context *ctx' as an argument of all cuda functions.
namespace singa {

// TODO(wangwei) make all function templates.
namespace cuda {

// 0 input
void set(const size_t n, const float v, float *out, cudaStream_t s);

// 1 input
void abs(const size_t n, const float *in, float *out, cudaStream_t s);
void sign(const size_t n, const float *in, float *out, cudaStream_t s);
void exp(const size_t n, const float *in, float *out, cudaStream_t s);
void ceil2(const size_t n, const float *in, float *out, cudaStream_t s);
void cast_float_2_int(const size_t n, const float *src, int *dst,
                      cudaStream_t s);
void cast_int_2_float(const size_t n, const int *src, float *dst,
                      cudaStream_t s);
void log(const size_t n, const float *in, float *out, cudaStream_t s);
void sqrt(const size_t n, const float *in, float *out, cudaStream_t s);
void square(const size_t n, const float *in, float *out, cudaStream_t s);
void cos(const size_t n, const float *in, float *out, cudaStream_t s);
void cosh(const size_t n, const float *in, float *out, cudaStream_t s);
void acos(const size_t n, const float *in, float *out, cudaStream_t s);
void acosh(const size_t n, const float *in, float *out, cudaStream_t s);
void sin(const size_t n, const float *in, float *out, cudaStream_t s);
void sinh(const size_t n, const float *in, float *out, cudaStream_t s);
void asin(const size_t n, const float *in, float *out, cudaStream_t s);
void asinh(const size_t n, const float *in, float *out, cudaStream_t s);
void tan(const size_t n, const float *in, float *out, cudaStream_t s);
void tanh(const size_t n, const float *in, float *out, cudaStream_t s);
void atan(const size_t n, const float *in, float *out, cudaStream_t s);
void atanh(const size_t n, const float *in, float *out, cudaStream_t s);
void relu(const size_t n, const float *in, float *out, cudaStream_t s);
void sigmoid(const size_t n, const float *in, float *out, cudaStream_t s);
void softplus(const size_t n, const float *in, float *out, cudaStream_t s);
void softsign(const size_t n, const float *in, float *out, cudaStream_t s);
void clamp(const size_t n, const float low, const float high, const float *in,
           float *out, cudaStream_t s);

void pow(const size_t n, const float *in, const float x, float *out,
         cudaStream_t s);

void add(const size_t n, const float *in, const float x, float *out,
         cudaStream_t s);

void mult(const size_t n, const float *in, const float x, float *out,
          cudaStream_t s);

void broadcast_to(const size_t n, size_t nDim, const float *in,
                  const float *shape, const float *stride, float *out,
                  cudaStream_t s);

void div(const size_t n, const float x, const float *in, float *out,
         cudaStream_t s);

void threshold(const size_t n, const float x, const float *in, float *out,
               cudaStream_t s);

void relubackward(const size_t num, const float *in1, const float *in2,
                  float *out, cudaStream_t s);

void gt(const size_t num, const float *in, const float x, float *out,
        cudaStream_t s);
void gt(const size_t num, const float *in1, const float *in2, float *out,
        cudaStream_t s);

void ge(const size_t num, const float *in, const float x, float *out,
        cudaStream_t s);
void ge(const size_t num, const float *in1, const float *in2, float *out,
        cudaStream_t s);

void lt(const size_t num, const float *in, const float x, float *out,
        cudaStream_t s);
void lt(const size_t num, const float *in1, const float *in2, float *out,
        cudaStream_t s);

void le(const size_t num, const float *in, const float x, float *out,
        cudaStream_t s);
void le(const size_t num, const float *in1, const float *in2, float *out,
        cudaStream_t s);

// 2 inputs
void pow(const size_t n, const float *in1, const float *in2, float *out,
         cudaStream_t s);

void add(const size_t n, const float *in1, const float *in2, float *out,
         cudaStream_t s);

void sub(const size_t n, const float *in1, const float *in2, float *out,
         cudaStream_t s);

void mult(const size_t n, const float *in1, const float *in2, float *out,
          cudaStream_t s);

void div(const size_t n, const float *in1, const float *in2, float *out,
         cudaStream_t s);

// void sum(const size_t n, const float *in, float *out, cudaStream_t s);

void ComputeCrossEntropy(bool int_target, const size_t batchsize,
                         const size_t dim, const float *p, const int *t,
                         float *loss, cudaStream_t stream);
void SoftmaxCrossEntropyBwd(bool int_target, const size_t batchsize,
                            const size_t dim, const float *p, const int *t,
                            float *grad, cudaStream_t stream);

void RowMax(const size_t nrow, const size_t ncol, const float *inPtr,
            float *outPtr, cudaStream_t stream);

void float2half(const size_t n, const float *in, __half *out, cudaStream_t s);

void half2float(const size_t n, const __half *in, float *out, cudaStream_t s);

void sparsabs(const size_t n, const float threshold, const float *in,
              float *out, cudaStream_t s);

void sparsindex(const size_t n, const float *in, int *out, cudaStream_t s);

void generateindex(const size_t n, int *out, cudaStream_t s);

void removezeroval(const size_t n, float *in, cudaStream_t s);

void removezeroidx(const size_t n, int *in, cudaStream_t s, int *address);

void sortbykey(const size_t n, float *key, int *value, cudaStream_t s);

}  // namespace cuda

}  // namespace singa

#endif
#endif  // SRC_CORE_TENSOR__MATH_KERNEL_H_
