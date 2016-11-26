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
void log(const size_t n, const float *in, float *out, cudaStream_t s);
void sqrt(const size_t n, const float *in, float *out, cudaStream_t s);
void square(const size_t n, const float *in, float *out, cudaStream_t s);
void tanh(const size_t n, const float *in, float *out, cudaStream_t s);
void relu(const size_t n, const float *in, float *out, cudaStream_t s);
void sigmoid(const size_t n, const float *in, float *out, cudaStream_t s);
void softplus(const size_t n, const float *in, float *out, cudaStream_t s);
void clamp(const size_t n, const float low, const float high, const float *in,
           float *out, cudaStream_t s);

void pow(const size_t n, const float *in, const float x, float *out,
         cudaStream_t s);

void add(const size_t n, const float *in, const float x, float *out,
         cudaStream_t s);

void mult(const size_t n, const float *in, const float x, float *out,
          cudaStream_t s);

void div(const size_t n, const float x, const float *in, float *out,
         cudaStream_t s);

void threshold(const size_t n, const float x, const float *in, float *out,
               cudaStream_t s);

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
}  // cuda

}  // namespace singa

#endif
#endif  // SRC_CORE_TENSOR__MATH_KERNEL_H_
