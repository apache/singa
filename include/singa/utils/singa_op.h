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

#ifndef SINGA_UTILS_SINGA_OP_H_
#define SINGA_UTILS_SINGA_OP_H_

#include <cmath>
#include <algorithm>

#ifdef USE_GPU
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "singa/utils/math_kernel.h"
#endif  // USE_GPU

namespace singa {

namespace op {

/**
 * b = e^a
 */
template<Dtype>
struct Exp {
  inline static void Map(const float & a, float * b) {
    *b = exp(a);
  }
#ifdef USE_GPU
  inline static void CudaMap(float alpha,  const float * a,
      float * b, int n) {
    singa::singa_gpu_exp(a, b, alpha, n);
  }
#endif  // USE_GPU
};
/**
 * b = log(a), base is e
 */
template<Dtype>
struct Log {
  inline static void Map(const float & a, float *b) {
    *b = log(a);
  }
}

template<Dtype>
struct Sigmoid {
  inline static void Map(const float & a, float * b) {
    *b = 1.0f / (1.0f + expf(-a * alpha));
  }
#ifdef USE_GPU
  inline static void CudaMap(const float * a,
      float * b, int n) {
    singa::singa_gpu_sigmoid(a, b, 1, n);
  }
#endif  // USE_GPU
};
template<Dtype>
struct SigmoidGrad {
  inline static void Map(const float & a, float * b) {
    *b = a * (1.0f - a);
  }
#ifdef USE_GPU
  inline static void CudaMap(float alpha,  const float * a, float * b, int n) {
    singa::singa_gpu_sigmoid_grad(a, b, 1, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct Relu {
  inline static void Map(const float & a, float * b) {
    *b = std::max(a, 0.0f);
  }
#ifdef USE_GPU
  inline static void CudaMap(const float * a, float * b, int n) {
    singa::singa_gpu_relu(a, b, 1, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct ReluGrad {
  inline static void Map(const float & a, float * b) {
    *b = a > 0 ? 1 : 0;
  }
#ifdef USE_GPU
  inline static void CudaMap(const float * a, float * b, int n) {
    singa::singa_gpu_relu_grad(a, b, 1, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct Tanh {
  inline static void Map(const float & a, float * b) {
    *b = tanhf(a);
  }
#ifdef USE_GPU
  inline static void CudaMap(float alpha,  const float * a, float * b, int n) {
    singa::singa_gpu_tanh(a, b, alpha, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct TanhGrad {
  inline static void Map(const float & a, float * b) {
    *b = 1 - a * a;
  }
#ifdef USE_GPU
  inline static void CudaMap(float alpha,  const float * a, float * b, int n) {
    singa::singa_gpu_tanh_grad(a, b, alpha, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct Softplus {
  inline static void Map(const float & a, float * b) {
    *b = logf(1 + expf(a));
  }
#ifdef USE_GPU
  inline static void CudaMap(const float * a, float * b, int n) {
    singa::singa_gpu_softplus(a, b, 1, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct SoftplusGrad {
  inline static void Map(const float & a, float * b) {
    *b = 1.0f / (1.0f + expf(-a));
  }
#ifdef USE_GPU
  inline static void CudaMap(const float * a,
      float * b, int n) {
    singa::singa_gpu_softplus_grad(a, b, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct Square {
  inline static void Map(const float & a, float * b) {
    *b = a * a;
  }
#ifdef USE_GPU
  inline static void CudaMap(const float * a,
      float * b, int n) {
    singa::singa_gpu_square(a, b, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct SquareGrad {
  inline static void Map(const float & a, float * b) {
    *b = 2 * sqrt(a);
  }
#ifdef USE_GPU
  inline static void CudaMap(const float * a,
      float * b, int n) {
    singa::singa_gpu_square_grad(a, b, 1, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct Sqrt {
  inline static void Map(const float & a, float * b) {
    *b = sqrt(a);
  }
#ifdef USE_GPU
  inline static void CudaMap(const float * a,
      float * b, int n) {
    singa::singa_gpu_sqrt(a, b, n);
  }
#endif  // USE_GPU
};

/*********************************************************************/
/**
 * c = pow(a, b), i.e., c = a^b
 */
template<Dtype>
struct Pow {
  inline static void Map(const float & a, const float &b, float * c) {
    *c = pow(a, b);
  }
}
template<Dtype>
struct Mult {
  inline static void Map(const float & a, const float & b, float * c) {
    *c =  a * b;
  }
#ifdef USE_GPU
  inline static void CudaMap(const float* a, const float* b, float* c, int n) {
    singa::singa_gpu_mult(a, b, c, 1, 1, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct Div {
  inline static void Map(const float & a, const float & b, float * c) {
    *c =  a / b;
  }
#ifdef USE_GPU
  inline static void CudaMap(const float * a,
      const float * b, float * c, int n) {
    singa::singa_gpu_div(a, b, c, 1, 1, n);
  }
#endif  // USE_GPU
};


/*********************************************************************/
template<Dtype>
struct Set {
  inline static void Map(float alpha, float * a) {
    *a = alpha;
  }
#ifdef USE_GPU
  inline static void CudaMap(float alpha, float * a, int n) {
    singa::singa_gpu_set_value(a, alpha, n);
  }
#endif  // USE_GPU
};

template<Dtype>
struct Threshold {
  inline static void Map(float alpha, const float & a, float * b) {
    *b =  a < alpha ? 1.0f : 0.0f;
  }
#ifdef USE_GPU
  inline static void CudaMap(float alpha,  const float * a,
      float * b, int n) {
    singa::singa_gpu_threshold(a, b, alpha, n);
  }
#endif  // USE_GPU
};

};  // namespace op

};  // namespace singa

#endif  // SINGA_UTILS_SINGA_OP_H_
