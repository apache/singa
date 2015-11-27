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
#include <cublas_v2.h>
#include "singa/utils/math_kernel.h"
#endif  // USE_GPU

namespace singa {

namespace op {

/**
 * b = e^a
 */
template<typename Dtype>
struct Exp {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = exp(a);
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_exp(a, b, n);
  }
#endif  // USE_GPU
};
/**
 * b = log(a), base is e
 */
template<typename Dtype>
struct Log {
  inline static void Map(const Dtype & a, Dtype *b) {
    *b = log(a);
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_log(a, b, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Sigmoid {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = 1.0f / (1.0f + expf(-a));
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_sigmoid(a, b, n);
  }
#endif  // USE_GPU
};
template<typename Dtype>
struct SigmoidGrad {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = a * (1.0f - a);
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_sigmoid_grad(a, b, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Relu {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = std::max(a, 0.0f);
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_relu(a, b, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct ReluGrad {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = a > 0 ? 1 : 0;
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_relu_grad(a, b, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Tanh {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = tanhf(a);
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_tanh(a, b, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct TanhGrad {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = 1 - a * a;
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_tanh_grad(a, b, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Softplus {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = logf(1 + expf(a));
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_softplus(a, b, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct SoftplusGrad {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = 1.0f / (1.0f + expf(-a));
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_softplus_grad(a, b, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Square {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = a * a;
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_square(a, b, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct SquareGrad {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = 2 * sqrt(a);
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_square_grad(a, b, 1, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Sqrt {
  inline static void Map(const Dtype & a, Dtype * b) {
    *b = sqrt(a);
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a, Dtype * b, int n) {
    singa::singa_gpu_sqrt(a, b, n);
  }
#endif  // USE_GPU
};

/*********************************************************************/
/**
 * c = pow(a, b), i.e., c = a^b
 */
template<typename Dtype>
struct Pow {
  inline static void Map(const Dtype & a, const Dtype &b, Dtype * c) {
    *c = pow(a, b);
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a,
      const Dtype * b, Dtype * c, int n) {
    singa::singa_gpu_pow(a, b, c, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Add {
  inline static void Map(const Dtype & a, const Dtype & b, Dtype * c) {
    *c =  a + b;
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a,
      const Dtype * b, Dtype * c, int n) {
//    singa::singa_gpu_add(a, b, c, n); // TODO(haibo)
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Sub {
  inline static void Map(const Dtype & a, const Dtype & b, Dtype * c) {
    *c =  a - b;
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a,
      const Dtype * b, Dtype * c, int n) {
//    singa::singa_gpu_add(a, b, c, n);  // TODO(haibo)
  }
#endif  // USE_GPU
};


template<typename Dtype>
struct Mult {
  inline static void Map(const Dtype & a, const Dtype & b, Dtype * c) {
    *c =  a * b;
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a,
      const Dtype * b, Dtype * c, int n) {
    singa::singa_gpu_mult(a, b, c, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Div {
  inline static void Map(const Dtype & a, const Dtype & b, Dtype * c) {
    *c =  a / b;
  }
#ifdef USE_GPU
  inline static void CudaMap(const Dtype * a,
      const Dtype * b, Dtype * c, int n) {
    singa::singa_gpu_div(a, b, c, n);
  }
#endif  // USE_GPU
};


/*********************************************************************/
template<typename Dtype>
struct Set {
  inline static void Map(Dtype alpha, Dtype * a) {
    *a = alpha;
  }
#ifdef USE_GPU
  inline static void CudaMap(Dtype alpha, Dtype * a, int n) {
    singa::singa_gpu_set_value(a, alpha, n);
  }
#endif  // USE_GPU
};

template<typename Dtype>
struct Threshold {
  inline static void Map(Dtype alpha, const Dtype & a, Dtype * b) {
    *b =  a < alpha ? 1.0f : 0.0f;
  }
#ifdef USE_GPU
  inline static void CudaMap(Dtype alpha,  const Dtype * a,
      Dtype * b, int n) {
    singa::singa_gpu_threshold(a, b, alpha, n);
  }
#endif  // USE_GPU
};

};  // namespace op

};  // namespace singa

#endif  // SINGA_UTILS_SINGA_OP_H_
