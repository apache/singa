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

#ifndef SINGA_BLOB_SINGA_OP_H_
#define SINGA_BLOB_SINGA_OP_H_

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
// #include "cublas_v2.h"
#include "singa/blob/math_kernel.h"


namespace singa {
    enum XPU { cpu, gpu, any};

namespace op {
struct Set {
    inline static void Map(float alpha, float * a) {
        *a = alpha;
    }
    inline static void CudaMap(float alpha, float * a, int n) {
        singa::singa_gpu_set_value(a, alpha, n);
    }
};

struct Scale {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = alpha * a;
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_scale(a, b, alpha, n);
    }
};

struct Scale_grad {
    inline static void Map(float alpha,  float * output) {
        *output = alpha;
    }
    inline static void CudaMap(float alpha,  float * output, int n) {
        singa::singa_gpu_scale_grad(output, alpha, n);
    }
};

struct Exp {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = pow(a, alpha);
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_exp(a, b, alpha, n);
    }
};

struct Exp_grad {
    inline static void Map(float alpha,  const float & a, float * b) {
        // log is the natrual log based on e
        *b = a * log(alpha);
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_exp_grad(a, b, alpha, n);
    }
};

struct Gsigmoid {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = 1.0f / (1.0f + expf(-a * alpha));
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_sigmoid(a, b, alpha, n);
    }
};

struct Gsigmoid_grad {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = alpha * a * (1.0f - a);
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_sigmoid_grad(a, b, alpha, n);
    }
};

struct Grelu {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = (1 - alpha) * std::max(a, 0.0f) + alpha * a;
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_relu(a, b, alpha, n);
    }
};

struct Grelu_grad {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = a > 0.0f ? 1.0f : alpha;
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_relu_grad(a, b, alpha, n);
    }
};

struct Gtanh {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = tanhf(a * alpha);
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_tanh(a, b, alpha, n);
    }
};

struct Gtanh_grad {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = alpha * (1.0f - a * a);
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_tanh_grad(a, b, alpha, n);
    }
};

struct Softplus {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = logf(1 + expf(a));
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_softplus(a, b, alpha, n);
    }
};

struct Softplus_grad {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = 1.0f / (1.0f + expf(-a));
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_softplus_grad(a, b, alpha, n);
    }
};

struct Square {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = a * a;
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_square(a, b, alpha, n);
    }
};

struct Square_grad {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = 2 * sqrt(a);
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_square_grad(a, b, alpha, n);
    }
};

struct Sqrt {
    inline static void Map(float alpha,  const float & a, float * b) {
        *b = sqrt(a);
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_sqrt(a, b, alpha, n);
    }
};

struct Threshold {
    inline static void Map(float alpha, const float & a, float * b) {
        *b =  a < alpha ? 1.0f : 0.0f;
    }
    inline static void CudaMap(float alpha,  const float * a,
    float * b, int n) {
        singa::singa_gpu_threshold(a, b, alpha, n);
    }
};

struct Add {
    inline static void Map(float alpha, float beta, const float & a,
    const float & b, float * c) {
        *c =  a + b;
    }
    inline static void CudaMap(float alpha, float beta, const float * a,
    const float * b, float * c, int n) {
        singa::singa_gpu_add(a, b, c, alpha, beta, n);
    }
};

struct Sub {
    inline static void Map(float alpha, float beta, const float & a,
    const float & b, float * c) {
        *c =  a - b;
    }
    inline static void CudaMap(float alpha, float beta, const float * a,
    const float * b, float * c, int n) {
        singa::singa_gpu_sub(a, b, c, alpha, beta, n);
    }
};

struct Mult {
    inline static void Map(float alpha, float beta, const float & a,
    const float & b, float * c) {
        *c =  a * b;
    }
    inline static void CudaMap(float alpha, float beta, const float * a,
    const float * b, float * c, int n) {
        singa::singa_gpu_mult(a, b, c, alpha, beta, n);
    }
};

struct Div {
    inline static void Map(float alpha, float beta, const float & a,
    const float & b, float * c) {
        *c =  a / b;
    }
    inline static void CudaMap(float alpha, float beta, const float * a,
    const float * b, float * c, int n) {
        singa::singa_gpu_div(a, b, c, alpha, beta, n);
    }
};

struct Sum {
    inline static void Map(const float * a, int n, float * b) {
        *b = 0;
        for (int i = 0 ; i < n ; i++) {
                    *b += a[i];
        }
    }

    inline static void CudaMap(const float * a, int n, float * b) {
        float *sum = NULL;
        cudaMalloc(<void**>(&sum), n*sizeof(float));

        singa::singa_gpu_sum_vec(a, sum, n);

        cudaMemcpyAsync(b, sum, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(sum);
    }
};

struct Expand_Div {
    inline static void Map(const float & a, int n, float * b) {
        for (int i = 0 ; i < n ; i++) {
                    b[i] /= a;
        }
    }
    inline static void CudaMap(const float & a, int n, float * b) {
        singa::singa_gpu_scale(b, b, a, n);
    }
};

struct Repmat {
    inline static void Map(const float & a, int n, float * b) {
        for (int i = 0 ; i < n ; i++) {
                    b[i] = a;
        }
    }
    inline static void CudaMap(const float & a, int n, float * b) {
        singa::singa_gpu_set_value(b, a, n);
    }
};

};  // namespace op

};  // namespace singa



#endif  // SINGA_BLOB_SINGA_OP_H_
