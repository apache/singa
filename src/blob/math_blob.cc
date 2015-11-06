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

#include "singa/blob/math_blob.h"
#ifdef SINGA_GPU
#include "singa/blob/math_kernel.h"
#endif  // SINGA_GPU

namespace singa {

/*****************************************************************************/
// shape_check function

int get_size(const std::vector<int>& shape) {
    int sum = 1;
    for (unsigned int i = 0; i < shape.size(); i++) sum *= shape[i];
    return sum;
}

/*****************************************************************************/
// class1 matrix operation


void GEMM(XPU xpu, const Blob<float> & A, const Blob<float> & B,
Blob<float> * C, float alpha, float beta) {
    if (check_shape_mmm(A, B, *C)) {
        int m = C->shape().at(0);
        int n = C->shape().at(1);
        int k = A.isTranspose() ? A.shape().at(0) : A.shape().at(1);
        bool TranA = A.isTranspose();
        bool TranB = B.isTranspose();
        if (xpu == cpu) {
            cpu_gemm(A.cpu_data(), B.cpu_data(), m, n, k, alpha, beta,
            TranA, TranB, C->mutable_cpu_data());
        }
        #ifdef SINGA_GPU
        if (xpu == gpu) {
            // gpu part
            gpu_gemm(A.gpu_data(), B.gpu_data(), m, n, k, alpha, beta,
            TranA, TranB, C->mutable_gpu_data());
        }
        #endif  // SINGA_GPU
    } else {
        // report errors here
    }
}
// C = alpha*A*B+beta*C, A, B and C are matrix

void MMDot(XPU xpu, const Blob<float> & A, const Blob<float> & B,
Blob<float> * C) {
    GEMM(xpu, A, B, C, 1, 0);
}
// A,B and C are matrix


void MVDot(XPU xpu, const Blob<float> & A, const Blob<float> & B,
Blob<float> * C) {
    if (check_shape_mvv(A, B, *C)) {
        int m = B.shape().at(0);
        int n = C->shape().at(0);
        bool TranA = A.isTranspose();
        if (xpu == cpu) {
            cpu_gemv(A.cpu_data(), B.cpu_data(), m, n, 1, 0, TranA,
            C->mutable_cpu_data());
        }
        #ifdef SINGA_GPU
        if (xpu == gpu) {
            // gpu part
            gpu_gemv(A.gpu_data(), B.gpu_data(), m, n, 1, 0, TranA,
            C->mutable_gpu_data());
        }
        #endif  // SINGA_GPU
    } else {
        // report errors here
    }
}
// A is matrix,B and C are vector


void VVDot(XPU xpu, const Blob<float> & A, const Blob<float> & B,
Blob<float> * C) {
    if (check_shape_vvm(A, B, *C)) {
        int m = C->shape().at(0);
        int n = C->shape().at(1);
        if (xpu == cpu) {
            cpu_gemm(A.cpu_data(), B.cpu_data(), m, n, 1, 1, 0,
            false, false, C->mutable_cpu_data());
        }
        #ifdef SINGA_GPU
        if (xpu == gpu) {
            // gpu part
            gpu_gemm(A.gpu_data(), B.gpu_data(), m, n, 1, 1, 0,
            false, false, C->mutable_gpu_data());
        }
        #endif  // SINGA_GPU
    } else {
        // report errors here
    }
}
// C is matrix,A and B are vector


float VVdot(XPU xpu, const Blob<float> & A, const Blob<float> & B) {
    float res = 0;
    if (check_shape_equal(A, B, B)) {
        int n = get_size(A.shape());
        if (xpu == cpu) {
            res = cpu_dot(A.cpu_data(), B.cpu_data(), n);
        }
        #ifdef SINGA_GPU
        if (xpu == gpu) {
            // gpu part
            res = gpu_dot(A.gpu_data(), B.gpu_data(), n);
        }
        #endif  // SINGA_GPU
    } else {
        // report errors here
    }
    return res;
}
// A and B are vectors

void AXPY(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha) {
    if (check_shape_equal(A, *B, *B)) {
        if (xpu == cpu) {
            cpu_axpy(A.cpu_data(), get_size(A.shape()),
            alpha, B->mutable_cpu_data());
        }
        #ifdef SINGA_GPU
        if (xpu == gpu) {
            gpu_axpy(A.gpu_data(), get_size(A.shape()),
            alpha, B->mutable_gpu_data());
        }
        #endif  // SINGA_GPU
    } else {
        // report errors here
    }
}
// element-wise operation: Bi = alpha*Ai+Bi  A and B should have the same size

inline void Repmat(XPU xpu, const Blob<float> & A, Blob<float> * B) {
    MVAdd(xpu, A, B, 1, 0);
}
// A is a vector, B is a matrix , let each row of B to be A

void MVAdd(XPU xpu, const Blob<float> & A, Blob<float> * B,
float alpha, float beta) {
    if (check_shape_mv(*B, A)) {
        int m = get_size(A.shape());
        int n = get_size(B->shape()) / m;
        if (xpu == cpu) {
            const float * univ = cpu_uni_vec(n);
            cpu_gemm(A.cpu_data(), univ, m, n, 1, alpha, beta,
            false, false, B->mutable_cpu_data());
            delete univ;
        }
        #ifdef SINGA_GPU
        if (xpu == gpu) {
            singa_gpu_add_vec_row(B->gpu_data(),
            A.gpu_data(), A.gpu_data(), m, n, n);
            // gpu part
        }
        #endif  // SINGA_GPU
    } else {
        // report errors here
    }
}
// A is a vector, B is a matrix , Bij = alpha*Ai+beta*Bij
// will use gemm. faster than general expand_f

void MVSum(XPU xpu, const Blob<float> & A, Blob<float> * B,
float alpha, float beta) {
    if (check_shape_mv(A, *B)) {
        int m = get_size(B->shape());
        int n = get_size(A.shape()) / m;
        if (xpu == cpu) {
            const float * univ = cpu_uni_vec(n);
            cpu_gemm(A.cpu_data(), univ, m, 1, n, alpha, beta,
            false, false, B->mutable_cpu_data());
            delete univ;
        }
        #ifdef SINGA_GPU
        if (xpu == gpu) {
            singa_gpu_sum_col(A.gpu_data(), B->gpu_data(), m, n, n);
            // gpu part
        }
        #endif  // SINGA_GPU
    } else {
        // report errors here
    }
}
// B is a vector, A is a matrix , Bi = \sigma_j_{alpha*Aij}+beta*Bi
// will use gemm. faster than general reduce_f

}  // namespace singa

