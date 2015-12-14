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

#ifndef SINGA_UTILS_MATH_ADDR_H_
#define SINGA_UTILS_MATH_ADDR_H_

extern "C" {
#include <cblas.h>
}
#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include "singa/utils/singa_op.h"

namespace singa {
template<typename Dtype>
Dtype cpu_asum(int n, const Dtype* A, int inc) {
  return cblas_sasum(n, A, inc);
}

template<typename Dtype>
void cpu_gemm(const Dtype * A, const Dtype * B,
    const int m, const int n, const int k, const Dtype alpha, const Dtype beta,
    const bool TranA, const bool TranB, Dtype * C) {
  int lda, ldb;
  CBLAS_TRANSPOSE tA, tB;
  lda = TranA ? m : k;
  ldb = TranB ? k : n;
  tA = TranA ? CblasTrans : CblasNoTrans;
  tB = TranB ? CblasTrans : CblasNoTrans;
  cblas_sgemm(CblasRowMajor, tA, tB, m, n, k, alpha, A, lda,
      B, ldb, beta, C, n);
}

// should be very careful:
// m is the length of B, and n is the length of C , A is a n*m matrix
template<typename Dtype>
void cpu_gemv(const Dtype * A, const Dtype * B, const int m, const int n,
    const Dtype alpha, const Dtype beta, const bool TranA, Dtype * C) {
  CBLAS_TRANSPOSE tA;
  tA = TranA ? CblasTrans : CblasNoTrans;
  cblas_sgemv(CblasRowMajor, tA, m, n, alpha, A, n, B, 1, beta, C, 1);
}

template<typename Dtype>
void cpu_axpy(const int n, const Dtype alpha, const Dtype * A, Dtype * B) {
  cblas_saxpy(n, alpha, A, 1, B, 1);
}

template<typename Dtype>
void cpu_scale(const int n, const Dtype alpha, Dtype * A) {
  cblas_sscal(n, alpha, A, 1);
}

template<typename Dtype>
void cpu_copy(const int n, const Dtype* A, Dtype *B) {
  cblas_scopy(n, A, 1, B, 1);
}

template<typename Dtype>
Dtype cpu_dot(const Dtype * A, const Dtype * B, const int n) {
  Dtype sum = 0;
  for (int i = 0 ; i < n ; i++)
    sum += A[i] * B[i];
  return sum;
}

// element-wise
template<typename Op, typename Dtype>
void cpu_e_f(const int n, const Dtype * A, Dtype * B) {
  for (int i = 0 ; i < n ; i++) {
    Op::Map(A[i], &B[i]);
  }
}

template<typename Op, typename Dtype>
void cpu_e_f(const int n, const Dtype * A, const Dtype * B, Dtype * C) {
  for (int i = 0 ; i < n ; i++) {
    Op::Map(A[i], B[i], &C[i]);
  }
}
template<typename Op, typename Dtype>
void cpu_e_f(const int n, const Dtype alpha, const Dtype * A, Dtype * B) {
  for (int i = 0 ; i < n ; i++) {
    Op::Map(alpha, A[i], &B[i]);
  }
}

template<typename Op, typename Dtype>
void cpu_e_f(const int n, const Dtype alpha, const Dtype * A, const Dtype * B,
    Dtype * C) {
  for (int i = 0 ; i < n ; i++) {
    Op::Map(alpha, A[i], B[i], &C[i]);
  }
}
// element-wise generalized operation defined in Op


// matrix/vector expand/reduce

template<typename Op, typename Dtype>
void cpu_reduce_f(const Dtype * A, const int m, const int n, Dtype * B) {
  for (int i = 0 ; i < m ; i++) {
    Op::Map(A+i*n, n, B[i]);
  }
}
// reduce each row of A to an element of B e.g. the sum operation in softmax
template<typename Op, typename Dtype>
void cpu_expand_f(const Dtype * A, const int m, const int n, Dtype * B) {
  for (int i = 0 ; i < m ; i++) {
    Op::Map(A[i], n, B+i*n);
  }
}


template<typename Dtype>
void cpu_softmax(int nb_rows, int nb_cols, const Dtype* A, Dtype* B) {
  for (int i = 0; i < nb_rows; i++) {
    const Dtype* dptr = A + i * nb_cols;
    Dtype mmax = dptr[0];
    for (int x = 1; x < nb_cols; ++x)
      if (mmax < dptr[x]) mmax = dptr[x];
    Dtype sum = 0.0f;
    for (int x = 0; x < nb_cols; ++x) {
      dptr[x] = std::exp(dptr[x] - mmax);
      sum += dptr[x];
    }
    for (int x = 0; x < nb_cols; ++x) {
      dptr[x] /= sum;
    }
  }
}



template<typename Dtype, typename URNG>
void cpu_sample_uniform(URNG& g, int n, Dtype low, Dtype high, Dtype* A) {
  std::uniform_real_distribution<Dtype> distribution(low, high);
  for (int i = 0; i < n; i++)
    A[i] = distribution(g);
}

template<typename Dtype, typename URNG>
void cpu_sample_gaussian(URNG& g, int n, Dtype mean, Dtype std, Dtype* A) {
  std::normal_distribution<Dtype> distribution(mean, std);
  for (int i = 0; i < n; i++)
    A[i] = distribution(g);
}

#ifdef USE_GPU
template<typename Dtype>
Dtype gpu_asum(cublasHandle_t handle, int n, const Dtype* A, int inc) {
  Dtype result = 0.0;
  cublasSasum(handle, n, A, inc, &result);
  return result;
}

template<typename Dtype>
void gpu_gemm(cublasHandle_t handle, const Dtype * A, const Dtype * B,
    const int m, const int n, const int k, const Dtype alpha, const Dtype beta,
    const bool TranA, const bool TranB, Dtype * C) {
  int lda = TranA ? m : k;
  int ldb = TranB ? k : n;
  int ldc = n;
  cublasOperation_t tA = (TranA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t tB = (TranB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasSgemm(handle, tB, tA, n, m, k, &alpha, B, ldb,
      A, lda, &beta, C, ldc);
}

template<typename Dtype>
void gpu_gemv(cublasHandle_t handle, const Dtype * A, const Dtype * B,
    const int m, const int n, const Dtype alpha, const Dtype beta,
    const bool TranA, Dtype * C) {
  int lda = n;
  cublasOperation_t tA = (TranA == true) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasSgemv(handle, tA, n, m, &alpha , A, lda, B, 1, &beta, C, 1);
}

template<typename Dtype>
void gpu_axpy(cublasHandle_t handle, const int n, const Dtype alpha,
    const Dtype * A, Dtype * B) {
  cublasSaxpy(handle, n, &alpha, A, 1, B, 1);
}

template<typename Dtype>
void gpu_scale(cublasHandle_t handle, const int n, const Dtype alpha,
    Dtype * A) {
  cublasSscal(handle, n, &alpha, A, 1);
}

template<typename Dtype>
Dtype gpu_dot(cublasHandle_t handle, const Dtype * A, const Dtype * B,
    const int n) {
  Dtype result = 0.0;
  cublasSdot(handle, n, A, 1, B, 1, &result);
  return result;
}

// element-wise
template<typename Op, typename Dtype>
void gpu_e_f(const int n, const Dtype alpha, Dtype * A) {
  Op::CudaMap(alpha, A, n);
}

template<typename Op, typename Dtype>
void gpu_e_f(const int n, const Dtype * A, Dtype * B) {
  Op::CudaMap(A, B, n);
}

template<typename Op, typename Dtype>
void gpu_e_f(const int n, const Dtype * A, const Dtype * B, Dtype * C) {
  Op::CudaMap(A, B, C, n);
}

template<typename Op, typename Dtype>
void gpu_e_f(const int n, const Dtype * A, const Dtype alpha, Dtype * B) {
  Op::CudaMap(alpha, A, B, n);
}

template<typename Op, typename Dtype>
void gpu_e_f(const int n, const Dtype * A, const Dtype * B,
    const Dtype alpha, const Dtype beta, Dtype * C) {
  Op::CudaMap(alpha, beta, A, B, C, n);
}
// element-wise generalized operation defined in Op

// matrix/vector expand/reduce

template<typename Op, typename Dtype>
void gpu_reduce_f(const Dtype * A, const int m, const int n, Dtype * B) {
  for (int i = 0 ; i < m ; i++) {
    Op::CudaMap(A+i*n, n, B[i]);
  }
}
// reduce each row of A to an element of B e.g. the sum operation in softmax
template<typename Op, typename Dtype>
void gpu_expand_f(const Dtype * A, const int m, const int n, Dtype * B) {
  for (int i = 0 ; i < m ; i++) {
    Op::CudaMap(A[i], n, B+i*n);
  }
}


template<typename Dtype, typename URNG>
void gpu_sample_uniform(URNG g, int n, Dtype low, Dtype high, Dtype* A) {
  curandGenerateUniform(g, A, n);
}

template<typename Dtype, typename URNG>
void gpu_sample_gaussian(URNG g, int n, Dtype mean, Dtype std, Dtype* A) {
  curandGenerateNormal(g, A, n, mean, std);
}

// expand each element in A into a row of B
#endif  // USE_GPU

}  // namespace singa
#endif  // SINGA_UTILS_MATH_ADDR_H_
