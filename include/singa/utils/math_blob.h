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

#ifndef SINGA_UTILS_MATH_BLOB_H_
#define SINGA_UTILS_MATH_BLOB_H_

#include <vector>
#include <algorithm>
#include <thread>
#include "singa/utils/blob.h"
#include "singa/utils/singa_op.h"
#include "singa/utils/math_addr.h"
#include "singa/utils/singleton.h"
#include "singa/utils/context.h"

namespace singa {
/**
 * \file math_blob.h is not tested thorough.
 * Only GEMM() and MMDot() MVSumRow() andMVAddRow() are used now.
 */
/************* BLAS level 1 *****************/
/**
 * Scale each element of A with alpha, and put the result into B.
 * Bi = alpha*Ai
 * Use blas scale internally.
 */
template<typename Dtype>
void Scale(Dtype alpha, Blob<Dtype> * B) {
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_scale(B->count(), alpha, B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_scale(context->cublas_handle(device), B->count(), alpha,
        B->mutable_gpu_data());
#endif
  }
}

/**
 * Element-wise operation: Bi = alpha*Ai+Bi. A and B should have the same size
 */
template<typename Dtype>
void AXPY(Dtype alpha, const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(A.count(), B->count());
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_axpy(A.count(), alpha, A.cpu_data(), B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_axpy(context->cublas_handle(device), A.count(), alpha, A.gpu_data(),
        B->mutable_gpu_data());
#endif  // USE_GPU
  }
}

/************* BLAS level 2 *****************/
/**
 * Matrix vector multiplication, C = alpha A(.T) * B + beta C.
 * Loose shape checking:
 * - dim of A >=2
 * - row of A is shape(0) (no transpose)
 * - column of A(.T) == B.count()
 * - rows of A(.T) == C.count()
 *
 * @param[in] alpha
 * @param[in] beta
 * @param[in] A, matrix
 * @param[in] B, vector
 * @param[in, out] C, vector
 */
template<typename Dtype>
void GEMV(Dtype alpha, Dtype beta, const Blob<Dtype>& A,
    const Blob<Dtype>& B, Blob<Dtype>* C) {
  CHECK_EQ(A.shape().size(), 2);
  int a1, a2, m, n;
  a1 = A.transpose() ? A.count() / A.shape(0) : A.shape(0);
  a2 = A.transpose() ? A.shape(0) : A.count() / A.shape(0);
  m = B.count();
  n = C->count();
  CHECK_EQ(a2, m) << "# columns of A(.T) must = length of B";
  CHECK_EQ(a1, n) << "# rows of A(.T) must = length of C";

  bool TranA = A.transpose();
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_gemv(A.cpu_data(), B.cpu_data(), m, n, alpha, beta, TranA,
        C->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_gemv(context->cublas_handle(device), A.gpu_data(), B.gpu_data(), m, n,
        alpha, beta, TranA, C->mutable_gpu_data());
#endif  // USE_GPU
  }
}
/**
 * Matrix vector multiplication, C = A(.T) * B, transpose is considered.
 * Loose shape checking:
 * - dim of A >=2
 * - A.count() % B.count() == 0
 * - B.count() == C.count()
 *
 * @param[in] A input matrix
 * @param[in] B input vector
 * @param[out] C output vector
 */
template <typename Dtype>
void MVDot(const Blob<Dtype>& A, const Blob<Dtype>& B,
    Blob<Dtype>* C) {
  GEMV(Dtype(1), Dtype(0), A, B, C);
}

/************* BLAS level 3 *****************/
/**
 * Matrix multiplication, C = alpha A*B + beta C, A, B and C are matrix.
 *
 * Tranpose is considered for A and B.
 * Loose shape checking:
 * - the first dimension is row (no transpose) or col (with transpose) size
 * - shapes match for matrix multiplication
 *
 * @param[in] alpha
 * @param[in] beta
 * @param[in] A, matrix
 * @param[in] B, matrix
 * @param[in, out] C, matrix
 */
template <typename Dtype>
void GEMM(Dtype alpha, Dtype beta, const Blob<Dtype>& A, const Blob<Dtype>& B,
    Blob<Dtype> * C) {
  CHECK_GE(A.shape().size(), 2);
  CHECK_GE(B.shape().size(), 2);
  CHECK_GE(C->shape().size(), 2);
  int a1, a2, b1, b2, m, n;
  CHECK(!C->transpose());
  a1 = A.transpose() ? A.count() / A.shape(0) : A.shape(0);
  a2 = A.count() / a1;
  b1 = B.transpose() ? B.count() /B.shape(0) : B.shape(0);
  b2 = B.count() / b1;
  m = C->shape(0);
  n = C->count() / m;
  CHECK_EQ(a2, b1);
  CHECK_EQ(a1, m);
  CHECK_EQ(b2, n);

  int k = a2;
  bool TranA = A.transpose();
  bool TranB = B.transpose();
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_gemm(A.cpu_data(), B.cpu_data(), m, n, k, alpha, beta, TranA, TranB,
        C->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_gemm(context->cublas_handle(device), A.gpu_data(), B.gpu_data(),
        m, n, k, alpha, beta, TranA, TranB, C->mutable_gpu_data());
#endif  // USE_GPU
  }
}
/**
 * Matrix multiplication, C = A(.T) * B(.T), transpose is considered.
 * Strict shape checking:
 * - all are matrix
 * - shapes match for matrix multiplication
 *
 * @param[in] A input matrix
 * @param[in] B input matrix
 * @param[out] C output matrix
 */
template <typename Dtype>
void MMDot(const Blob<Dtype>& A, const Blob<Dtype>& B,
    Blob<Dtype>* C) {
  GEMM(Dtype(1), Dtype(0), A, B, C);
}


/*********************** Inner and Outer product****************************/
/**
 * Inner product for two vectors.
 * Loose shape checking, A.count() == B.count.
 *
 * @param[in] A, input vector (shape checking using A.count()).
 * @param[in] B, input vector (shape checking using B.count()).
 * @return inner product value.
 */
template <typename Dtype>
Dtype VVDot(const Blob<Dtype> & A, const Blob<Dtype> & B) {
  Dtype res = 0;
  CHECK_EQ(A.count(), B.count());
  int n = A.count();
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    res = cpu_dot(A.cpu_data(), B.cpu_data(), n);
  } else {
#ifdef USE_GPU
    // gpu part
    res = gpu_dot(context->cublas_handle(device), A.gpu_data(), B.gpu_data(),
        n);
#endif  // USE_GPU
  }
  return res;
}

/**
 * Outer product, C = A ** B, transpose is disabled.
 * Loose shape checking, A.count() * B.count() == C.count()
 *
 * @param[in] A, input vector
 * @param[in] B, input vector
 * @param[out] C, output matrix
 */
template <typename Dtype>
void OuterProduct(const Blob<Dtype>& A, const Blob<Dtype>& B, Blob<Dtype> * C) {
  CHECK(!C->transpose());  // do not support C.T now.

  int m = A.count();
  int n = B.count();
  CHECK_EQ(C->count(), m * n);
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_gemm(A.cpu_data(), B.cpu_data(), m, n, 1, 1, 0, false, false,
        C->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_gemm(context->cublas_handle(device), A.gpu_data(), B.gpu_data(),
        m, n, 1, 1, 0, false, false, C->mutable_gpu_data());
#endif  // USE_GPU
  }
}
/*********************** Element-wise functions ***********************/
/**
 * Apply the function from Op for each element in A and put the result into B,
 * i.e., Bi = Op(Ai).
 * Loose shape checking, A.count() == B.count().
 */
template<typename Op, typename Dtype>
void Map(const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(A.count(), B->count()) << "Blobs must have the same size";
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_e_f<Op>(A.count(), A.cpu_data(), B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_e_f<Op>(A.count(), A.gpu_data(), B->mutable_gpu_data());
#else
    LOG(ERROR) << "Not implemented";
#endif  // USE_GPU
  }
}

/**
 * Apply the function from Op for each element in A and B, and put the result
 * into C, i.e., Ci = Op(Ai, Bi).
 * Loose shape checking, A, B and C are of the same size.
 */
template<typename Op, typename Dtype>
void Map(const Blob<Dtype> & A, const Blob<Dtype> & B, Blob<Dtype> * C) {
  CHECK_EQ(A.count(), B.count()) << "Blobs must have the same size";
  CHECK_EQ(A.count(), C->count()) << "Blobs must have the same size";
  //cpu_e_f<Op>(A.count(), A.cpu_data(), B.cpu_data(), C->mutable_cpu_data());
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_e_f<Op>(A.count(), A.cpu_data(), B.cpu_data(), C->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    // gpu part
    gpu_e_f<Op>(A.count(), A.gpu_data(), B.gpu_data(), C->mutable_gpu_data());
#endif  // USE_GPU
  }
}

/**
 * Bi = Op(alpha, Ai)
 * Loose shape checking, A.count() == B.count().
 */
template<typename Op, typename Dtype>
void Map(Dtype alpha, const Blob<Dtype>& A, Blob<Dtype>* B) {
  CHECK_EQ(A.count(), B->count()) << "Blobs must have the same size";
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_e_f<Op>(A.count(), alpha, A.cpu_data(), B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_e_f<Op>(A.count(), A.gpu_data(), alpha, B->mutable_gpu_data());
#else
    LOG(FATAL) << "Not implemented";
#endif  // USE_GPU
  }
}
/**
 * Ci = Op(alpha, Ai, Bi)
 * Loose shape checking, A, B and C are of the same size.
 */
template<typename Op, typename Dtype>
void Map(Dtype alpha, const Blob<Dtype>& A, const Blob<Dtype>& B,
    Blob<Dtype>* C) {
  CHECK_EQ(A.count(), B->count()) << "Blobs must have the same size";
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_e_f<Op>(A.count(), alpha, A.cpu_data(), B->cpu_data(),
        C->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    LOG(ERROR) << "Not implemented";
#endif  // USE_GPU
  }
}

/**
 * Currently use std::copy which has shown better performance than memcpy.
 * http://stackoverflow.com/questions/4707012/c-memcpy-vs-stdcopy
 * TODO(wangwei) test blas copy vs std::copy.
 *
 * Loose shape checking, A.count() == B.count().
 */
template<typename Dtype>
void Copy(const Blob<Dtype>& A, Blob<Dtype>* B) {
  CHECK_EQ(A.count(), B->count()) << "Blobs must have the same size";
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    std::copy(A.cpu_data(), A.cpu_data() + A.count(), B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
  CUDA_CHECK(cudaMemcpy(static_cast<Dtype*>(B->mutable_gpu_data()),
             A.gpu_data(), sizeof(Dtype) * A.count(), cudaMemcpyDefault));
#else
  LOG(FATAL) << "Not implemented";
#endif
  }
}


/**
 * B = alpha + A
 * Implemented using Copy and AXPY.
 */
template<typename Dtype>
void Add(Dtype alpha,  const Blob<Dtype> & A, Blob<Dtype> * B) {
  Map<singa::op::Add<Dtype>>(alpha, A, B);
}

/**
 * C = A + B
 * Implemented using Copy and AXPY.
 */
template<typename Dtype>
void Add(const Blob<Dtype> & A, const Blob<Dtype> & B,
    Blob<Dtype> * C) {
  Copy(A, C);
  AXPY(Dtype(1), B, C);
}

/**
 * B = alpha - A
 * Implemented using Copy and AXPY.
 */
template<typename Dtype>
void Sub(Dtype alpha, const Blob<Dtype> & A, Blob<Dtype>* B) {
  Map<singa::op::Sub<Dtype>>(alpha, A, B);
}

/**
 * C = A - B
 * Implemented using Copy and AXPY.
 */
template<typename Dtype>
void Sub(const Blob<Dtype> & A, const Blob<Dtype> & B,
    Blob<Dtype> * C) {
  Copy(A, C);
  AXPY(Dtype(-1), B, C);
}

/**
 * C = A * B, implemented using
 * Map(const Blob<Dtype>&, const Blob<Dtype>&, Blob<Dtype>*).
 */
template<typename Dtype>
void Mult(const Blob<Dtype> & A, const Blob<Dtype> & B,
    Blob<Dtype> * C) {
  Map<singa::op::Mult<Dtype>>(A, B, C);
  // TODO(wangwei) use MKL's vector func
}

/**
 * C = A / B, implemented using
 * Map(const Blob<Dtype>&, const Blob<Dtype>&, Blob<Dtype>*).
 */
template<typename Dtype>
void Div(const Blob<Dtype> & A, const Blob<Dtype> & B,
    Blob<Dtype> * C) {
  Map<singa::op::Div<Dtype>>(A, B, C);
  // TODO(wangwei) use MKL's vector func
}
/**
 * B = sqrt(A)
 */
template<typename Dtype>
void Sqrt(const Blob<Dtype> & A, Blob<Dtype>* B) {
  Map<singa::op::Sqrt<Dtype>, Dtype>(A, B);
}
/**
 * B = square(A)
 */
template<typename Dtype>
void Square(const Blob<Dtype> & A, Blob<Dtype>* B) {
  Map<singa::op::Square<Dtype>, Dtype>(A, B);
}
/**
 * B = exp(A)
 */
template<typename Dtype>
void Exp(const Blob<Dtype> & A, Blob<Dtype>* B) {
  Map<singa::op::Exp<Dtype>, Dtype>(A, B);
}
/**
 * B = log(A)
 */
template<typename Dtype>
void Log(const Blob<Dtype>& A, Blob<Dtype>* B) {
  Map<singa::op::Log<Dtype>, Dtype>(A, B);
}
/**
 * B = tanh(A)
 */
template<typename Dtype>
void Tanh(const Blob<Dtype>& A, Blob<Dtype>* B) {
  Map<singa::op::Tanh<Dtype>, Dtype>(A, B);
}
/*************************1D<-->2D op/transform***************************/
/**
 * Add A to each column of B, i.e., Bij = alpha*Ai + beta*Bij
 * Loose shape checking, B.count() % A.count() == 0.
 * # columns of B = B.count() / A.count().
 */
template<typename Dtype>
void MVAddCol(Dtype alpha, Dtype beta, const Blob<Dtype> & A, Blob<Dtype> * B) {
  if (B->transpose()) {
    B->set_transpose(false);
    MVAddRow(alpha, beta, A, B);
    B->set_transpose(true);
  } else {
    CHECK_EQ(B->count() % A.count(), 0) << "#col of B not match length of A";
    int m = A.count(), n = B->count() / m;
    Blob<Dtype> one(n);
    one.SetValue(1);
    auto context = Singleton<Context>::Instance();
    int device = context->device_id(std::this_thread::get_id());
    if (device == -1) {
      cpu_gemm(A.cpu_data(), one.cpu_data(), m, n, 1, alpha, beta, false, false,
          B->mutable_cpu_data());
    } else {
#ifdef USE_GPU
      singa_gpu_add_vec_row(A.gpu_data(), B->gpu_data(), B->mutable_gpu_data(),
          m, n, n);
#endif  // USE_GPU
    }
  }
}
/**
 * Add A to each column of B, i.e., Bij = Ai + Bij
 * Loose shape checking, B.count() % A.count() == 0.
 * # columns of B = B.count() / A.count().
 */
template<typename Dtype>
void MVAddCol(const Blob<Dtype> & A, Blob<Dtype>* B) {
  MVAddCol(Dtype(1), Dtype(1), A, B);
}

/**
 * Add A to each row of B, i.e., Bij = alpha*Aj + beta*Bij
 * Loose shape checking, B.count() % A.count() == 0.
 * # rows of B = B.count() / A.count().
 */
template<typename Dtype>
void MVAddRow(Dtype alpha, Dtype beta, const Blob<Dtype> & A, Blob<Dtype> * B) {
  if (B->transpose()) {
    B->set_transpose(false);
    MVAddCol(alpha, beta, A, B);
    B->set_transpose(true);
  } else {
    CHECK_EQ(B->count() % A.count(), 0) << "#col of B not match length of A";
    int n = A.count(), m = B->count() / n;
    auto context = Singleton<Context>::Instance();
    int device = context->device_id(std::this_thread::get_id());
    if (device == -1) {
      Blob<Dtype> one(m);
      one.SetValue(1);
      cpu_gemm(one.cpu_data(), A.cpu_data(), m, n, 1, alpha, beta,
          false, false, B->mutable_cpu_data());
    } else {
#ifdef USE_GPU
      singa_gpu_add_vec_row(A.gpu_data(), B->gpu_data(), B->mutable_gpu_data(),
          m, n, n);
#endif  // USE_GPU
    }
  }
}
/**
 * Add A to each row of B, i.e., Bij = Aj + Bij
 * Loose shape checking, B.count() % A.count() == 0.
 * # rows of B = B.count() / A.count().
 */
template<typename Dtype>
void MVAddRow(const Blob<Dtype> & A, Blob<Dtype>* B) {
  MVAddRow(Dtype(1), Dtype(1), A, B);
}

/**
 * Copy A to each column of B, i.e., Bij = Ai
 * Loose shape checking, B.count() % A.count() == 0,
 * # columns of B = B.count() / A.count().
 */
template<typename Dtype>
void RepmatCol(const Blob<Dtype> & A, Blob<Dtype> * B) {
  MVAddCol(Dtype(1), Dtype(0), A, B);
}

/**
 * Copy A to each row of B, i.e., Bij = Aj
 * Loose shape checking, B.count() % A.count() == 0,
 * # rows of B = B.count() / A.count().
 */
template<typename Dtype>
void RepmatRow(const Blob<Dtype> & A, Blob<Dtype> * B) {
  MVAddRow(Dtype(1), Dtype(0), A, B);
}

/**
 * Sum all columns of matrix A to a column vector B,
 * i.e., Bi = \sum_j {alpha*Aij}+beta*Bi
 * Loose shape checking, A.count() % B.count() == 0.
 * # columns of A = A.count() / B.count().
 */
template<typename Dtype>
void MVSumCol(Dtype alpha, Dtype beta, const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(A.count() % B->count(), 0) << "length of B must = # of cols of A";
  int m = B->count(), n = A.count() / m;
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    Blob<Dtype> one(n);
    one.SetValue(1);
    cpu_gemm(A.cpu_data(), one.cpu_data(), m, 1, n, alpha, beta,
        A.transpose(), false, B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    singa_gpu_sum_col(A.gpu_data(), B->mutable_gpu_data(), m, n, n);
    // gpu part (TODO check transpose case)
#endif  // USE_GPU
  }
}

/**
 * Sum all rows of matrix A to a row vector B,
 * i.e., Bj = \sum_i {alpha*Aij}+beta*Bj
 * Loose shape checking, A.count() % B.count() == 0.
 * # rows of A = A.count() / B.count().
 */
template<typename Dtype>
void MVSumRow(Dtype alpha, Dtype beta, const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(A.count() % B->count(), 0) << "length of B must = # of cols of A";
  int n = B->count(), m = A.count() / n;
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    Blob<Dtype> one(m);
    one.SetValue(1);
    cpu_gemm(one.cpu_data(), A.cpu_data(), 1, n, m, alpha, beta, false,
             A.transpose(), B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    singa_gpu_sum_row(A.gpu_data(), B->mutable_gpu_data(), m, n, n);
    // gpu part (TODO check transpose case)
#endif  // USE_GPU
  }
}

/**
 * Reduce each row of A to an element of B.
 * Loose shape checking, A.count() % B.count() == 0.
 * # columns of A = A.count() / B.count().
 */
template<typename Op, typename Dtype>
void Reduce2D(const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(A.count() % B->count(), 0) << "Row size not match B length";
  int m = B->count(), n = A.count() / m;
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_reduce_f<Op>(A.cpu_data(), m, n, B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    // gpu part
    gpu_reduce_f<Op>(A.gpu_data(), m, n, B->mutable_gpu_data());
#endif  // USE_GPU
  }
}
/**
 * Duplicate each element of A into a row of B.
 * Loose shape checking, B.count() % A.count() == 0.
 * # columns of B = B.count() / A.count().
 */
template<typename Op, typename Dtype>
void Expand2D(const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(B->count() % A.count(), 0) << "Row size of B not match length of A";
  int m = A.count(), n = B->count() / m;
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_expand_f<Op>(A.cpu_data(), m, n, B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_expand_f<Op>(A.gpu_data(), m, n, B->mutable_gpu_data());
#endif  // USE_GPU
  }
}

/**
 * Average the absolute values.
 */
template<typename Dtype>
Dtype Asum(const Blob<Dtype>& A) {
  if (A.count() == 0) return Dtype(0);
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  Dtype ret = Dtype(0);
  if (device == -1) {
    ret = cpu_asum(A.count(), A.cpu_data(), 1) / A.count();
  } else {
#ifdef USE_GPU
    ret = gpu_asum(context->cublas_handle(device), A.count(), A.gpu_data(), 1)
      / A.count();
#endif
  }
  return ret;
}


/*************Random Sample***************/
template<typename Dtype>
void SampleUniform(Dtype low, Dtype high, Blob<Dtype>* A) {
  auto context = Singleton<Context>::Instance();
  const auto& thread = std::this_thread::get_id();
  int device = context->device_id(thread);
  if (device == -1) {
    cpu_sample_uniform(*context->rand_generator(thread), A->count(), low, high,
        A->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_sample_uniform(context->curand_generator(thread), A->count(), low, high,
        A->mutable_gpu_data());
#else
    LOG(FATAL) << "Not implemented";
#endif
  }
}

template<typename Dtype>
void SampleGaussian(Dtype mean, Dtype std, Blob<Dtype>* A) {
  auto context = Singleton<Context>::Instance();
  const auto& thread = std::this_thread::get_id();
  int device = context->device_id(thread);
  if (device == -1) {
    cpu_sample_gaussian(*context->rand_generator(thread), A->count(), mean, std,
        A->mutable_cpu_data());
  } else {
#ifdef USE_GPU
    gpu_sample_gaussian(context->curand_generator(thread), A->count(),
        mean, std, A->mutable_gpu_data());
#endif
  }
}

/************** Other functions ****************/
template<typename Dtype>
void Softmax(int nb_rows, const Blob<Dtype>& A, Blob<Dtype>* B) {
  CHECK_GT(nb_rows, 0);
  CHECK_EQ(A.count() % nb_rows, 0);
  CHECK_EQ(A.count(), B->count());
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    cpu_softmax(nb_rows, A.count() / nb_rows, A.cpu_data(),
      B->mutable_cpu_data());
  } else {
#ifdef USE_GPU
#endif  // USE_GPU
  }
}

template<typename Dtype>
void Zero(Blob<Dtype>* B) {
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  if (device == -1) {
    B->SetValue(0);
  } else {
#ifdef USE_GPU
    cudaMemset(B->mutable_gpu_data(), 0, B->count() * sizeof(float));
#else
    LOG(FATAL) << "Not implemented";
#endif  // USE_GPU
  }
}
}  // end of namespace singa

#endif  // SINGA_UTILS_MATH_BLOB_H_
