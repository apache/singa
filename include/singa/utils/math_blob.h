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
#include "singa/utils/blob.h"
#include "singa/utils/singa_op.h"
#include "singa/utils/math_addr.h"


namespace singa {
enum XPU {cpu, gpu, any};

/************* BLAS level 1 *****************/
/**
 * Scale each element of A with alpha, and put the result into B.
 * Bi = alpha*Ai
 * Use blas scale internally.
 */
template<typename Dtype>
void Scale(XPU xpu, Dtype alpha, const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(A.count(), B->count());
  if (xpu == cpu)
    cpu_scale(A.count(), alpha, A.cpu_data(), B->mutable_cpu_data());
#ifdef USE_GPU
#endif
}

/**
 * Element-wise operation: Bi = alpha*Ai+Bi. A and B should have the same size
 */
template<typename Dtype>
void AXPY(XPU xpu, Dtype alpha, const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(A.count(), B->count());
  if (xpu == cpu) {
    cpu_axpy(A.cpu_data(), A.count(),
        alpha, B->mutable_cpu_data());
  }
#ifdef USE_GPU
  if (xpu == gpu) {
    gpu_axpy(A.gpu_data(), A.count(),
        alpha, B->mutable_gpu_data());
  }
#endif  // USE_GPU
}

/************* BLAS level 2 *****************/
/**
 * Matrix vector multiplication, C = alpha A(.T) * B + beta C.
 * Strict shape checking:
 * - dim of A ==2
 *   columsn of A(.T) == B.count()
 * - rows of A(.T) == C.count()
 *
 * @param[in] alpha
 * @param[in] beta
 * @param[in] A, matrix
 * @param[in] B, vector
 * @param[in, out] C, vector
 */
template<typename Dtype>
void GEMV(XPU xpu, Dtype alpha, Dtype beta, const Blob<Dtype>& A,
    const Blob<Dtype>& B, Blob<Dtype>* C) {
  CHECK_EQ(A.shape().size(), 2) << "A must be a matrix";
  int a1, a2, m, n;
  a1 = A.transpose() ? A.shape(1) : A.shape(0);
  a2 = A.transpose() ? A.shape(0) : A.shape(1);
  m = B.count();
  n = C->count();
  CHECK_EQ(a2, m) << "# columns of A(.T) must = length of B";
  CHECK_EQ(a1, n) << "# rows of A(.T) must = length of C";

  bool TranA = A.transpose();
  if (xpu == cpu) {
    cpu_gemv(A.cpu_data(), B.cpu_data(), m, n, alpha, beta, TranA,
        C->mutable_cpu_data());
  }
#ifdef USE_GPU
  if (xpu == gpu) {
    // gpu part
    gpu_gemv(A.gpu_data(), B.gpu_data(), m, n, alpha, beta, TranA,
        C->mutable_gpu_data());
  }
#endif  // USE_GPU
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
void MVDot(XPU xpu, const Blob<Dtype>& A, const Blob<Dtype>& B,
    Blob<Dtype>* C) {
  GEMV(xpu, Dtype(1), Dtype(0), A, B, C);
}

/************* BLAS level 3 *****************/
/**
 * Matrix multiplication, C = alpha A*B + beta C, A, B and C are matrix.
 *
 * Tranpose is considered for A and B.
 * Strict shape checking:
 * - all are matrix
 * - shapes match for matrix multiplication
 *
 * @param[in] alpha
 * @param[in] beta
 * @param[in] A, matrix
 * @param[in] B, matrix
 * @param[in, out] C, matrix
 */
template <typename Dtype>
void GEMM(XPU xpu, Dtype alpha, Dtype beta, const Blob<Dtype>& A,
    const Blob<Dtype> & B, Blob<Dtype> * C) {
  CHECK_EQ(A.shape().size(), 2);
  CHECK_EQ(B.shape().size(), 2);
  CHECK_EQ(C->shape().size(), 2);
  int a1, a2, b1, b2, m, n;
  CHECK(!C->transpose());
  a1 = A.transpose() ? A.shape(1) : A.shape(0);
  a2 = A.transpose() ? A.shape(0) : A.shape(1);
  b1 = B.transpose() ? B.shape(1) : B.shape(0);
  b2 = B.transpose() ? B.shape(0) : B.shape(1);
  m = C->shape(0);
  n = C->shape(1);
  CHECK_EQ(a2, b1);
  CHECK_EQ(a1, m);
  CHECK_EQ(b2, n);

  int k = A.transpose() ? A.shape(0) : A.shape(1);
  bool TranA = A.transpose();
  bool TranB = B.transpose();
  if (xpu == cpu) {
    cpu_gemm(A.cpu_data(), B.cpu_data(), m, n, k, alpha, beta,
        TranA, TranB, C->mutable_cpu_data());
  }
#ifdef USE_GPU
  if (xpu == gpu) {
    // gpu part
    gpu_gemm(A.gpu_data(), B.gpu_data(), m, n, k, alpha, beta,
        TranA, TranB, C->mutable_gpu_data());
  }
#endif  // USE_GPU
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
void MMDot(XPU xpu, const Blob<Dtype>& A, const Blob<Dtype>& B,
    Blob<Dtype>* C) {
  GEMM(xpu, Dtype(1), Dtype(0), A, B, C);
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
Dtype VVDot(XPU xpu, const Blob<Dtype> & A, const Blob<Dtype> & B) {
  Dtype res = 0;
  CHECK_EQ(A.count(), B.count());
  int n = A.count();
  if (xpu == cpu) {
    res = cpu_dot(A.cpu_data(), B.cpu_data(), n);
  }
#ifdef USE_GPU
  if (xpu == gpu) {
    // gpu part
    res = gpu_dot(A.gpu_data(), B.gpu_data(), n);
  }
#endif  // USE_GPU
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
void OuterProduct(XPU xpu, const Blob<Dtype>& A, const Blob<Dtype>& B,
    Blob<Dtype> * C) {
  CHECK(!C->transpose());  // do not support C.T now.

  int m = A.count();
  int n = B.count();
  CHECK_EQ(C->count(), m * n);

  if (xpu == cpu) {
    cpu_gemm(A.cpu_data(), B.cpu_data(), m, n, 1, 1, 0,
        false, false, C->mutable_cpu_data());
  }
#ifdef USE_GPU
  if (xpu == gpu) {
    // gpu part
    gpu_gemm(A.gpu_data(), B.gpu_data(), m, n, 1, 1, 0,
        false, false, C->mutable_gpu_data());
  }
#endif  // USE_GPU
}
/*********************** Element-wise functions ***********************/
/**
 * Apply the function from Op for each element in A and put the result into B,
 * i.e., Bi = Op(Ai).
 * Loose shape checking, A.count() == B.count().
 */
template<typename Op, typename Dtype>
void Map(XPU xpu, const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(A.count(), B->count()) << "Blobs must have the same size";
  if (xpu == cpu) {
    cpu_e_f<Op>(A.count(), A.cpu_data(), B->mutable_cpu_data());
  }
#ifdef SINGA_GPU
  if (xpu == gpu) {
    // gpu part
    gpu_e_f<Op>(A.count(), A.gpu_data(), B->mutable_gpu_data());
  }
#endif  // SINGA_GPU
}

/**
 * Apply the function from Op for each element in A and B, and put the result
 * into C, i.e., Ci = Op(Ai, Bi).
 * Loose shape checking, A, B and C are of the same size.
 */
template<typename Op, typename Dtype>
void Map(XPU xpu, const Blob<Dtype> & A, const Blob<Dtype> & B,
    Blob<Dtype> * C) {
  CHECK_EQ(A.count(), B.count()) << "Blobs must have the same size";
  CHECK_EQ(A.count(), C->count()) << "Blobs must have the same size";
  if (xpu == cpu) {
    cpu_e_f<Op>(A.count(), A.cpu_data(), B.cpu_data(), C->mutable_cpu_data());
  }
#ifdef SINGA_GPU
  if (xpu == gpu) {
    // gpu part
    gpu_e_f<Op>(A.count(), A.gpu_data(), B.gpu_data(), C->mutable_gpu_data());
  }
#endif  // SINGA_GPU
}

/**
 * Bi = Op(alpha, Ai)
 * Loose shape checking, A.count() == B.count().
 */
template<typename Op, typename Dtype>
void Map(XPU xpu, Dtype alpha, const Blob<Dtype>& A, Blob<Dtype>* B) {
  CHECK_EQ(A.count(), B->count()) << "Blobs must have the same size";
  if (xpu == cpu) {
    cpu_e_f<Op>(A.count(), alpha, A.cpu_data(), B->mutable_cpu_data());
  }
#ifdef SINGA_GPU
#endif  // SINGA_GPU
}
/**
 * Ci = Op(alpha, Ai, Bi)
 * Loose shape checking, A, B and C are of the same size.
 */
template<typename Op, typename Dtype>
void Map(XPU xpu, Dtype alpha, const Blob<Dtype>& A, const Blob<Dtype>& B,
    Blob<Dtype>* C) {
  CHECK_EQ(A.count(), B->count()) << "Blobs must have the same size";
  if (xpu == cpu) {
    cpu_e_f<Op>(A.count(), alpha, A.cpu_data(), B->cpu_data(),
        C->mutable_cpu_data());
  }
#ifdef SINGA_GPU
#endif  // SINGA_GPU
}

/**
 * Currently use std::copy which has shown better performance than memcpy.
 * http://stackoverflow.com/questions/4707012/c-memcpy-vs-stdcopy
 * TODO(wangwei) test blas copy vs std::copy.
 *
 * Loose shape checking, A.count() == B.count().
 */
template<typename Dtype>
void Copy(XPU xpu, const Blob<Dtype>& A, Blob<Dtype>* B) {
  CHECK_EQ(A.count(), B->count()) << "Blobs must have the same size";
  if (xpu == cpu) {
    std::copy(A.cpu_data(), A.cpu_data() + A.count(), B->mutable_cpu_data());
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

/**
 * C = A + B
 * Implemented using Copy and AXPY.
 */
template<typename Dtype>
void Add(XPU xpu, const Blob<Dtype> & A, const Blob<Dtype> & B,
    Blob<Dtype> * C) {
  Copy(A, C);
  AXPY(B, C, 1);
}

/**
 * C = A - B
 * Implemented using Copy and AXPY.
 */
template<typename Dtype>
void Sub(XPU xpu, const Blob<Dtype> & A, const Blob<Dtype> & B,
    Blob<Dtype> * C) {
  Copy(xpu, A, C);
  AXPY(xpu, B, C, -1);
}

/**
 * C = A * B, implemented using
 * Map(XPU, const Blob<Dtype>&, const Blob<Dtype>&, Blob<Dtype>*).
 */
template<typename Dtype>
void Mult(XPU xpu, const Blob<Dtype> & A, const Blob<Dtype> & B,
    Blob<Dtype> * C) {
  Map<singa::op::Mult<Dtype>>(xpu, A, B, C);
  // TODO(wangwei) use MKL's vector func
}

/**
 * C = A / B, implemented using
 * Map(XPU, const Blob<Dtype>&, const Blob<Dtype>&, Blob<Dtype>*).
 */
template<typename Dtype>
void Div(XPU xpu, const Blob<Dtype> & A, const Blob<Dtype> & B,
    Blob<Dtype> * C) {
  Map<singa::op::Div<Dtype>>(xpu, A, B, C);
  // TODO(wangwei) use MKL's vector func
}
/*************************1D<-->2D op/transform***************************/
/**
 * Add A to each column of B, i.e., Bij = alpha*Ai + beta*Bij
 * Loose shape checking, B.count() % A.count() == 0.
 * # columns of B = B.count() / A.count().
 */
template<typename Dtype>
void MVAddCol(XPU xpu, Dtype alpha, Dtype beta, const Blob<Dtype> & A,
    Blob<Dtype> * B) {
  if (B->transpose()) {
    Blob<Dtype>* tmp = Transpose(* B);
    MVAddRow(xpu, alpha, beta, A, tmp);
    delete tmp;
  } else {
    CHECK_EQ(B->count() % A.count(), 0) << "#col of B not match length of A";
    int m = A.count(), n = B->count() / m;
    if (xpu == cpu) {
      Blob<Dtype> one(n);
      one.SetValue(1);
      cpu_gemm(A.cpu_data(), one.cpu_data(), m, n, 1, alpha, beta,
          false, false, B->mutable_cpu_data());
    }
#ifdef USE_GPU
    if (xpu == gpu) {
      singa_gpu_add_vec_row(B->gpu_data(),
          A.gpu_data(), A.gpu_data(), m, n, n);
      // gpu part
    }
#endif  // USE_GPU
  }
}
/**
 * Add A to each column of B, i.e., Bij = Ai + Bij
 * Loose shape checking, B.count() % A.count() == 0.
 * # columns of B = B.count() / A.count().
 */
template<typename Dtype>
void MVAddCol(XPU xpu, const Blob<Dtype> & A, Blob<Dtype>* B) {
  MVAddCol(xpu, Dtype(1), Dtype(1), A, B);
}

/**
 * Add A to each row of B, i.e., Bij = alpha*Aj + beta*Bij
 * Loose shape checking, B.count() % A.count() == 0.
 * # rows of B = B.count() / A.count().
 */
template<typename Dtype>
void MVAddRow(XPU xpu, Dtype alpha, Dtype beta, const Blob<Dtype> & A,
    Blob<Dtype> * B) {
  if (B->transpose()) {
    Blob<Dtype>* tmp = Transpose(* B);
    MVAddCol(xpu, alpha, beta, A, tmp);
    delete tmp;
  } else {
    CHECK_EQ(B->count() % A.count(), 0) << "#col of B not match length of A";
    int m = A.count(), n = B->count() / m;
    if (xpu == cpu) {
      Blob<Dtype> one(n);
      one.SetValue(1);
      cpu_gemm(one.cpu_data(), A.cpu_data(), n, m, 1, alpha, beta,
          false, false, B->mutable_cpu_data());
    }
#ifdef USE_GPU
    if (xpu == gpu) {
      // gpu part
      singa_gpu_add_vec_row(B->gpu_data(),
          A.gpu_data(), A.gpu_data(), m, n, n);
    }
#endif  // USE_GPU
  }
}
/**
 * Add A to each row of B, i.e., Bij = Aj + Bij
 * Loose shape checking, B.count() % A.count() == 0.
 * # rows of B = B.count() / A.count().
 */
template<typename Dtype>
void MVAddRow(XPU xpu, const Blob<Dtype> & A, Blob<Dtype>* B) {
  MVAddRow(xpu, Dtype(1), Dtype(1), A, B);
}

/**
 * Copy A to each column of B, i.e., Bij = Ai
 * Loose shape checking, B.count() % A.count() == 0,
 * # columns of B = B.count() / A.count().
 */
template<typename Dtype>
void RepmatCol(XPU xpu, const Blob<Dtype> & A, Blob<Dtype> * B) {
  MVAddCol(xpu, Dtype(1), Dtype(0), A, B);
}

/**
 * Copy A to each row of B, i.e., Bij = Aj
 * Loose shape checking, B.count() % A.count() == 0,
 * # rows of B = B.count() / A.count().
 */
template<typename Dtype>
void RepmatRow(XPU xpu, const Blob<Dtype> & A, Blob<Dtype> * B) {
  MVAddRow(xpu, Dtype(1), Dtype(0), A, B);
}

/**
 * Sum all columns of matrix A to a column vector B, 
 * i.e., Bi = \sum_j {alpha*Aij}+beta*Bi
 * Loose shape checking, A.count() % B.count() == 0.
 * # columns of A = A.count() / B.count().
 */
template<typename Dtype>
void MVSumCol(XPU xpu, Dtype alpha, Dtype beta, const Blob<Dtype> & A,
    Blob<Dtype> * B) {
  CHECK_EQ(A.count() % B->count(), 0) << "length of B must = # of cols of A";
  int m = B->count(), n = A.count() / m;
  if (xpu == cpu) {
    Blob<Dtype> one(n);
    one.SetValue(1);
    cpu_gemm(A.cpu_data(), one.cpu_data(), m, 1, n, alpha, beta,
        A.transpose(), false, B->mutable_cpu_data());
  }
#ifdef USE_GPU
  if (xpu == gpu) {
    singa_gpu_sum_col(A.gpu_data(), B->gpu_data(), m, n, n);
    // gpu part (TODO check transpose case)
  }
#endif  // USE_GPU
}

/**
 * Sum all rows of matrix A to a row vector B, 
 * i.e., Bj = \sum_i {alpha*Aij}+beta*Bj
 * Loose shape checking, A.count() % B.count() == 0.
 * # rows of A = A.count() / B.count().
 */
template<typename Dtype>
void MVSumRow(XPU xpu, Dtype alpha, Dtype beta, const Blob<Dtype> & A,
    Blob<Dtype> * B) {
  CHECK_EQ(A.count() % B->count(), 0) << "length of B must = # of cols of A";
  int m = B->count(), n = A.count() / m;
  if (xpu == cpu) {
    Blob<Dtype> one(n);
    one.SetValue(1);
    cpu_gemm(one.cpu_data(), A.cpu_data(), 1, m, n, alpha, beta,
        A.transpose(), false, B->mutable_cpu_data());
  }
#ifdef USE_GPU
  if (xpu == gpu) {
    singa_gpu_sum_row(A.gpu_data(), B->gpu_data(), m, n, n);
    // gpu part (TODO check transpose case)
  }
#endif  // USE_GPU
}

/**
 * Reduce each row of A to an element of B.
 * Loose shape checking, A.count() % B.count() == 0.
 * # columns of A = A.count() / B.count().
 */
template<typename Op, typename Dtype>
void Reduce2D(XPU xpu, const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(A.count() % B->count(), 0) << "Row size not match B length";
  int m = B->count(), n = A.count() / m;
  if (xpu == cpu) {
    cpu_reduce_f<Op>(A.cpu_data(), m, n, B->mutable_cpu_data());
  }
#ifdef SINGA_GPU
  if (xpu == gpu) {
    // gpu part
    gpu_reduce_f<Op>(A.gpu_data(), m, n, B->mutable_gpu_data());
  }
#endif  // SINGA_GPU
}
/**
 * Duplicate each element of A into a row of B.
 * Loose shape checking, B.count() % A.count() == 0.
 * # columns of B = B.count() / A.count().
 */
template<typename Op, typename Dtype>
void Expand2D(XPU xpu, const Blob<Dtype> & A, Blob<Dtype> * B) {
  CHECK_EQ(B->count() % A.count(), 0) << "Row size of B not match length of A";
  int m = A.count(), n = B->count() / m;
  if (xpu == cpu) {
    cpu_expand_f<Op>(A.cpu_data(), m, n, B->mutable_cpu_data());
  }
#ifdef SINGA_GPU
  if (xpu == gpu) {
    // gpu part
    gpu_expand_f<Op>(A.gpu_data(), m, n, B->mutable_gpu_data());
  }
#endif  // SINGA_GPU
}

/**
 * Average the absolute values.
 */
template <typename Dtype>
Dtype Asum(XPU xpu, const Blob<Dtype>& A) {
  if (A.count() == 0) return Dtype(0);
  if (xpu == cpu)
    return cpu_asum(A.count(), A.cpu_data(), 1) / A.count();
  return Dtype(0); // avoid compile warning
#ifdef USE_GPU
#endif
}

}  // end of namespace singa

#endif  // SINGA_UTILS_MATH_BLOB_H_
