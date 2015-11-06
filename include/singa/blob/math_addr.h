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

#ifndef SINGA_BLOB_MATH_ADDR_H_
#define SINGA_BLOB_MATH_ADDR_H_

namespace singa {

const float * cpu_uni_vec(const int n);

void cpu_gemm(const float * A, const float * B,
const int m, const int n, const int k, const float alpha, const float beta,
const bool TranA, const bool TranB, float * C);

void cpu_gemv(const float * A, const float * B, const int m, const int n,
const float alpha, const float beta, const bool TranA, float * C);
// should be very careful:
// m is the length of B, and n is the length of C , A is a n*m matrix

void cpu_axpy(const float * A, const int n, const float alpha, float * B);

float cpu_dot(const float * A, const float * B, const int n);

// element-wise
template<typename Op>
void cpu_e_f(const int n, const float alpha, float * A) {
                for (int i = 0 ; i < n ; i++) {
                                Op::Map(alpha, &A[i]);
                }
}

template<typename Op>
void cpu_e_f(const int n, const float * A, const float alpha, float * B) {
                for (int i = 0 ; i < n ; i++) {
                                Op::Map(alpha, A[i], &B[i]);
                }
}

template<typename Op>
void cpu_e_f(const int n, const float * A, const float * B,
const float alpha, const float beta, float * C) {
                for (int i = 0 ; i < n ; i++) {
                                Op::Map(alpha, beta, A[i], B[i], &C[i]);
                }
}
// element-wise generalized operation defined in Op


// matrix/vector expand/reduce

template<typename Op>
void cpu_reduce_f(const float * A, const int m, const int n, float * B) {
                for (int i = 0 ; i < m ; i++) {
                                Op::Map(A+i*n, n, B[i]);
                }
}
// reduce each row of A to an element of B e.g. the sum operation in softmax
template<typename Op>
void cpu_expand_f(const float * A, const int m, const int n, float * B) {
                for (int i = 0 ; i < m ; i++) {
                                Op::Map(A[i], n, B+i*n);
                }
}
// expand each element in A into a row of B

#ifdef SINGA_GPU
void gpu_gemm(const float * A, const float * B,
const int m, const int n, const int k, const float alpha, const float beta,
const bool TranA, const bool TranB, float * C);

void gpu_gemv(const float * A, const float * B, const int m, const int n,
const float alpha, const float beta, const bool TranA, float * C);

void gpu_axpy(const float * A, const int n, const float alpha, float * B);

float gpu_dot(const float * A, const float * B, const int n);

// element-wise
template<typename Op>
void gpu_e_f(const int n, const float alpha, float * A) {
    Op::CudaMap(alpha, A, n);
}

template<typename Op>
void gpu_e_f(const int n, const float * A, const float alpha, float * B) {
    Op::CudaMap(alpha, A, B, n);
}

template<typename Op>
void gpu_e_f(const int n, const float * A, const float * B,
const float alpha, const float beta, float * C) {
    Op::CudaMap(alpha, beta, A, B, C, n);
}
// element-wise generalized operation defined in Op

// matrix/vector expand/reduce

template<typename Op>
void gpu_reduce_f(const float * A, const int m, const int n, float * B) {
                for (int i = 0 ; i < m ; i++) {
                                Op::CudaMap(A+i*n, n, B[i]);
                }
}
// reduce each row of A to an element of B e.g. the sum operation in softmax
template<typename Op>
void gpu_expand_f(const float * A, const int m, const int n, float * B) {
                for (int i = 0 ; i < m ; i++) {
                                Op::CudaMap(A[i], n, B+i*n);
                }
}
// expand each element in A into a row of B
#endif  // SINGA_GPU  

}  // namespace singa
#endif  // SINGA_BLOB_MATH_ADDR_H_
