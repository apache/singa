/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SINGA_CORE_MATH_H_
#define SINGA_CORE_MATH_H_
#include <type_traits>
#include "singa/core/common.h"
#include "singa/utils/logging.h"

namespace singa {

/// \file math.h Math functions for linear algebra, neural net and random
/// operations.
/// All functions have a template argument, DType for DataType, Lang for the
/// device programming language, e.g., Langice::kCpp, Langice::kCuda
///
/// TODO(wangwei) Clean the functions to make the function APIs consistent:
/// 1. All function names should be like XxxYyy or XY, i.e., capitablize the
/// first
///    letter.
/// 2. Order functions based on function name in alphabetical order.
/// 3. Function arguments order is [const basic type] [const Blob] [mutable
/// Blob].
/// 4. Function argument names, use 'num' for total number of elements in
///    elementwise operations; use 'in1' 'in2' for in blobs; use 'out' for
///    output blob or value. With exceptions for some functions, e.g.,
///      Scale(const float alpha, const Blob* in, Blob* out);
///    For such cases, use x, v, alpha, etc for scalar types.
///    For blas functions, follow the blas style for argument names.
///    Use 'M' and 'v' for matrix and vector tensors in functions involving both
///    matrix and vectors.
/// 5. For Blob argument xxx, name its raw pointer as xxxPtr.
/// 6. Pass the 'cudaStream_t s' to every function in math_kernel.h
/// 7. Use size_t for the number of elements, rows or columns.
/// 8. Use the same name for the Tensor and Blob level math functions.

// =============Element-wise operations====================================
/// out[i] = |in[i]|
template <typename DType, typename Lang>
void Abs(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  LOG(FATAL) << "Abs Not Implemented";
}

/// out = in + x
template <typename DType, typename Lang>
void Add(const size_t num, const Blob *in, const DType x, Blob *out,
         Context *ctx) {
  LOG(FATAL) << "Add Not Implemented";
}

/// out = in1 + in2
template <typename DType, typename Lang>
void Add(const size_t num, const Blob *in1, const Blob *in2, Blob *out,
         Context *ctx) {
  LOG(FATAL) << "Add-Pair Not Implemented";
}
/// Element-wise operation, clamp every element into [low, high]
/// if x>high, then x=high; if x<low, then x=low.
template <typename DType, typename Lang>
void Clamp(const size_t num, const DType low, const DType high, const Blob *in,
           Blob *out, Context *ctx) {
  LOG(FATAL) << "Clamp Not Implemented";
}

/// out = x / in
template <typename DType, typename Lang>
void Div(const size_t num, const DType x, const Blob *in, Blob *out,
         Context *ctx) {
  LOG(FATAL) << "Div Not Implemented";
}

template <typename DType, typename Lang>
void Div(const size_t num, const Blob *in, const DType x, Blob *out,
         Context *ctx) {
  CHECK_NE(x, 0.f);
  EltwiseMult<DType, Lang>(num, in, DType(1) / x, out, ctx);
}

/// out = in1 / in2
template <typename DType, typename Lang>
void Div(const size_t num, const Blob *in1, const Blob *in2, Blob *out,
         Context *ctx) {
  LOG(FATAL) << "Div-Pair Not Implemented";
}

/// out = in * x
template <typename DType, typename Lang>
void EltwiseMult(const size_t num, const Blob *in, const DType x, Blob *out,
                 Context *ctx) {
  LOG(FATAL) << "EltwiseMult Not Implemented";
}

/// out = in2 * in2
template <typename DType, typename Lang>
void EltwiseMult(const size_t num, const Blob *in1, const Blob *in2, Blob *out,
                 Context *ctx) {
  LOG(FATAL) << "EltwiseMult-Pair Not Implemented";
}

/// Base is e, Neper number. out[i]=exp(in[i])
template <typename DType, typename Lang>
void Exp(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  LOG(FATAL) << "Exp Not Implemented";
}

/// out[i]=(in[i]<=x)?1.f:0.f
template <typename DType, typename Lang>
void LE(const size_t num, const Blob *in, const DType x, Blob *out,
        Context *ctx) {
  LOG(FATAL) << "LE Not Implemented";
}
/// Natual logarithm, the base is e, Neper number out[i]=log(in[i]).
template <typename DType, typename Lang>
void Log(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  LOG(FATAL) << "Log Not Implemented";
}
/// out[i]=(in[i]<x)?1.f:0.f
template <typename DType, typename Lang>
void LT(const size_t num, const Blob *in, const DType x, Blob *out,
        Context *ctx) {
  LOG(FATAL) << "LT Not Implemented";
}
/// out[i]=(in[i]>=x)?1.f:0.f
template <typename DType, typename Lang>
void GE(const size_t num, const Blob *in, const DType x, Blob *out,
        Context *ctx) {
  LOG(FATAL) << "GE Not Implemented";
}
/// out[i]=(in[i]>x)?1.f:0.f
template <typename DType, typename Lang>
void GT(const size_t num, const Blob *in, const DType x, Blob *out,
        Context *ctx) {
  LOG(FATAL) << "GT Not Implemented";
}
/// Element-wise operation, do v^x for every v from the in tensor
template <typename DType, typename Lang>
void Pow(const size_t num, const Blob *in, const DType x, Blob *out,
         Context *ctx) {
  LOG(FATAL) << "Pow Not Implemented";
}

/// Element-wise operation, do v^x for every v from the lhs and every x from rhs
template <typename DType, typename Lang>
void Pow(const size_t num, const Blob *in1, const Blob *in2, Blob *out,
         Context *ctx) {
  LOG(FATAL) << "Pow-Pair Not Implemented";
}

/// Element-wise operation, out[i]=max(0, in[i])
template <typename DType, typename Lang>
void ReLU(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  LOG(FATAL) << "ReLU Not Implemented";
}

template <typename DType, typename Lang>
void Set(const size_t num, const DType x, Blob *out, Context *ctx) {
  LOG(FATAL) << "Set Not Implemented";
}
/// Element-wise operation, out[i]=sigmoid([in[i])
template <typename DType, typename Lang>
void Sigmoid(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  LOG(FATAL) << "Sigmoid Not Implemented";
}

/// out[i] = sign(in[i])
template <typename DType, typename Lang>
void Sign(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  LOG(FATAL) << "Sign Not Implemented";
}
/// Element-wise operation, out[i]=sqrt([in[i])
template <typename DType, typename Lang>
void Sqrt(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  LOG(FATAL) << "Sqrt Not Implemented";
}

/// Element-wise operation, out[i]=square([in[i])
template <typename DType, typename Lang>
void Square(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  LOG(FATAL) << "Square Not Implemented";
}

/// out =  in - x
template <typename DType, typename Lang>
void Sub(const size_t num, const Blob *in, const DType x, Blob *out,
         Context *ctx) {
  Add<DType, Lang>(num, in, -x, out, ctx);
}

/// out = in1 - in2
template <typename DType, typename Lang>
void Sub(const size_t num, const Blob *in1, const Blob *in2, Blob *out,
         Context *ctx) {
  LOG(FATAL) << "Sub-Pair Not Implemented";
}
/// sum all elements of in into out
template <typename DType, typename Lang>
void Sum(const size_t num, const Blob *in, DType *out, Context *ctx) {
  LOG(FATAL) << "Sum Not Implemented";
}

/// Element-wise operation, out[i]=tanh([in[i])
template <typename DType, typename Lang>
void Tanh(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  LOG(FATAL) << "Tanh Not Implemented";
}

// =========== Matrix operations ===========================================
/// Add the vector v to every column of A as the column of out
template <typename DType, typename Lang>
void AddCol(const size_t nrow, const size_t ncol, const Blob *A, const Blob *v,
            Blob *out, Context *ctx) {
  LOG(FATAL) << "AddCol Not Implemented";
}
// TODO(wangwei) unify AddRow and AddCol.
/// Add the vector v to every row of A as the row of out
template <typename DType, typename Lang>
void AddRow(const size_t nrow, const size_t ncol, const Blob *A, const Blob *v,
            Blob *out, Context *ctx) {
  LOG(FATAL) << "AddRow Not Implemented";
}
/// outer-product.
/// in1 and in2 are vectors of len m and n. out is matrix of shape m * n
template <typename DType, typename Lang>
void Outer(const size_t m, const size_t n, const Blob *in1, const Blob *in2,
           Blob *out, Context *ctx) {
  LOG(FATAL) << "Outer Not Implemented";
}
// Do softmax for each row invidually
template <typename DType, typename Lang>
void Softmax(const size_t nrow, const size_t ncol, const Blob *in, Blob *out,
             Context *ctx) {
  LOG(FATAL) << "Softmax Not Implemented";
}
/// Sum the columns of the in matrix into a vector
template <typename DType, typename Lang>
void SumColumns(const size_t nrow, const size_t ncol, const Blob *in, Blob *out,
                Context *ctx) {
  LOG(FATAL) << "SumColumns Not Implemented";
}
// TODO(wangwei) unify SumRow and SumCol.
/// Sum the rows of the in matrix into a vector
template <typename DType, typename Lang>
void SumRows(const size_t nrow, const size_t ncol, const Blob *in, Blob *out,
             Context *ctx) {
  LOG(FATAL) << "SumRows Not Implemented";
}

// ================Random functions===========================================
/// Each element of out would be 1 with prob p and 0 with 1-p. 0<= p <= 1
// Get the random generator from 'ctx'
// If DType is not float, then convert the threshold to DType
template <typename DType, typename Lang>
void Bernoulli(const size_t num, const float p, Blob *out, Context *ctx) {
  LOG(FATAL) << "Bernoulli Not Implemented";
}
// The random generator should be extracted from ctx.
// If DType is not float, then convert the mean and std to DType
template <typename DType, typename Lang>
void Gaussian(const size_t num, const float mean, const float std, Blob *out,
              Context *ctx) {
  LOG(FATAL) << "Gaussian Not Implemented";
}
// The random generator should be extracted from ctx.
// If DType is not float, then convert the low and high to DType
template <typename DType, typename Lang>
void Uniform(const size_t num, const float low, const float high, Blob *out,
             Context *ctx) {
  LOG(FATAL) << "Uniform Not Implemented";
}

// ===== BLAS functions, ref to http://docs.nvidia.com/cuda/cublas
/// outurn the index of the element with the max value.
template <typename DType, typename Lang>
void Amax(const size_t num, const Blob *in, size_t *out, Context *ctx) {
  LOG(FATAL) << "Amax Not Implemented";
}

/// outurn the index of the element with the min value.
template <typename DType, typename Lang>
void Amin(const size_t num, const Blob *in, size_t *out, Context *ctx) {
  LOG(FATAL) << "Amin Not Implemented";
}
/// out = sum |x| for all x in in
template <typename DType, typename Lang>
void Asum(const size_t num, const Blob *in, DType *out, Context *ctx) {
  LOG(FATAL) << "Asum Not Implemented";
}

/// out = alpha * in + out
template <typename DType, typename Lang>
void Axpy(const size_t num, const DType alpha, const Blob *in, Blob *out,
          Context *ctx) {
  LOG(FATAL) << "Axpy Not Implemented";
}

/// out *= x
template <typename DType, typename Lang>
void Scale(const size_t num, const DType x, Blob *out, Context *ctx) {
  LOG(FATAL) << "Scale Not Implemented";
}

template <typename DType, typename Lang>
void Dot(const size_t num, const Blob *in1, const Blob *in2, DType *out,
         Context *ctx) {
  LOG(FATAL) << "Dot Not Implemented";
}

/// out = alpha * A * v + beta * out.
/// transA indicates if the internal data layout is transposed of A
template <typename DType, typename Lang>
void GEMV(bool trans, const size_t m, const size_t n, const DType alpha,
          const Blob *A, const Blob *v, const DType beta, Blob *out,
          Context *ctx) {
  LOG(FATAL) << "GEMV Not Implemented";
}

/// multiply a matrix with a diagnoal matrix constructed using values from 'v'.
/// if matrix_lef_side is true, do M*v; else do v*M
template <typename DType, typename Lang>
void DGMM(const bool side_right, const size_t nrow, const size_t ncol,
          const Blob *M, const Blob *v, Blob *out, Context *ctx) {
  LOG(FATAL) << "DGMM Not Implemented";
}

/// C = alpha * A * B + beta * C.
/// transA indicates if the internal data layout is transposed of A
template <typename DType, typename Lang>
void GEMM(const bool transA, const bool transB, const size_t nrowA,
          const size_t ncolB, const size_t ncolA, const DType alpha,
          const Blob *A, const Blob *B, const DType beta, Blob *C,
          Context *ctx) {
  LOG(FATAL) << "GEMM Not Implemented";
}

}  // namespace singa
#endif  // SINGA_CORE_MATH_H_
