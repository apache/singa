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
/// 1. All function names should be like XxxYyy or XY, i.e., capitablize the first
///    letter.
/// 2. Order functions based on function name in alphabetical order.
/// 3. Function arguments order is [const basic type] [const Blob] [mutable Blob].
/// 4. Function argument names, use 'num' for total number of elements in
///    elementwise operations; use 'in1' 'in2' for input blobs; use 'out' for
///    output blob or value. With exceptions for some functions, e.g.,
///      Scale(const float alpha, const Blob* in, Blob* out);
///    For such cases, use x, v, alpha, etc for scalar types.
///    For blas functions, follow the blas style for argument names.


// ================Linear algebra functions====================================
/// ret[i] = |input[i]|
template <typename DType, typename Lang>
void Abs(int count, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

template <typename DType, typename Lang>
void Set(int count, DType x, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}
/// sum all elements of input into ret
template <typename DType, typename Lang>
void Sum(int count, const Blob* input, DType* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// ret[i] = sign(input[i])
template <typename DType, typename Lang>
void Sign(int count, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Base is e, Neper number. ret[i]=exp(input[i])
template <typename DType, typename Lang>
void Exp(int count, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Natual logarithm, the base is e, Neper number ret[i]=log(input[i]).
template <typename DType, typename Lang>
void Log(int count, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Element-wise operation, ret[i]=sqrt([input[i])
template <typename DType, typename Lang>
void Sqrt(int count, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Element-wise operation, ret[i]=square([input[i])
template <typename DType, typename Lang>
void Square(int count, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Element-wise operation, ret[i]=tanh([input[i])
template <typename DType, typename Lang>
void Tanh(int count, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}
/// Element-wise operation, ret[i]=max(0, input[i])
template <typename DType, typename Lang>
void ReLU(int count, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}
/// Element-wise operation, ret[i]=sigmoid([input[i])
template <typename DType, typename Lang>
void Sigmoid(int count, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Do softmax for each row invidually
template <typename DType, typename Lang>
void Softmax(int nrow, int ncol, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

// TODO(wangwei) unify SumRow and SumCol.
/// Sum the rows of the input matrix into a vector
template <typename DType, typename Lang>
void SumRows(int nrow, int ncol, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Sum the columns of the input matrix into a vector
template <typename DType, typename Lang>
void SumColumns(int nrow, int ncol, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

// TODO(wangwei) unify AddRow and AddCol.
/// Add the vector v to every row of A as the row of ret
template <typename DType, typename Lang>
void AddRow(int nrow, int ncol, const Blob* A, const Blob* v, Blob* ret,
            Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Add the vector v to every column of A as the column of ret
template <typename DType, typename Lang>
void AddCol(int nrow, int ncol, const Blob* A, const Blob* v, Blob* ret,
            Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}


/// Element-wise operation, do v^x for every v from the input tensor
template <typename DType, typename Lang>
void Pow(int count, const Blob* input, DType x, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Element-wise operation, do v^x for every v from the lhs and every x from rhs
template <typename DType, typename Lang>
void Pow(int count, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// Element-wise operation, clamp every element into [low, high]
/// if x>high, then x=high; if x<low, then x=low.
template <typename DType, typename Lang>
void Clamp(int count, DType low, DType high, const Blob* input, Blob* ret,
           Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// ret = input + x
template <typename DType, typename Lang>
void Add(int count, const Blob* input, DType x, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}
/// ret =  input - x
template <typename DType, typename Lang>
void Sub(int count, const Blob* input, DType x, Blob* ret, Context* ctx) {
  Add<DType, Lang>(count, input, -x, ret, ctx);
}
/// ret = input * x
template <typename DType, typename Lang>
void EltwiseMult(int count, const Blob* input, DType x, Blob* ret, Context* ctx)
{
  LOG(FATAL) << "Not Implemented";
}
/// ret = input / x
template <typename DType, typename Lang>
void Div(int count, const Blob* input, DType x, Blob* ret, Context* ctx) {
  EltwiseMult<DType, Lang>(count, input, DType(1) / x, ret, ctx);
}

/// ret = lhs + rhs
template <typename DType, typename Lang>
void Add(int count, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// ret = lhs - rhs
template <typename DType, typename Lang>
void Sub(int count, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// ret = lhs * rhs
template <typename DType, typename Lang>
void EltwiseMult(int count, const Blob* lhs, const Blob* rhs, Blob* ret,
          Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// ret = lhs / rhs
template <typename DType, typename Lang>
void Div(int count, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// outer-product.
/// lhs and rhs are vectors of len m and n. ret is matrix of shape m * n
template <typename DType, typename Lang>
void Outer(int m, int n, const Blob* lhs, const Blob* rhs, Blob* ret,
           Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

// ===== BLAS functions, ref to http://docs.nvidia.com/cuda/cublas
// ===== Level 1
/// return the index of the element with the max value.
template <typename DType, typename Lang>
void Amax(int count, const Blob* input, int* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// return the index of the element with the min value.
template <typename DType, typename Lang>
void Amin(int count, const Blob* input, int* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}
/// ret = sum |x| for all x in input
template <typename DType, typename Lang>
void Asum(int count, const Blob* input, DType* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// ret = alpha * input + ret
template <typename DType, typename Lang>
void Axpy(int count, DType alpha, const Blob* input, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/// ret *= x
template <typename DType, typename Lang>
void Scale(int count, DType x, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

template <typename DType, typename Lang>
void Dot(int count, const Blob* lhs, const Blob* rhs, DType* ret,
         Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

// ===== Level 2
/// ret = alpha * op(A) * v + beta * ret.
/// op(A) = A if trans = false; A^T otherwise; rows(op(A)) = m, cols(op(A)) = n.
template <typename DType, typename Lang>
void GEMV(bool trans, int m, int n, DType alpha, const Blob* A, const Blob* v,
          DType beta, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

// ===== Level 3
/// ret = alpha * op(A) * op(B) + beta * ret.
/// op(A) = A if trans = false; A^T otherwise; rows(ret) = m, cols(ret) = n.
template <typename DType, typename Lang>
void GEMM(bool transA, bool transB, int m, int n, int k, DType alpha,
          const Blob* A, const Blob* B, DType beta, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

// ================Random functions===========================================
/// Each element of ret would be 1 with prob p and 0 with 1-p. 0<= p <= 1
// Get the random generator from 'ctx'
// If DType is not float, then convert the threshold to DType
template <typename DType, typename Lang>
void Bernoulli(int count, float p, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}
// The random generator should be extracted from ctx.
// If DType is not float, then convert the low and high to DType
template <typename DType, typename Lang>
void Uniform(int count, float low, float high, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}
// The random generator should be extracted from ctx.
// If DType is not float, then convert the mean and std to DType
template <typename DType, typename Lang>
void Gaussian(int count, float mean, float std, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

/*Some operations would have many config/hyper-parameters, e.g., Conv, and
these config vary among diff implementations, e.g., cuda/cudnn/opencl.
To separate the modules, we pass a OpConf pointer to the Tensor Op function.
The specific fields are implemented by inheriting OpConf, and casting the
pointer between the base and the sub-class.
class OpConf {
 public:
  template <typename T>
  T* CastTo() {
    static_assert(std::is_base_of<OpConf, T>::value,
                  "The cast type must be a sub-class of OpConf");
    return static_cast<T*>(this);
  }
};
*/
}  // namespace singa

#endif  // SINGA_CORE_MATH_H_
