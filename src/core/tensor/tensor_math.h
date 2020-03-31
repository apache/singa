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
#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "singa/core/common.h"
#include "singa/core/tensor.h"
#include "singa/utils/logging.h"

namespace singa {

/// \file math.h Math functions for linear algebra, neural net and random
/// operations.
/// All functions have a template argument, DType for DataType, Lang for the
/// device programming language, e.g., Langice::kCpp, Langice::kCuda
///
/// TODO(wangwei) Clean the functions to make the function APIs consistent:
/// 1. All function names should be like XxxYyy or XY, i.e., capitalize the
/// first letter.
/// 2. Order functions based on function name in alphabetical order.
/// 3. Function arguments order is [const basic type] [const Tensor] [mutable
/// Tensor].
/// 4. Function argument names, use 'num' for total number of elements in
///    elementwise operations; use 'in1' 'in2' for in Tensors; use 'out' for
///    output Tensor or value. With exceptions for some functions, e.g.,
///      Scale(const float alpha, const Tensor &in, Tensor* out);
///    For such cases, use x, v, alpha, etc for scalar types.
///    For blas functions, follow the blas style for argument names.
///    Use 'M' and 'v' for matrix and vector tensors in functions involving both
///    matrix and vectors.
/// 5. For Tensor argument xxx, name its raw pointer as xxxPtr.
/// 6. Pass the 'cudaStream_t s' to every function in math_kernel.h
/// 7. Use size_t for the number of elements, rows or columns.
/// 8. Use the same name for the Tensor and Tensor level math functions.

const std::string vec2str(const std::vector<int> &vec) {
  std::ostringstream vts;
  if (!vec.empty()) {
    // Convert all but the last element to avoid a trailing ","
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(vts, ", "));
  }
  return vts.str();
}

const std::string vec2str(const std::vector<size_t> &vec) {
  std::ostringstream vts;
  if (!vec.empty()) {
    // Convert all but the last element to avoid a trailing ","
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<size_t>(vts, ", "));
  }
  return vts.str();
}

// **************************************
// // Element-wise functions
// // Cpp tensors support multi-dimensional broadcasting;
// // Cuda supports unidirectional broadcasting,
// // i.e., the lhs and the output have the same shape
// // **************************************

/// out[i] = |in[i]|
template <typename DType, typename Lang>
void Abs(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Abs Not Implemented";
}

template <typename DTypeSrc, typename DTypeDst, typename Lang>
void CastCopy(const Tensor *src, Tensor *dst, Context *ctx) {
  LOG(FATAL) << "CastCopy Not Implemented";
}

template <typename DType, typename Lang>
void Ceil(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Ceil Not Implemented";
}

/// out[i] = in[i] + x
template <typename DType, typename Lang>
void Add(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Add Not Implemented";
}

/// out[i] = in1[i] + in2[i]
template <typename DType, typename Lang>
void Add(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Add-Pair Not Implemented";
}
/// Clamp every element into [low, high]
/// if in[i]>high, then out[i]=high; if in[i]<low, then out[i]=low.
template <typename DType, typename Lang>
void Clamp(const DType low, const DType high, const Tensor &in, Tensor *out,
           Context *ctx) {
  LOG(FATAL) << "Clamp Not Implemented";
}

/// out[i] = x / in[i]
template <typename DType, typename Lang>
void Div(const DType x, const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Div Not Implemented";
}

/// out[i] = in[i] / x
template <typename DType, typename Lang>
void Div(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  CHECK_NE(x, 0.f);
  EltwiseMult<DType, Lang>(in, DType(1) / x, out, ctx);
}

/// out[i] = in1[i] / in2[i]
template <typename DType, typename Lang>
void Div(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Div-Pair Not Implemented";
}

/// out[i] = in[i] * x
template <typename DType, typename Lang>
void EltwiseMult(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "EltwiseMult Not Implemented";
}

/// out[i] = in1[i] * in2[i]
template <typename DType, typename Lang>
void EltwiseMult(const Tensor &in1, const Tensor &in2, Tensor *out,
                 Context *ctx) {
  LOG(FATAL) << "EltwiseMult-Pair Not Implemented";
}

/// out[i]=(in2[i]>0)?in1[i]:0.f
template <typename DType, typename Lang>
void ReLUBackward(const Tensor &in1, const Tensor &in2, Tensor *out,
                  Context *ctx) {
  LOG(FATAL) << "ReLUBackward Not Implemented";
}

/// Base is e, Neper number. out[i]=exp(in[i])
template <typename DType, typename Lang>
void Exp(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Exp Not Implemented";
}

/// out[i]=(in[i]<=x)?1.f:0.f
template <typename DType, typename Lang>
void LE(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "LE Not Implemented";
}
/// out[i]=(in1[i]<=in2[i])?1.f:0.f
template <typename DType, typename Lang>
void LE(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Tensor-Tensor LE Not Implemented";
}
/// Natural logarithm, the base is e, Neper number out[i]=log(in[i]).
template <typename DType, typename Lang>
void Log(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Log Not Implemented";
}
/// out[i]=(in[i]<x)?1.f:0.f
template <typename DType, typename Lang>
void LT(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "LT Not Implemented";
}
/// out[i]=(in1[i]<in2[i])?1.f:0.f
template <typename DType, typename Lang>
void LT(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Tensor-Tensor LT Not Implemented";
}
/// out[i]=(in[i]>=x)?1.f:0.f
template <typename DType, typename Lang>
void GE(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "GE Not Implemented";
}
/// out[i]=(in1[i]>=in2[i])?1.f:0.f
template <typename DType, typename Lang>
void GE(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Tensor-Tensor GE Not Implemented";
}
/// out[i]=(in[i]>x)?1.f:0.f
template <typename DType, typename Lang>
void GT(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "GT Not Implemented";
}
/// out[i]=(in[i]>in2[i])?1.f:0.f
template <typename DType, typename Lang>
void GT(const Tensor &in, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Tensor-Tensor GT Not Implemented";
}
/// out[i] = pow(in[i], x)
template <typename DType, typename Lang>
void Pow(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Pow Not Implemented";
}

/// out[i]=pow(in1[i], in2[i])
template <typename DType, typename Lang>
void Pow(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Pow-Pair Not Implemented";
}

/// out[i]=max(0, in[i])
template <typename DType, typename Lang>
void ReLU(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "ReLU Not Implemented";
}

/// out[i] = x
template <typename DType, typename Lang>
void Set(const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Set Not Implemented";
}
/// out[i]=sigmoid(in[i])
template <typename DType, typename Lang>
void Sigmoid(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Sigmoid Not Implemented";
}

/// out[i] = log(exp(in[i]) + 1)
template <typename DType, typename Lang>
void SoftPlus(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "SoftPlus Not Implemented";
}

/// out[i] = in[i] / (abs(in[i]) + 1)
template <typename DType, typename Lang>
void SoftSign(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "SoftSign Not Implemented";
}

/// out[i] = sign(in[i])
template <typename DType, typename Lang>
void Sign(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Sign Not Implemented";
}
/// out[i]=sqrt(in[i])
template <typename DType, typename Lang>
void Sqrt(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Sqrt Not Implemented";
}

/// out[i]=square(in[i])
template <typename DType, typename Lang>
void Square(const Tensor &in, Tensor *out, Context *ctx) {
  EltwiseMult<DType, Lang>(in, in, out, ctx);
}

/// out[i] =  in[i] - x
template <typename DType, typename Lang>
void Sub(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  Add<DType, Lang>(in, -x, out, ctx);
}

/// out[i] = in1[i] - in2[i]
template <typename DType, typename Lang>
void Sub(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Sub-Pair Not Implemented";
}

/// sum all elements of in into out
template <typename DType, typename Lang>
void Sum(const Tensor &in, DType *out, Context *ctx) {
  LOG(FATAL) << "Sum Not Implemented";
}

/// out[i]=fn(in[i])
#define GenUnaryNotImplemented(fn, stringfn)             \
  template <typename DType, typename Lang>               \
  void fn(const Tensor &in, Tensor *out, Context *ctx) { \
    std::string str = stringfn;                          \
    str += " Not Implemented";                           \
    LOG(FATAL) << str;                                   \
  }

GenUnaryNotImplemented(Cos, "Cos");
GenUnaryNotImplemented(Cosh, "Cosh");
GenUnaryNotImplemented(Acos, "Acos");
GenUnaryNotImplemented(Acosh, "Acosh");
GenUnaryNotImplemented(Sin, "Sin");
GenUnaryNotImplemented(Sinh, "Sinh");
GenUnaryNotImplemented(Asin, "Asin");
GenUnaryNotImplemented(Asinh, "Asinh");
GenUnaryNotImplemented(Tan, "Tan");
GenUnaryNotImplemented(Tanh, "Tanh");
GenUnaryNotImplemented(Atan, "Atan");
GenUnaryNotImplemented(Atanh, "Atanh");

/// similar to cudnnTransformTensor
/// copies the data from one tensor to another tensor with a different layout
/// the tensors must have the same dimensions but not necessarily the same
/// strides
template <typename DType, typename Lang>
void Transform(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Transform Not Implemented";
}

// **************************************
// Random functions
// **************************************
/// Each element of out would be 1 with prob p and 0 with 1-p. 0<= p <= 1
// Get the random generator from 'ctx'
// If DType is not float, then convert the threshold to DType
template <typename DType, typename Lang>
void Bernoulli(const float p, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Bernoulli Not Implemented";
}
// The random generator should be extracted from ctx.
// If DType is not float, then convert the mean and std to DType
template <typename DType, typename Lang>
void Gaussian(const float mean, const float std, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Gaussian Not Implemented";
}
// The random generator should be extracted from ctx.
// If DType is not float, then convert the low and high to DType
template <typename DType, typename Lang>
void Uniform(const float low, const float high, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Uniform Not Implemented";
}

// *********************************************************
// BLAS functions, ref to http://docs.nvidia.com/cuda/cublas
// *********************************************************

/// outurn the index of the element with the max value.
template <typename DType, typename Lang>
void Amax(const Tensor &in, size_t *out, Context *ctx) {
  LOG(FATAL) << "Amax Not Implemented";
}

/// outurn the index of the element with the min value.
template <typename DType, typename Lang>
void Amin(const Tensor &in, size_t *out, Context *ctx) {
  LOG(FATAL) << "Amin Not Implemented";
}
/// out = sum |x| for all x in in
template <typename DType, typename Lang>
void Asum(const Tensor &in, DType *out, Context *ctx) {
  LOG(FATAL) << "Asum Not Implemented";
}

/// out = alpha * in + out
template <typename DType, typename Lang>
void Axpy(const DType alpha, const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Axpy Not Implemented";
}

/// out = ||in||_2^2, i.e, L2 norm.
template <typename DType, typename Lang>
void Nrm2(const Tensor &in, float *out, Context *ctx) {
  LOG(FATAL) << "Nrm2 Not Implemented";
}

/// out *= x
template <typename DType, typename Lang>
void Scale(const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Scale Not Implemented";
}

/// inner product of array in1 and in2
template <typename DType, typename Lang>
void Dot(const Tensor &in1, const Tensor &in2, DType *out, Context *ctx) {
  LOG(FATAL) << "Dot Not Implemented";
}
template <typename DType, typename Lang>
void Dot(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Dot Not Implemented";
}

/// out = alpha * A * v + beta * out.
/// transA indicates if the internal data layout is transposed of A
template <typename DType, typename Lang>
void GEMV(const DType alpha, const Tensor &A, const Tensor &v, const DType beta,
          Tensor *out, Context *ctx) {
  LOG(FATAL) << "GEMV Not Implemented";
}

/// multiply a matrix with a diagnoal matrix constructed using values from 'v'.
/// if matrix_lef_side is true, do M*v; else do v*M
template <typename DType, typename Lang>
void DGMM(const bool side_right, const Tensor &M, const Tensor &v, Tensor *out,
          Context *ctx) {
  LOG(FATAL) << "DGMM Not Implemented";
}

/// C = alpha * A * B + beta * C.
/// transA indicates if the internal data layout is transposed of A
template <typename DType, typename Lang>
void GEMM(const DType alpha, const Tensor &A, const Tensor &B, const DType beta,
          Tensor *C, Context *ctx) {
  LOG(FATAL) << "GEMM Not Implemented";
}

template <typename DType, typename Lang>
void GEMMBatched(const DType alpha, const Tensor &A, const Tensor &B,
                 const DType beta, Tensor *C, Context *ctx) {
  LOG(FATAL) << "GEMM Batched Not Implemented";
}

template <typename DType, typename Lang>
void SoftMax(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Not Implemented";
}

template <typename DType, typename Lang>
void SoftMaxBackward(const Tensor &in, Tensor *out, const Tensor &fdout,
                     Context *ctx) {
  LOG(FATAL) << "Not Implemented";
}

// yisen todo
template <typename DType, typename Lang>
void ComputeCrossEntropy(bool int_target, const size_t batchsize,
                         const size_t dim, const Block *p, const Block *t,
                         Block *loss, Context *ctx) {
  LOG(FATAL) << "Not Implemented";
}

template <typename DType, typename Lang>
void SoftmaxCrossEntropyBwd(bool int_target, const size_t batchsize,
                            const size_t dim, const Block *p, const Block *t,
                            Block *grad, Context *ctx) {
  LOG(FATAL) << "Not Implemented";
}

template <typename DType, typename Lang>
void RowMax(const Tensor &in, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Not Implemented";
}
// **************************************
// Matrix functions
// **************************************
/*
/// Add the vector v to every column of A as the column of out
template <typename DType, typename Lang>
void AddCol(const size_t nrow, const size_t ncol, const Tensor &A, const Tensor
&v,
            Tensor *out, Context *ctx) {
  LOG(FATAL) << "AddCol Not Implemented";
}
// TODO(wangwei) unify AddRow and AddCol.
/// Add the vector v to every row of A as the row of out
template <typename DType, typename Lang>
void AddRow(const size_t nrow, const size_t ncol, const Tensor &A, const Tensor
&v,
            Tensor *out, Context *ctx) {
  LOG(FATAL) << "AddRow Not Implemented";
}
/// outer-product.
/// in1 and in2 are vectors of len m and n. out is matrix of shape m * n
template <typename DType, typename Lang>
void Outer(const size_t m, const size_t n, const Tensor &in1, const Tensor &in2,
           Tensor *out, Context *ctx) {
  LOG(FATAL) << "Outer Not Implemented";
}

/// Sum the columns of the in matrix into a vector
template <typename DType, typename Lang>
void SumColumns(const size_t nrow, const size_t ncol, const Tensor &in, Tensor
*out,
                Context *ctx) {
  LOG(FATAL) << "SumColumns Not Implemented";
}
template <typename DType, typename Lang>
void Set(const DType x, Tensor *out, Context *ctx) {
  LOG(FATAL) << "Not Implemented";
}

// TODO(wangwei) unify SumRow and SumCol.
/// Sum the rows of the in matrix into a vector
template <typename DType, typename Lang>
void SumRows(const size_t nrow, const size_t ncol, const Tensor &in, Tensor
*out,
             Context *ctx) {
  LOG(FATAL) << "SumRows Not Implemented";
}
*/

}  // namespace singa
#endif  // SINGA_CORE_MATH_H_
