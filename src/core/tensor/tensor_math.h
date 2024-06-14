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

#define LOG_FATAL(Op, DType, Lang)                                          \
  LOG(FATAL) << Op << " not Implemented for DType=" << typeid(DType).name() \
             << " Lang=" << typeid(Lang).name()

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
  LOG_FATAL("Abs", DType, Lang);
}

template <typename DType, typename Lang>
void Erf(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Erf", DType, Lang);
}

template <typename DTypeSrc, typename DTypeDst, typename Lang>
void CastCopy(const Tensor *src, Tensor *dst, Context *ctx) {
  LOG(FATAL) << "CastCopy not Implemented for DTypeSrc="
             << typeid(DTypeSrc).name()
             << " DTypeDst=" << typeid(DTypeDst).name()
             << " Lang=" << typeid(Lang).name();
}

template <typename DType, typename Lang>
void Ceil(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Ceil", DType, Lang);
}

template <typename DType, typename Lang>
void Floor(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Floor", DType, Lang);
}

template <typename DType, typename Lang>
void Round(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Round", DType, Lang);
}

template <typename DType, typename Lang>
void RoundE(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("RoundE", DType, Lang);
}

/// out[i] = in[i] + x
template <typename DType, typename Lang>
void Add(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("Add", DType, Lang);
}

/// out[i] = in1[i] + in2[i]
template <typename DType, typename Lang>
void Add(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG_FATAL("Add-Pair", DType, Lang);
}
/// Clamp every element into [low, high]
/// if in[i]>high, then out[i]=high; if in[i]<low, then out[i]=low.
template <typename DType, typename Lang>
void Clamp(const DType low, const DType high, const Tensor &in, Tensor *out,
           Context *ctx) {
  LOG_FATAL("Clamp", DType, Lang);
}

/// out[i] = x / in[i]
template <typename DType, typename Lang>
void Div(const DType x, const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Div", DType, Lang);
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
  LOG_FATAL("Div-Pair", DType, Lang);
}

/// out[i] = in[i] * x
template <typename DType, typename Lang>
void EltwiseMult(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("EltwiseMult", DType, Lang);
}

/// out[i] = in1[i] * in2[i]
template <typename DType, typename Lang>
void EltwiseMult(const Tensor &in1, const Tensor &in2, Tensor *out,
                 Context *ctx) {
  LOG_FATAL("EltwiseMult-Pair", DType, Lang);
}

/// out[i]=(in2[i]>0)?in1[i]:0.f
template <typename DType, typename Lang>
void ReLUBackward(const Tensor &in1, const Tensor &in2, Tensor *out,
                  Context *ctx) {
  LOG_FATAL("ReLUBackward", DType, Lang);
}

/// Base is e, Neper number. out[i]=exp(in[i])
template <typename DType, typename Lang>
void Exp(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Exp", DType, Lang);
}

/// out[i]=(in[i]<=x)?1.f:0.f
template <typename DType, typename Lang>
void LE(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("LE", DType, Lang);
}
/// out[i]=(in1[i]<=in2[i])?1.f:0.f
template <typename DType, typename Lang>
void LE(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG_FATAL("Tensor <= Tensor", DType, Lang);
}
/// Natural logarithm, the base is e, Neper number out[i]=log(in[i]).
template <typename DType, typename Lang>
void Log(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Log", DType, Lang);
}
/// out[i]=(in[i]<x)?1.f:0.f
template <typename DType, typename Lang>
void LT(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("LT", DType, Lang);
}
/// out[i]=(in1[i]<in2[i])?1.f:0.f
template <typename DType, typename Lang>
void LT(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG_FATAL("Tensor Tensor LT", DType, Lang);
}
/// out[i]=(in[i]>=x)?1.f:0.f
template <typename DType, typename Lang>
void GE(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("GE", DType, Lang);
}
/// out[i]=(in1[i]>=in2[i])?1.f:0.f
template <typename DType, typename Lang>
void GE(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG_FATAL("Tensor Tensor GE", DType, Lang);
}
/// out[i]=(in[i]>x)?1.f:0.f
template <typename DType, typename Lang>
void GT(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("GT", DType, Lang);
}
/// out[i]=(in[i]>in2[i])?1.f:0.f
template <typename DType, typename Lang>
void GT(const Tensor &in, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG_FATAL("Tensor Tensor GT", DType, Lang);
}
/// out[i]=(in[i]==x)?1.f:0.f
template <typename DType, typename Lang>
void EQ(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("EQ", DType, Lang);
}
/// out[i]=(in[i]==in2[i])?1.f:0.f
template <typename DType, typename Lang>
void EQ(const Tensor &in, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG_FATAL("Tensor Tensor EQ", DType, Lang);
}
/// out[i] = pow(in[i], x)
template <typename DType, typename Lang>
void Pow(const Tensor &in, const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("Pow", DType, Lang);
}

/// out[i]=pow(in1[i], in2[i])
template <typename DType, typename Lang>
void Pow(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG_FATAL("Tensor Tensor Pow", DType, Lang);
}

/// out[i]=max(0, in[i])
template <typename DType, typename Lang>
void ReLU(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("ReLU", DType, Lang);
}

/// out[i] = x
template <typename DType, typename Lang>
void Set(const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("Set", DType, Lang);
}
/// out[i]=sigmoid(in[i])
template <typename DType, typename Lang>
void Sigmoid(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Sigmoid", DType, Lang);
}

/// out[i] = log(exp(in[i]) + 1)
template <typename DType, typename Lang>
void SoftPlus(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("SoftPlus", DType, Lang);
}

/// out[i] = in[i] / (abs(in[i]) + 1)
template <typename DType, typename Lang>
void SoftSign(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("SoftSign", DType, Lang);
}

/// out[i] = sign(in[i])
template <typename DType, typename Lang>
void Sign(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Sign", DType, Lang);
}
/// out[i]=sqrt(in[i])
template <typename DType, typename Lang>
void Sqrt(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Sqrt", DType, Lang);
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
  LOG_FATAL("Tensor Tensor Sub", DType, Lang);
}

/// sum all elements of in into out
template <typename DType, typename Lang>
void Sum(const Tensor &in, DType *out, Context *ctx) {
  LOG_FATAL("Sum", DType, Lang);
}

/// out[i]=fn(in[i])
#define GenUnaryNotImplemented(fn, stringfn)             \
  template <typename DType, typename Lang>               \
  void fn(const Tensor &in, Tensor *out, Context *ctx) { \
    std::string str = stringfn;                          \
    LOG_FATAL(str, DType, Lang);                         \
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
  LOG_FATAL("Transform", DType, Lang);
}

// **************************************
// Random functions
// **************************************
/// Each element of out would be 1 with prob p and 0 with 1-p. 0<= p <= 1
// Get the random generator from 'ctx'
// If DType is not float, then convert the threshold to DType
template <typename DType, typename Lang>
void Bernoulli(const float p, Tensor *out, Context *ctx) {
  LOG_FATAL("Bernoulli", DType, Lang);
}
// The random generator should be extracted from ctx.
// If DType is not float, then convert the mean and std to DType
template <typename DType, typename Lang>
void Gaussian(const DType mean, const DType std, Tensor *out, Context *ctx) {
  LOG_FATAL("Gaussian", DType, Lang);
}
// The random generator should be extracted from ctx.
// If DType is not float, then convert the low and high to DType
template <typename DType, typename Lang>
void Uniform(const DType low, const DType high, Tensor *out, Context *ctx) {
  LOG_FATAL("Uniform", DType, Lang);
}

// *********************************************************
// BLAS functions, ref to http://docs.nvidia.com/cuda/cublas
// *********************************************************

/// outurn the index of the element with the max value.
template <typename DType, typename Lang>
void Amax(const Tensor &in, size_t *out, Context *ctx) {
  LOG_FATAL("Amax", DType, Lang);
}

/// outurn the index of the element with the min value.
template <typename DType, typename Lang>
void Amin(const Tensor &in, size_t *out, Context *ctx) {
  LOG_FATAL("Amin", DType, Lang);
}
/// out = sum |x| for all x in in
template <typename DType, typename Lang>
void Asum(const Tensor &in, DType *out, Context *ctx) {
  LOG_FATAL("Asum", DType, Lang);
}

/// out = alpha * in + out
template <typename DType, typename Lang>
void Axpy(const DType alpha, const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Axpy", DType, Lang);
}

/// out = alpha * in + out
template <typename DType, typename Lang>
void Axpy(const Tensor &alpha, const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("Axpy Tensor alpha", DType, Lang);
}

/// out = ||in||_2^2, i.e, L2 norm.
template <typename DType, typename Lang>
void Nrm2(const Tensor &in, float *out, Context *ctx) {
  LOG_FATAL("Nrm2", DType, Lang);
}

/// out *= x
template <typename DType, typename Lang>
void Scale(const DType x, Tensor *out, Context *ctx) {
  LOG_FATAL("Scale", DType, Lang);
}

/// inner product of array in1 and in2
template <typename DType, typename Lang>
void Dot(const Tensor &in1, const Tensor &in2, DType *out, Context *ctx) {
  LOG_FATAL("Inner-product Dot", DType, Lang);
}
template <typename DType, typename Lang>
void Dot(const Tensor &in1, const Tensor &in2, Tensor *out, Context *ctx) {
  LOG_FATAL("Dot", DType, Lang);
}

/// out = alpha * A * v + beta * out.
/// transA indicates if the internal data layout is transposed of A
template <typename DType, typename Lang>
void GEMV(const DType alpha, const Tensor &A, const Tensor &v, const DType beta,
          Tensor *out, Context *ctx) {
  LOG_FATAL("GEMV", DType, Lang);
}

/// multiply a matrix with a diagnoal matrix constructed using values from 'v'.
/// if matrix_lef_side is true, do M*v; else do v*M
template <typename DType, typename Lang>
void DGMM(const bool side_right, const Tensor &M, const Tensor &v, Tensor *out,
          Context *ctx) {
  LOG_FATAL("DGMM", DType, Lang);
}

/// C = alpha * A * B + beta * C.
/// transA indicates if the internal data layout is transposed of A
template <typename DType, typename Lang>
void GEMM(const DType alpha, const Tensor &A, const Tensor &B, const DType beta,
          Tensor *C, Context *ctx) {
  LOG_FATAL("GEMM", DType, Lang);
}

template <typename DType, typename Lang>
void GEMMBatched(const DType alpha, const Tensor &A, const Tensor &B,
                 const DType beta, Tensor *C, Context *ctx) {
  LOG_FATAL("GEMMBatched", DType, Lang);
}

template <typename DType, typename Lang>
void SoftMax(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("SoftMax", DType, Lang);
}

template <typename DType, typename Lang>
void SoftMaxBackward(const Tensor &in, Tensor *out, const Tensor &fdout,
                     Context *ctx) {
  LOG_FATAL("SoftMaxBackend", DType, Lang);
}

// yisen todo
template <typename DType, typename Lang>
void ComputeCrossEntropy(bool int_target, const size_t batchsize,
                         const size_t dim, const Tensor &p, const Tensor &t,
                         Tensor *loss, Context *ctx) {
  LOG_FATAL("ComputeCrossEntropy", DType, Lang);
}

template <typename DType, typename Lang>
void SoftmaxCrossEntropyBwd(bool int_target, const size_t batchsize,
                            const size_t dim, const Tensor &p, const Tensor &t,
                            Tensor *grad, Context *ctx) {
  LOG_FATAL("ComputeCrossEntropyBwd", DType, Lang);
}

template <typename DType, typename Lang>
void RowMax(const Tensor &in, Tensor *out, Context *ctx) {
  LOG_FATAL("RowMax", DType, Lang);
}

}  // namespace singa
#endif  // SINGA_CORE_MATH_H_
