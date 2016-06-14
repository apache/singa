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

#ifndef  SINGA_CORE_TENSOR_TENSOR_MATH_OPENCL_H_
#include "./tensor_math.h"

// Kernel names
constexpr const char* clkernel_abs = "clkernel_abs";

namespace singa {

// ============================================================================
// ========================== Reduction functions =============================
// ============================================================================

template<>
void Abs<float, lib::Opencl>(int count, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_abs";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Set<float, lib::Opencl>(int count, float x, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_set";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Sum<float, lib::Opencl>(int count, const Blob* input, float* ret, Context* ctx) {
  std::string kname = "clkernel_reduce";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, ret);
  kernel.setArg(3, cl::Local(sizeof(float) * 512));
  
  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Sign<float, lib::Opencl>(int count, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_sign";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Exp<float, lib::Opencl>(int count, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_exp";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Log<float, lib::Opencl>(int count, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_log";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

// ============================================================================
// ======================== Element-wise functions ============================
// ============================================================================

template<>
void Sqrt<float, lib::Opencl>(int count, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_sqrt";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Square<float, lib::Opencl>(int count, const Blob* input, Blob* ret, Context* ctx) {
  Pow<float, lib::Opencl>(count, 2, input, ret, ctx);
}

template<>
void Tanh<float, lib::Opencl>(int count, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_tanh";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void ReLU<float, lib::Opencl>(int count, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_relu";
  auto kernel - oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Sigmoid<float, lib::Opencl>(int count, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_sigmoid";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Softmax<float, lib::Opencl>(int nrow, int ncol, const Blob *input, Blob *ret, Context *ctx) {
  std::string kname = "clkernel_softmax";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, nrow);
  kernel.setArg(1, ncol);
  kernel.setArg(2, static_cast<const float*>(input->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

template<>
void SumRows<float, lib::Opencl>(int nrow, int ncol, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_sumrow";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, nrow);
  kernel.setArg(1, ncol);
  kernel.setArg(2, static_cast<const float*>(input->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

template<>
void SumColumns<float, lib::Opencl>(int nrow, int ncol, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_sumcol";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, nrow);
  kernel.setArg(1, ncol);
  kernel.setArg(2, static_cast<const float*>(input->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

template<>
void AddRow<float, lib::Opencl>(int nrow, int ncol, const Blob* A, const Blob* v, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_addrow";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, nrow);
  kernel.setArg(1, ncol);
  kernel.setArg(2, static_cast<const float*>(A->data()));
  kernel.setArg(3, static_cast<const float*>(v->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

template<>
void AddCol<float, lib::Opencl>(int nrow, int ncol, const Blob* A, const Blob* v, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_addcol";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, nrow);
  kernel.setArg(1, ncol);
  kernel.setArg(2, static_cast<const float*>(A->data()));
  kernel.setArg(3, static_cast<const float*>(v->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

template<>
void Pow<float, lib::Opencl>(int count, float exp, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_pow_scalar";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, exp);
  kernel.setArg(2, static_cast<const float*>(input->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Pow<float, lib::Opencl>(int count, const Blob* base, const Blob* exp, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_pow";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(base->data()));
  kernel.setArg(2, static_cast<const float*>(exp->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Clamp<float, lib::Opencl>(int count, float low, float high, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_clamp";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, low);
  kernel.setArg(2, high);
  kernel.setArg(3, static_cast<const float*>(input->data()));
  kernel.setArg(4, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Add<float, lib::Opencl>(int count, float x, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_add_scalar";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<const float*>(input->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Mult<float, lib::Opencl>(int count, float x, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_mult_scalar";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<const float*>(input->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Add<float, lib::Opencl>(int count, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_add";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(lhs->data()));
  kernel.setArg(2, static_cast<const float*>(rhs->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Mult<float, lib::Opencl>(int count, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_mult";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(lhs->data()));
  kernel.setArg(2, static_cast<const float*>(rhs->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Sub<float, lib::Opencl>(int count, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_subtract";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(lhs->data()));
  kernel.setArg(2, static_cast<const float*>(rhs->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Div<float, lib::Opencl>(int count, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_divide";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(lhs->data()));
  kernel.setArg(2, static_cast<const float*>(rhs->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Outer<float, lib::Opencl>(int m, int n, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_outerproduct";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, m);
  kernel.setArg(1, n);
  kernel.setArg(2, static_cast<const float*>(lhs->data()));
  kernel.setArg(3, static_cast<const float*>(rhs->data()));
  kernel.setArg(4, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(m, n));
}

template<>
void LE<float, lib::Opencl>(const size_t num, const Blob *in, const float x, Blob *out, Context *ctx) {
  std::string kname = "clkernel_le";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, x);
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void GT<float, lib::Opencl>(const size_t num, const Blob *in, const float x, Blob *out, Context *ctx) {
  std::string kname = "clkernel_gt";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, x);
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void GE<float, lib::Opencl>(const size_t num, const Blob *in, const float x, Blob *out, Context *ctx) {
  std::string kname = "clkernel_ge";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, x);
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

// ============================================================================
// ============================= BLAS Level 1 =================================
// ============================================================================

template<>
void Amax<float, lib::Opencl>(int count, const Blob* input, int* ret, Context* ctx) {
  std::string kname = "clkernel_amax";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, ret);
  kernel.setArg(3, cl::Local(sizeof(float) * count));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Amin<float, lib::Opencl>(int count, const Blob* input, int* ret, Context* ctx) {
  std::string kname = "clkernel_amin";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, ret);
  kernel.setArg(3, cl::Local(sizeof(float) * count));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Asum<float, lib::Opencl>(int count, const Blob* input, float* ret, Context* ctx) {
  std::string kname = "clkernel_asum";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, static_cast<const float*>(input->data()));
  kernel.setArg(2, ret);
  kernel.setArg(3, cl::Local(sizeof(float) * count));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Axpy<float, lib::Opencl>(int count, float alpha, const Blob* input, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_axpy";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, alpha);
  kernel.setArg(2, static_cast<const float*>(input->data()));
  kernel.setArg(3, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Scale<float, lib::Opencl>(int count, float x, Blob* ret, Context* ctx) {
  std::string kname = "clkernel_scale";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, count);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<float*>(ret->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(count));
}

template<>
void Dot<float, lib::Opencl>(const size_t num, const Blob *in1, const Blob *in2, DType *out, Context *ctx) {
  std::string kname = "clkernel_dot";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in1->data()));
  kernel.setArg(2, static_cast<const float*>(in2->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));
  kernel.setArg(4, cl::Local(sizeof(float) * num))

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

// ============================================================================
// ============================= BLAS Level 3 =================================
// ============================================================================

template<>
void DGMM<float, lib::Opencl>(const bool side_right,
		  const size_t nrow, const size_t ncol,
		  const Blob *M, const Blob *v, Blob *out, Context *ctx) {
  std::string kname = side_right ? "clkernel_dgmm_right" : "clkernel_dgmm_left";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)nrow);
  kernel.setArg(1, (int)ncol);
  kernel.setArg(2, static_cast<const float*>(M->data()));
  kernel.setArg(3, static_cast<const float*>(v->data()));
  kernel.setArg(4, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

template<>
void GEMM<float, lib::Opencl>(const bool transA, const bool transB,
		  const size_t nrowA, const size_t ncolB, const size_t ncolA,
		  const float alpha, const Blob *A, const Blob *B, const float beta,
		  Blob *C, Context *ctx) {
  std::string kname = "clkernel_gemm";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)nrowA);
  kernel.setArg(1, (int)ncolB);
  kernel.setArg(2, (int)ncolA);
  kernel.setArg(3, alpha);
  kernel.setArg(4, static_cast<const float*>(A->data()));
  kernel.setArg(5, static_cast<const float*>(B->data()));
  kernel.setArg(6, beta);
  kernel.setArg(7, static_cast<float*>(C->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrowA, ncolA));
}

template<>
void LT<float, lib::Opencl>(const size_t num, const Blob *in, const float x, Blob *out, Context *ctx) {
  std::string kname = "clkernel_lt";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, x);
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

} //  namespace singa

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_OPENCL_H_
