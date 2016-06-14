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

// **************************************
// Element-wise functions
// **************************************

template<>
void Abs<float, lib::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_abs";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Add<float, lib::Opencl>(const size_t num, const Block* in, float x, Block* out, Context* ctx) {
  std::string kname = "clkernel_add_scalar";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<const float*>(in->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Add<float, lib::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  std::string kname = "clkernel_add";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in1->data()));
  kernel.setArg(2, static_cast<const float*>(in2->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Clamp<float, lib::Opencl>(const size_t num, float low, float high, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_clamp";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, low);
  kernel.setArg(2, high);
  kernel.setArg(3, static_cast<const float*>(in->data()));
  kernel.setArg(4, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Div<float, lib::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  std::string kname = "clkernel_divide_scalar";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<const float*>(in->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Div<float, lib::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  std::string kname = "clkernel_divide";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in1->data()));
  kernel.setArg(2, static_cast<const float*>(in2->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void EltwiseMult<float, lib::Opencl>(const size_t num, const Block* in, float x, Block* out, Context* ctx) {
  std::string kname = "clkernel_eltmult_scalar";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<const float*>(in->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void EltwiseMult<float, lib::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  std::string kname = "clkernel_eltmult";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in1->data()));
  kernel.setArg(2, static_cast<const float*>(in2->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Exp<float, lib::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_exp";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void LE<float, lib::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  std::string kname = "clkernel_le";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, x);
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Log<float, lib::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_log";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void LT<float, lib::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  std::string kname = "clkernel_lt";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, x);
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void GE<float, lib::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  std::string kname = "clkernel_ge";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, x);
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void GT<float, lib::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  std::string kname = "clkernel_gt";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, x);
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Pow<float, lib::Opencl>(const size_t num, const Block* in, float x, Block* out, Context* ctx) {
  std::string kname = "clkernel_pow_scalar";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<const float*>(in->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Pow<float, lib::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  std::string kname = "clkernel_pow";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in1->data()));
  kernel.setArg(2, static_cast<const float*>(in2->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void ReLU<float, lib::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_relu";
  auto kernel - oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Set<float, lib::Opencl>(const size_t num, float x, Block* out, Context* ctx) {
  std::string kname = "clkernel_set";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Sigmoid<float, lib::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_sigmoid";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Sign<float, lib::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_sign";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Sqrt<float, lib::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_sqrt";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Square<float, lib::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  Pow<float, lib::Opencl>(num, 2, in, out, ctx);
}

template<>
void Sub<float, lib::Opencl>(const size_t num, const Block* in, const float rhs, Block* out, Context* ctx) {
  std::string kname = "clkernel_subtract_scalar";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, x);
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Sub<float, lib::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  std::string kname = "clkernel_subtract";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in1->data()));
  kernel.setArg(2, static_cast<const float*>(in2->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Sum<float, lib::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  std::string kname = "clkernel_reduce";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, out);
  kernel.setArg(3, cl::Local(sizeof(float) * 512));
  
  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Tanh<float, lib::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_tanh";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

// **************************************
// Random functions
// **************************************

// TODO: Bernoulli

// TODO: Gaussian

// TODO: Uniform

// *********************************************************
// BLAS functions, ref to http://docs.nvidia.com/cuda/cublas
// *********************************************************

template<>
void Amax<float, lib::Opencl>(const size_t num, const Block* in, int* out, Context* ctx) {
  std::string kname = "clkernel_amax";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, out);
  kernel.setArg(3, cl::Local(sizeof(float) * num));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Amin<float, lib::Opencl>(const size_t num, const Block* in, int* out, Context* ctx) {
  std::string kname = "clkernel_amin";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, out);
  kernel.setArg(3, cl::Local(sizeof(float) * num));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Asum<float, lib::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  std::string kname = "clkernel_asum";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in->data()));
  kernel.setArg(2, out);
  kernel.setArg(3, cl::Local(sizeof(float) * num));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Axpy<float, lib::Opencl>(const size_t num, float alpha, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_axpy";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, alpha);
  kernel.setArg(2, static_cast<const float*>(in->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

// TODO:: Nrm2

template<>
void Scale<float, lib::Opencl>(const size_t num, float x, Block* out, Context* ctx) {
  std::string kname = "clkernel_scale";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

template<>
void Dot<float, lib::Opencl>(const size_t num, const Block *in1, const Block *in2, DType *out, Context *ctx) {
  std::string kname = "clkernel_dot";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)num);
  kernel.setArg(1, static_cast<const float*>(in1->data()));
  kernel.setArg(2, static_cast<const float*>(in2->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));
  kernel.setArg(4, cl::Local(sizeof(float) * num))

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(num));
}

// TODO: GEMV

template<>
void DGMM<float, lib::Opencl>(const bool side_right,
		  const size_t nrow, const size_t ncol,
		  const Block *M, const Block *v, Block *out, Context *ctx) {
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
		  const float alpha, const Block *A, const Block *B, const float beta,
		  Block *C, Context *ctx) {
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

// TODO: ComputeCrossEntropy

// TODO: SoftmaxCrossEntropyBwd

// **************************************
// Matrix functions
// **************************************

template<>
void AddCol<float, lib::Opencl>(const size_t nrow, const size_t ncol, const Block* A, const Block* v, Block* out, Context* ctx) {
  std::string kname = "clkernel_addcol";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)nrow);
  kernel.setArg(1, (int)ncol);
  kernel.setArg(2, static_cast<const float*>(A->data()));
  kernel.setArg(3, static_cast<const float*>(v->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

template<>
void AddRow<float, lib::Opencl>(const size_t nrow, const size_t ncol, const Block* A, const Block* v, Block* out, Context* ctx) {
  std::string kname = "clkernel_addrow";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)nrow);
  kernel.setArg(1, (int)ncol);
  kernel.setArg(2, static_cast<const float*>(A->data()));
  kernel.setArg(3, static_cast<const float*>(v->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

template<>
void Outer<float, lib::Opencl>(const size_t m, const size_t n, const Block* lhs, const Block* rhs, Block* out, Context* ctx) {
  std::string kname = "clkernel_outerproduct";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)m);
  kernel.setArg(1, (int)n);
  kernel.setArg(2, static_cast<const float*>(lhs->data()));
  kernel.setArg(3, static_cast<const float*>(rhs->data()));
  kernel.setArg(4, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(m, n));
}

template<>
void SumColumns<float, lib::Opencl>(const size_t nrow, const size_t ncol, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_sumcol";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)nrow);
  kernel.setArg(1, (int)ncol);
  kernel.setArg(2, static_cast<const float*>(in->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

template<>
void SumRows<float, lib::Opencl>(const size_t nrow, const size_t ncol, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_sumrow";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (int)nrow);
  kernel.setArg(1, (int)ncol);
  kernel.setArg(2, static_cast<const float*>(in->data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  oclDevice.cmdQueue.enqueueNDRangeKernel(kernel, nullptr, NDRange(nrow, ncol));
}

} //  namespace singa

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_OPENCL_H_
