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

#ifdef USE_OPENCL
#include <limits>

#include "singa/utils/opencl_utils.h"
#include "tensor_math.h"

namespace singa {

// Some forward declarations of utility functions that only exist here.
void Transpose(const size_t nrow, const size_t ncol, cl::Buffer& in, cl::Buffer& out, Context* ctx);
void DiagVec_Left(const size_t size, cl::Buffer& in, cl::Buffer& out, Context* ctx);
void DiagVec_Right(const size_t size, cl::Buffer& in, cl::Buffer& out, Context* ctx);

// **************************************
// Element-wise functions
// **************************************

template<>
void Abs<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_abs";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Add<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_add_scalar";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Add<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_add";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Clamp<float, lang::Opencl>(const size_t num, const float low, const float high, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_clamp";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, low);
  kernel.setArg(2, high);
  kernel.setArg(3, inbuf);
  kernel.setArg(4, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Div<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_divide_scalar_matx";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Div<float, lang::Opencl>(const size_t num, const float x, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_divide_scalar_xmat";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Div<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_divide";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void EltwiseMult<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_eltmult_scalar";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void EltwiseMult<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_eltmult";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Exp<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_exp";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void LE<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_le";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Log<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_log";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void LT<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_lt";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void GE<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_ge";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void GT<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_gt";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Pow<float, lang::Opencl>(const size_t num, const Block* in, float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_pow_scalar";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Pow<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_pow";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void ReLU<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_relu";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}

template<>
void Set<float, lang::Opencl>(const size_t num, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_set";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Sigmoid<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_sigmoid";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Sign<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_sign";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Sqrt<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_sqrt";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Square<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  Pow<float, lang::Opencl>(num, in, 2, out, ctx);
}


template<>
void Sub<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_subtract_scalar";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Sub<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_subtract";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Sum<float, lang::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_reduce";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  
  size_t size = sizeof(float) * num;
  cl::Buffer outval(ctx->ocl_ctx, CL_MEM_WRITE_ONLY, size, nullptr, &status);
  OCL_CHECK(status, "Failed to create buffer!");

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outval);
  kernel.setArg(3, cl::Local(size));
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");

  float* temp = new float[num];
  status = ctx->ocl_cmdq.enqueueReadBuffer(outval, CL_TRUE, 0, size, temp);
  OCL_CHECK(status, "Failed to read from buffer!");
  out[0] = temp[0];
  delete temp;
}


template<>
void Tanh<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_tanh";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}

// **************************************
// Random functions
// **************************************

/// Seed value required for generating distributions.
static unsigned int seed[4] = {0, 32, 42, 888};
/// Number of generation rounds used in the current algorithm.
static cl_uint rounds = 8;

template<>
void Bernoulli<float, lang::Opencl>(const size_t num, const float p, Block* out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "PRNG_threefry4x32_bernoulli";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, outbuf);
  kernel.setArg(1, seed);
  kernel.setArg(2, 0.0f); // inf
  kernel.setArg(3, 1.0f); // sup
  kernel.setArg(4, p); // threshold
  kernel.setArg(5, rounds);
  kernel.setArg(6, cl_uint(num) / 4);
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num/4));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Gaussian<float, lang::Opencl>(const size_t num, const float mean, const float std, Block* out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "PRNG_threefry4x32_gaussian";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, outbuf);
  kernel.setArg(1, seed);
  kernel.setArg(2, mean); // E
  kernel.setArg(3, std);  // V
  kernel.setArg(4, rounds);
  kernel.setArg(5, cl_uint(num) / 4);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num/4));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Uniform<float, lang::Opencl>(const size_t num, const float low, const float high, Block* out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "PRNG_threefry4x32_uniform";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));
  
  status = kernel.setArg(0, outbuf); OCL_CHECK(status, "kernel arg 0");
  status = kernel.setArg(1, seed); OCL_CHECK(status, "kernel arg 1");
  status = kernel.setArg(2, low); OCL_CHECK(status, "kernel arg 2");
  status = kernel.setArg(3, high); OCL_CHECK(status, "kernel arg 3");
  status = kernel.setArg(4, rounds); OCL_CHECK(status, "kernel arg 4");
  status = kernel.setArg(5, cl_uint(num) / 4); OCL_CHECK(status, "kernel arg 5");

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num/4));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}

// *********************************************************
// BLAS functions, ref to http://docs.nvidia.com/cuda/cublas
// *********************************************************

template<>
void Amax<float, lang::Opencl>(const size_t num, const Block* in, size_t* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_amax";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));

  size_t size = sizeof(size_t) * num;
  cl::Buffer outval(ctx->ocl_ctx, CL_MEM_WRITE_ONLY, size, nullptr, &status);
  OCL_CHECK(status, "Failed to create buffer!");

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outval);
  kernel.setArg(3, cl::Local(size));
  kernel.setArg(4, cl::Local(sizeof(size_t)));

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");

  size_t* temp = new size_t[num];
  status = ctx->ocl_cmdq.enqueueReadBuffer(outval, CL_TRUE, 0, size, temp);
  OCL_CHECK(status, "Failed to read from buffer!");
  out[0] = temp[0];
  delete temp;
}


template<>
void Amin<float, lang::Opencl>(const size_t num, const Block* in, size_t* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_amin";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));

  size_t size = sizeof(size_t) * num;
  cl::Buffer outval(ctx->ocl_ctx, CL_MEM_WRITE_ONLY, size, nullptr, &status);
  OCL_CHECK(status, "Failed to create buffer!");

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outval);
  kernel.setArg(3, cl::Local(size));
  kernel.setArg(4, cl::Local(sizeof(size_t)));

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");

  size_t* temp = new size_t[num];
  status = ctx->ocl_cmdq.enqueueReadBuffer(outval, CL_TRUE, 0, size, temp);
  OCL_CHECK(status, "Failed to read from buffer!");
  out[0] = temp[0];
  delete temp;
}


template<>
void Asum<float, lang::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_asum";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  
  size_t size = sizeof(float) * num;
  cl::Buffer outval(ctx->ocl_ctx, CL_MEM_WRITE_ONLY, size, nullptr, &status);
  OCL_CHECK(status, "Failed to create buffer!");

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outval);
  kernel.setArg(3, cl::Local(size));

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");

  float* temp = new float[num];
  status = ctx->ocl_cmdq.enqueueReadBuffer(outval, CL_TRUE, 0, size, temp);
  OCL_CHECK(status, "Failed to read from buffer!");
  out[0] = temp[0];
  delete temp;
}


template<>
void Axpy<float, lang::Opencl>(const size_t num, const float alpha, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_axpy";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, alpha);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Nrm2<float, lang::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_nrm2";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));

  size_t size = sizeof(float) * num;
  cl::Buffer outval(ctx->ocl_ctx, CL_MEM_WRITE_ONLY, size, nullptr, &status);
  OCL_CHECK(status, "Failed to create buffer!");

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outval);
  kernel.setArg(3, cl::Local(sizeof(float) * (std::pow(2, num))));

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");

  float* temp = new float[num];
  status = ctx->ocl_cmdq.enqueueReadBuffer(outval, CL_TRUE, 0, size, temp);
  OCL_CHECK(status, "Failed to read from buffer!");
  out[0] = temp[0];
  delete temp;
}


template<>
void Scale<float, lang::Opencl>(const size_t num, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_scale";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void Dot<float, lang::Opencl>(const size_t num, const Block *in1, const Block *in2, float *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_dot";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));

  size_t size = sizeof(float) * num;
  cl::Buffer outval(ctx->ocl_ctx, CL_MEM_WRITE_ONLY, size, nullptr, &status);
  OCL_CHECK(status, "Failed to create buffer!");

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outval);
  kernel.setArg(4, cl::Local(size));

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
  OCL_CHECK(status, "Failed to enqueue kernel function!");

  float* temp = new float[num];
  status = ctx->ocl_cmdq.enqueueReadBuffer(outval, CL_TRUE, 0, size, temp);
  OCL_CHECK(status, "Failed to read from buffer!");
  out[0] = temp[0];
  delete temp;
}


template<>
void GEMV<float, lang::Opencl>(bool trans, const size_t m, const size_t n, const float alpha,
		  const Block *A, const Block *v, const float beta, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_gemv";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer Abuf = *(static_cast<cl::Buffer*>(A->mutable_data()));
  cl::Buffer vbuf = *(static_cast<cl::Buffer*>(v->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)m);
  kernel.setArg(1, (cl_int)n);
  kernel.setArg(2, alpha);
  kernel.setArg(3, Abuf);
  kernel.setArg(4, vbuf);
  kernel.setArg(5, beta);
  kernel.setArg(6, outbuf);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(m, n));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void DGMM<float, lang::Opencl>(bool side_right,
		  const size_t nrow, const size_t ncol,
		  const Block *M, const Block *v, Block *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  cl::Buffer Mbuf = *(static_cast<cl::Buffer*>(M->mutable_data()));
  cl::Buffer vbuf = *(static_cast<cl::Buffer*>(v->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  std::string kname;
  if (side_right) {
	DiagVec_Right(ncol, vbuf, vbuf, ctx);
	kname = "clkernel_dgmm_right";
  } else {
	DiagVec_Left(nrow, vbuf, vbuf, ctx);
	kname = "clkernel_dgmm_left";
  }

  auto kernel = ctx->kernels->at(kname);

  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, Mbuf);
  kernel.setArg(3, vbuf);
  kernel.setArg(4, outbuf);
  kernel.setArg(5, cl::Local(sizeof(float) * nrow * ncol));

  cl::NDRange global(nrow); // Only nrow because current implementation is 1 dimensional
//  cl::NDRange local();

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


template<>
void GEMM<float, lang::Opencl>(const bool transA, const bool transB,
		  const size_t nrowA, const size_t ncolB, const size_t ncolA,
		  const float alpha, const Block *A, const Block *B, const float beta,
		  Block *C, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_gemm";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer Abuf = *(static_cast<cl::Buffer*>(A->mutable_data()));
  cl::Buffer Bbuf = *(static_cast<cl::Buffer*>(B->mutable_data()));
  cl::Buffer Cbuf = *(static_cast<cl::Buffer*>(C->mutable_data()));

  // If matrix A needs to be transposed, do it.
  if (transA)
	Transpose(nrowA, ncolA, Abuf, Abuf, ctx);

  // If vector B needs to be transposed, do it.
  if (transB)
	Transpose(nrowA, ncolB, Bbuf, Bbuf, ctx);

  kernel.setArg(0, (cl_int)nrowA);
  kernel.setArg(1, (cl_int)ncolB);
  kernel.setArg(2, (cl_int)ncolA);
  kernel.setArg(3, alpha);
  kernel.setArg(4, Abuf);
  kernel.setArg(5, Bbuf);
  kernel.setArg(6, beta);
  kernel.setArg(7, Cbuf);
  kernel.setArg(8, cl::Local(sizeof(float) * nrowA * ncolB));
  kernel.setArg(9, cl::Local(sizeof(float) * nrowA * ncolB));
  
// TODO: Try to make the work group size a power of 2 given an arbitrary matrix.
  cl::NDRange global(nrowA, ncolB);
  cl::NDRange local(nrowA, ncolB);
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global, local);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}

template <>
void ComputeCrossEntropy<float, lang::Opencl>(const size_t batchsize, const size_t dim,
                         const Block *p, const Block *t, Block *loss,
                         Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_crossentropy";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer pbuf = *(static_cast<cl::Buffer*>(p->mutable_data()));
  cl::Buffer tbuf = *(static_cast<cl::Buffer*>(t->mutable_data()));
  cl::Buffer lossbuf = *(static_cast<cl::Buffer*>(loss->mutable_data()));

  kernel.setArg(0, (cl_uint)batchsize);
  kernel.setArg(1, (cl_uint)dim);
  kernel.setArg(2, pbuf);
  kernel.setArg(3, tbuf);
  kernel.setArg(4, lossbuf);

  cl::NDRange global(batchsize);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}

template <>
void SoftmaxCrossEntropyBwd<float, lang::Opencl>(const size_t batchsize, const size_t dim,
                            const Block *p, const Block *t, Block *grad,
                            Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_softmaxentropy";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer pbuf = *(static_cast<cl::Buffer*>(p->mutable_data()));
  cl::Buffer tbuf = *(static_cast<cl::Buffer*>(t->mutable_data()));
  cl::Buffer gradbuf = *(static_cast<cl::Buffer*>(grad->mutable_data()));

  kernel.setArg(0, (cl_uint)batchsize);
  kernel.setArg(1, (cl_uint)dim);
  kernel.setArg(2, pbuf);
  kernel.setArg(3, tbuf);
  kernel.setArg(4, gradbuf);

  cl::NDRange global(batchsize);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}

// **************************************
// Matrix functions
// **************************************
/*
template<>
void AddCol<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* A, const Block* v, Block* out, Context* ctx) {
  std::string kname = "clkernel_addcol";
  auto kernel = ctx->kernels->at(kname);
  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, static_cast<const float*>(A->mutable_data()));
  kernel.setArg(3, static_cast<const float*>(v->mutable_data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
}

template<>
void AddRow<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* A, const Block* v, Block* out, Context* ctx) {
  std::string kname = "clkernel_addrow";
  auto kernel = ctx->kernels->at(kname);
  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, static_cast<const float*>(A->mutable_data()));
  kernel.setArg(3, static_cast<const float*>(v->mutable_data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
}

template<>
void Outer<float, lang::Opencl>(const size_t m, const size_t n, const Block* lhs, const Block* rhs, Block* out, Context* ctx) {
  std::string kname = "clkernel_outerproduct";
  auto kernel = ctx->kernels->at(kname);
  kernel.setArg(0, (cl_int)m);
  kernel.setArg(1, (cl_int)n);
  kernel.setArg(2, static_cast<const float*>(lhs->data()));
  kernel.setArg(3, static_cast<const float*>(rhs->data()));
  kernel.setArg(4, static_cast<float*>(out->mutable_data()));

  ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(m, n));
}

template<>
void SumColumns<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_sumcol";
  auto kernel = ctx->kernels->at(kname);
  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, static_cast<const float*>(in->mutable_data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
}*/
/*
template<>
void SumRows<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_sumrow";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);
  kernel.setArg(4, cl::Local(sizeof(float) * nrow * ncol));

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
}
*/


#define BLOCK_DIM 16

void Transpose(const size_t nrow, const size_t ncol, cl::Buffer& in, cl::Buffer& out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_transpose";
  auto kernel = ctx->kernels->at(kname);

  kernel.setArg(0, (cl_uint)nrow);
  kernel.setArg(1, (cl_uint)ncol);
  kernel.setArg(2, in);
  kernel.setArg(3, out);
  kernel.setArg(4, cl::Local((BLOCK_DIM + 1) * BLOCK_DIM));

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}

#undef BLOCK_DIM


/// This is a utility function that transforms a single-row vector into a diagonal matrix.
/// For example, a vector of size n will become a matrix of size n*n where only the positions nx == ny will have values.
void DiagVec_Left(const size_t size, cl::Buffer& in, cl::Buffer& out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_diagvec_left";
  auto kernel = ctx->kernels->at(kname);
  
  kernel.setArg(0, (cl_uint)size);
  kernel.setArg(1, in);
  kernel.setArg(2, out);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(size));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}

void DiagVec_Right(const size_t size, cl::Buffer& in, cl::Buffer& out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_diagvec_right";
  auto kernel = ctx->kernels->at(kname);

  kernel.setArg(0, (cl_uint)size);
  kernel.setArg(1, in);
  kernel.setArg(2, out);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(size));
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}

} // namespace singa

#endif // USE_OPENCL

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_OPENCL_H_
