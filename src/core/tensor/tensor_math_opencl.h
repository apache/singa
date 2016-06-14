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
#include <CL/cl2.hpp>
#include "./tensor_math.h"

// Kernel names
constexpr const char* clkernel_abs = "clkernel_abs";

namespace singa {

// **************************************
// Element-wise functions
// **************************************

template<>
void Abs<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_abs";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Add<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_add_scalar";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Add<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_add";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Clamp<float, lang::Opencl>(const size_t num, const float low, const float high, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_clamp";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, low);
  kernel.setArg(2, high);
  kernel.setArg(3, inbuf);
  kernel.setArg(4, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Div<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_divide_scalar_matx";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Div<float, lang::Opencl>(const size_t num, const float x, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_divide_scalar_xmat";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Div<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_divide";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void EltwiseMult<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_eltmult_scalar";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void EltwiseMult<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_eltmult";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Exp<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_exp";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void LE<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_le";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Log<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_log";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void LT<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_lt";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void GE<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_ge";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void GT<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  cl_int statuas = CL_SUCCESS;

  std::string kname = "clkernel_gt";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf;
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Pow<float, lang::Opencl>(const size_t num, const Block* in, float x, Block* out, Context* ctx) {
  cl_int statuas = CL_SUCCESS;

  std::string kname = "clkernel_pow_scalar";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Pow<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int statuas = CL_SUCCESS;

  std::string kname = "clkernel_pow";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void ReLU<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_relu";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}

template<>
void Set<float, lang::Opencl>(const size_t num, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_set";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Sigmoid<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_sigmoid";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Sign<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_sign";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Sqrt<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_sqrt";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Square<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  Pow<float, lang::Opencl>(num, 2, in, out, ctx);
}


template<>
void Sub<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_subtract_scalar";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, x);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Sub<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_subtract";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Sum<float, lang::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_reduce";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);
  kernel.setArg(3, cl::Local(sizeof(float) * num);
  
  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Tanh<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_tanh";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}

// **************************************
// Random functions
// **************************************

template<>
void Bernoulli<float, lang::Opencl>(const size_t num, const float p, Block* out, Context *ctx) {
  cl_int status = CL_SUCCESS;

//std::string kname = "clkernel_bernoulli";
  std::string kname = "PRNG_threefry4x32_bernoulli";
  auto kernel = oclDevice.GetKernel(kname);
  
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)(num / 4)); // Divide by 4 because kernel uses float4 as argument.
  kernel.setArg(1, p);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Gaussian<float, lang::Opencl>(const size_t num, const float mean, const float std, Block* out, Context *ctx) {
  cl_int status = CL_SUCCESS;

//std::string kname = "clkernel_gaussian";
  std::string kname = "PRNG_threefry4x32_gaussian";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)(num / 4));
  kernel.setArg(1, mean);
  kernel.setArg(2, std);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Uniform<float, lang::Opencl>(const size_t num, const float low, const float high, Block* out, Context *ctx) {
  cl_int status = CL_SUCCESS;

//std::string kname = "clkernel_uniform";
  std::string kname = "PRNG_threefry4x32_uniform";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)(num / 4));
  kernel.setArg(1, low);
  kernel.setArg(2, high);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}

// *********************************************************
// BLAS functions, ref to http://docs.nvidia.com/cuda/cublas
// *********************************************************

template<>
void Amax<float, lang::Opencl>(const size_t num, const Block* in, size_t* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_amax";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, out);
  kernel.setArg(3, cl::Local(total_mem));
  kernel.setArg(4, cl::Local(sizeof(size_t)));

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Amin<float, lang::Opencl>(const size_t num, const Block* in, size_t* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_amin";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);
  kernel.setArg(3, cl::Local(total_mem));
  kernel.setArg(4, cl::Local(sizeof(size_t)));

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Asum<float, lang::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_asum";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);
  kernel.setArg(3, cl::Local(sizeof(float) * num));

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Axpy<float, lang::Opencl>(const size_t num, const float alpha, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_axpy";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, alpha);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Nrm2<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_nrm2";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, alpha);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);
  kernel.setArg(4, cl::Local(sizeof(float) * (std::pow(2, num))));

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Scale<float, lang::Opencl>(const size_t num, const float x, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_scale";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, x);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void Dot<float, lang::Opencl>(const size_t num, const Block *in1, const Block *in2, float *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_dot";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer in1buf = *(static_cast<cl::Buffer*>(in1->mutable_data()));
  cl::Buffer in2buf = *(static_cast<cl::Buffer*>(in2->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)num);
  kernel.setArg(1, in1buf);
  kernel.setArg(2, in2buf);
  kernel.setArg(3, outbuf);
  kernel.setArg(4, cl::Local(sizeof(float) * num))

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(num));
}


template<>
void GEMV<float, lang::Opencl>(bool trans, const size_t m, const size_t n, const float alpha,
		  const Block *A, const Block *v, const float beta, Block* out) {
  cl_int status = CL_SUCCESS;

  std::string kname "clkernel_gemv";
  auto kernel = oclDevice.GetKernel(kname);

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

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(m, n));
}


template<>
void DGMM<float, lang::Opencl>(bool side_right,
		  const size_t nrow, const size_t ncol,
		  const Block *M, const Block *v, Block *out, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname;
  if (side_right) {
	DiagVec_Right(ncol, v, v);
	kname = "clkernel_dgmm_right";
  } else {
	DiagVec_Left(nrow, v, v);
	kname = "clkernel_dgmm_left";
  }

  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer Mbuf = *(static_cast<cl::Buffer*>(M->mutable_data()));
  cl::Buffer vbuf = *(static_cast<cl::Buffer*>(v->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, Mbuf);
  kernel.setArg(3, vbuf);
  kernel.setArg(4, outbuf);
  kernel.setArg(5, cl::Local(sizeof(float) * nrow * ncol));

  cl::NDRange global(nrow); // Only nrow because current implementation is 1 dimensional
//  cl::NDRange local();

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
}


template<>
void GEMM<float, lang::Opencl>(const bool transA, const bool transB,
		  const size_t nrowA, const size_t ncolB, const size_t ncolA,
		  const float alpha, const Block *A, const Block *B, const float beta,
		  Block *C, Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_gemm";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer Abuf = *(static_cast<cl::Buffer*>(A->mutable_data()));
  cl::Buffer Bbuf = *(static_cast<cl::Buffer*>(B->mutable_data()));
  cl::Buffer Cbuf = *(static_cast<cl::Buffer*>(C->mutable_data()));

  // If matrix A needs to be transposed, do it.
  if (!transA)
	Transpose<float, lang::Opencl>(nrowA, ncolA, A, A, ctx);

  // If vector B needs to be transposed, do it.
  if (!transB)
	Transpose<float, lang::Opencl>(nrowA, ncolB, B, B, ctx);

  kernel.setArg(0, (cl_int)nrowA);
  kernel.setArg(1, (cl_int)ncolB);
  kernel.setArg(2, (cl_int)ncolA);
  kernel.setArg(3, alpha);
  kernel.setArg(4, Abuf);
  kernel.setArg(5, Bbuf);
  kernel.setArg(6, beta);
  kernel.setArg(7, Cbuf);

  cl::NDRange global(nrowA, ncolA);
  cl::NDRange local(32, 32);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), global, local);
}

template <>
void ComputeCrossEntropy<float, lang::Opencl>(const size_t batchsize, const size_t dim,
                         const Block *p, const Block *t, Block *loss,
                         Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_crossentropy";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer pbuf = *(static_cast<cl::Buffer*>(p->mutable_data()));
  cl::Buffer tbuf = *(static_cast<cl::Buffer*>(t->mutable_data()));
  cl::Buffer lossbuf = *(static_cast<cl::Buffer*>(loss->mutable_data()));

  kernel.setArg(0, (cl_uint)batchsize);
  kernel.setArg(1, (cl_uint)dim);
  kernel.setArg(2, pbuf);
  kernel.setArg(3, tbuf);
  kernel.setArg(4, lossbuf);

  cl::NDRange global(batchsize);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
}

template <>
void SoftmaxCrossEntropyBwd<float, lang::Opencl>(const size_t batchsize, const size_t dim,
                            const Block *p, const Block *t, Block *grad,
                            Context *ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_softmaxentropy";
  auto kernel = oclDervice.GetKernel(kname);

  cl::Buffer pbuf = *(static_cast<cl::Buffer*>(p->mutable_data()));
  cl::Buffer tbuf = *(static_cast<cl::Buffer*>(t->mutable_data()));
  cl::Buffer gradbuf = *(static_cast<cl::Buffer*>(grad->mutable_data()));

  kernel.setArg(0, (cl_uint)batchsize);
  kernel.setArg(1, (cl_uint)dim);
  kernel.setArg(2, pbuf);
  kernel.setArg(3, tbuf);
  kernel.setArg(4, gradbuf);

  cl::NDRange global(batchsize);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
}

// **************************************
// Matrix functions
// **************************************
/*
template<>
void AddCol<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* A, const Block* v, Block* out, Context* ctx) {
  std::string kname = "clkernel_addcol";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, static_cast<const float*>(A->mutable_data()));
  kernel.setArg(3, static_cast<const float*>(v->mutable_data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
}

template<>
void AddRow<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* A, const Block* v, Block* out, Context* ctx) {
  std::string kname = "clkernel_addrow";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, static_cast<const float*>(A->mutable_data()));
  kernel.setArg(3, static_cast<const float*>(v->mutable_data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
}

template<>
void Outer<float, lang::Opencl>(const size_t m, const size_t n, const Block* lhs, const Block* rhs, Block* out, Context* ctx) {
  std::string kname = "clkernel_outerproduct";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (cl_int)m);
  kernel.setArg(1, (cl_int)n);
  kernel.setArg(2, static_cast<const float*>(lhs->data()));
  kernel.setArg(3, static_cast<const float*>(rhs->data()));
  kernel.setArg(4, static_cast<float*>(out->mutable_data()));

  ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(m, n));
}

template<>
void SumColumns<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* in, Block* out, Context* ctx) {
  std::string kname = "clkernel_sumcol";
  auto kernel = oclDevice.GetKernel(kname);
  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, static_cast<const float*>(in->mutable_data()));
  kernel.setArg(3, static_cast<float*>(out->mutable_data()));

  ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
}*/

template<>
void SumRows<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_sumrow";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_int)nrow);
  kernel.setArg(1, (cl_int)ncol);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);
  kernel.setArg(4, cl::Local(sizeof(float) * nrow * ncol));

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
}

#define BLOCK_DIM 16

template<>
void Transpose<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* in, Block* out, Context* ctx) {
  cl_int status = CL_SUCCESS;

  std::string kname = "clkernel_transpose";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_uint)nrow);
  kernel.setArg(1, (cl_uint)ncol);
  kernel.setArg(2, inbuf);
  kernel.setArg(3, outbuf);
  kernel.setArg(4, cl::Local((BLOCK_DIM + 1) * BLOCK_DIM));

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(nrow, ncol));
}

#undef BLOCK_DIM

/// This is a utility function that transforms a single-row vector into a diagonal matrix.
/// For example, a vector of size n will become a matrix of size n*n where only the positions nx == ny will have values.
void DiagVec_Left(const size_t size, const Block* in, Block* out, Context* ctx) {
  cl_int sstatus = CL_SUCCESS;

  std::string kname = "clkernel_diagvec_left";
  auto kernel = oclDevice.GetKernel(kname);

  cl::Buffer inbuf = *(static_cast<cl::Buffer*>(in->mutable_data()));
  cl::Buffer outbuf = *(static_cast<cl::Buffer*>(out->mutable_data()));

  kernel.setArg(0, (cl_uint)size);
  kernel.setArg(1, inbuf);
  kernel.setArg(2, outbuf);

  status = ctx.get_dev_cmdq(0).enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(size));
}

} // namespace singa

#endif // USE_OPENCL

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_OPENCL_H_
