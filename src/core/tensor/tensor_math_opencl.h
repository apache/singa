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

#include "tensor_math.h"
#include "singa/utils/opencl_utils.h"

#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>

#include <viennacl/linalg/prod.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/sum.hpp>
#include <viennacl/linalg/scalar_operations.hpp>
#include <viennacl/linalg/vector_operations.hpp>
#include <viennacl/linalg/matrix_operations.hpp>

#include <viennacl/ocl/kernel.hpp>

using viennacl::ocl::get_context;
using viennacl::ocl::enqueue;

namespace singa {

// **************************************
// Element-wise functions
// **************************************

template<>
void Abs<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_fabs");

  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = v_in;
  enqueue(kernel((cl_int)num, v_in, v_out));
}


template<>
void Add<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);

  viennacl::vector<float> x_in = viennacl::scalar_vector<float>(num, x, ocl_ctx);
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = v_in + x_in;
}


template<>
void Add<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  viennacl::vector<float> v_in1((const cl_mem)in1->data(), num);
  viennacl::vector<float> v_in2((const cl_mem)in2->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = v_in1 + v_in2;
}


template<>
void Clamp<float, lang::Opencl>(const size_t num, const float low, const float high,
                                const Block* in, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_clamp");

  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  enqueue(kernel((cl_int)num, low, high, v_in, v_out));
}


template<>
void Div<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);

  viennacl::vector<float> x_in = viennacl::scalar_vector<float>(num, x, ocl_ctx);
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_div(v_in, x_in);
}


template<>
void Div<float, lang::Opencl>(const size_t num, const float x, const Block* in, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);

  viennacl::vector<float> x_in = viennacl::scalar_vector<float>(num, x, ocl_ctx);
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_div(x_in, v_in);
}


template<>
void Div<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  viennacl::vector<float> v_in1((const cl_mem)in1->data(), num);
  viennacl::vector<float> v_in2((const cl_mem)in2->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_div(v_in1, v_in2);
}


template<>
void EltwiseMult<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);

  viennacl::vector<float> x_in = viennacl::scalar_vector<float>(num, x, ocl_ctx);
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_prod(v_in, x_in);
}


template<>
void EltwiseMult<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  viennacl::vector<float> v_in1((const cl_mem)in1->data(), num);
  viennacl::vector<float> v_in2((const cl_mem)in2->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_prod(v_in1, v_in2);
}


template<>
void Exp<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_exp(v_in);
}


template<>
void LE<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_le");

  viennacl::vector<float> in_buf((const cl_mem)in->data(), num);
  viennacl::vector<float> out_buf(static_cast<cl_mem>(out->mutable_data()), num);

  enqueue(kernel((cl_int)num, in_buf, x, out_buf));
}


template<>
void Log<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_log(v_in);
}


template<>
void LT<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_lt");

  viennacl::vector<float> in_buf((const cl_mem)in->data(), num);
  viennacl::vector<float> out_buf(static_cast<cl_mem>(out->mutable_data()), num);

  enqueue(kernel((cl_int)num, in_buf, x, out_buf));
}


template<>
void GE<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_ge");

  viennacl::vector<float> in_buf((const cl_mem)in->data(), num);
  viennacl::vector<float> out_buf(static_cast<cl_mem>(out->mutable_data()), num);

  enqueue(kernel((cl_int)num, in_buf, x, out_buf));
}


template<>
void GT<float, lang::Opencl>(const size_t num, const Block *in, const float x, Block *out, Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_gt");

  viennacl::vector<float> in_buf((const cl_mem)in->data(), num);
  viennacl::vector<float> out_buf(static_cast<cl_mem>(out->mutable_data()), num);

  enqueue(kernel((cl_int)num, in_buf, x, out_buf));
}


template<>
void Pow<float, lang::Opencl>(const size_t num, const Block* in, float x, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);

  viennacl::vector<float> x_in = viennacl::scalar_vector<float>(num, x, ocl_ctx);
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_pow(v_in, x_in);
}


template<>
void Pow<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  viennacl::vector<float> v_in1((const cl_mem)in1->data(), num);
  viennacl::vector<float> v_in2((const cl_mem)in2->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_pow(v_in1, v_in2);
}


template<>
void ReLU<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_relu");

  viennacl::vector<float> in_buf((const cl_mem)in->data(), num);
  viennacl::vector<float> out_buf(static_cast<cl_mem>(out->mutable_data()), num);

  enqueue(kernel((cl_int)num, in_buf, out_buf));
}


template<>
void Set<float, lang::Opencl>(const size_t num, const float x, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);

  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::scalar_vector<float>(num, x, ocl_ctx);
}


template<>
void Sigmoid<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);

  const viennacl::vector<float> zero = viennacl::zero_vector<float>(num, ocl_ctx);
  const viennacl::vector<float> one = viennacl::scalar_vector<float>(num, 1.0f, ocl_ctx);

  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_div(one, viennacl::linalg::element_exp(zero - v_in) + one);
}


template<>
void Sign<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_sign");

  viennacl::vector<float> in_buf((const cl_mem)in->data(), num);
  viennacl::vector<float> out_buf(static_cast<cl_mem>(out->mutable_data()), num);

  enqueue(kernel(num, in_buf, out_buf));
}


template<>
void Sqrt<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_sqrt(v_in);
}


template<>
void Square<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  Pow<float, lang::Opencl>(num, in, 2, out, ctx);
}


template<>
void Sub<float, lang::Opencl>(const size_t num, const Block* in, const float x, Block* out, Context* ctx) {
  Add<float, lang::Opencl>(num, in, -x, out, ctx);
}


template<>
void Sub<float, lang::Opencl>(const size_t num, const Block* in1, const Block* in2, Block* out, Context* ctx) {
  viennacl::vector<float> v_in1((const cl_mem)in1->data(), num);
  viennacl::vector<float> v_in2((const cl_mem)in2->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = v_in1 - v_in2;
}


template<>
void Sum<float, lang::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);

  out[0] = viennacl::linalg::sum(v_in);
}


template<>
void Tanh<float, lang::Opencl>(const size_t num, const Block* in, Block* out, Context* ctx) {
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_tanh(v_in);
}

// **************************************
// Random functions
// **************************************

/// Number of generation rounds used in the current algorithm.
static cl_uint rounds = 8;

template<>
void Bernoulli<float, lang::Opencl>(const size_t num, const float p, Block* out, Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_distribution", "PRNG_threefry4x32_bernoulli");

  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  viennacl::ocl::packed_cl_uint seed = {0, 32, 42, 888};

  enqueue(kernel(v_out, seed, 0.0f, 1.0f, p, rounds, cl_uint(num / 4)));
}


template<>
void Gaussian<float, lang::Opencl>(const size_t num, const float mean, const float std, Block* out, Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_distribution", "PRNG_threefry4x32_gaussian");

  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  viennacl::ocl::packed_cl_uint seed = {0, 32, 42, 888};

  enqueue(kernel(v_out, seed, mean, std, rounds, cl_uint(num/4)));
}


template<>
void Uniform<float, lang::Opencl>(const size_t num, const float low, const float high, Block* out, Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_distribution", "PRNG_threefry4x32_uniform");

  viennacl::ocl::packed_cl_uint seed = {0, 32, 42, 888};

  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  enqueue(kernel(v_out, seed, low, high, rounds, cl_uint(num/4)));
}

// *********************************************************
// BLAS functions, ref to http://docs.nvidia.com/cuda/cublas
// *********************************************************
/*
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
*/
	
template<>
void Asum<float, lang::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  viennacl::vector<float> v_in((const cl_mem)in->data(), num);

  viennacl::vector<float> temp = viennacl::linalg::element_fabs(v_in);

  out[0] = viennacl::linalg::sum(temp);
}

/// out = alpha * in + out
template<>
void Axpy<float, lang::Opencl>(const size_t num, const float alpha, const Block* in, Block* out, Context* ctx) {
  viennacl::vector<float> inbuf((const cl_mem)in->data(), num);
  viennacl::vector<float> outbuf(static_cast<cl_mem>(out->mutable_data()), num);

  outbuf += alpha * inbuf;
}

/// out = ||in||_2^2, i.e, L2 norm.
template<>
void Nrm2<float, lang::Opencl>(const size_t num, const Block* in, float* out, Context* ctx) {
  viennacl::vector<float> inbuf((const cl_mem)in->data(), num);

  out[0] = viennacl::linalg::norm_2(inbuf);
}


template<>
void Scale<float, lang::Opencl>(const size_t num, const float x, Block* out, Context* ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);

  viennacl::vector<float> x_in = viennacl::scalar_vector<float>(num, x, ocl_ctx);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), num);

  v_out = viennacl::linalg::element_prod(v_out, x_in);
}


template<>
void Dot<float, lang::Opencl>(const size_t num, const Block *in1, const Block *in2, float *out, Context *ctx) {
  viennacl::vector<float> in1_buf((const cl_mem)in1->data(), num);
  viennacl::vector<float> in2_buf((const cl_mem)in2->data(), num);

  out[0] = viennacl::linalg::inner_prod(in1_buf, in2_buf);
}

/// out = alpha * A * v + beta * out.
template<>
void GEMV<float, lang::Opencl>(bool trans, const size_t m, const size_t n, const float alpha,
		  const Block *A, const Block *v, const float beta, Block* out, Context* ctx) {
  viennacl::vector<float> v_buf((const cl_mem)v->data(), n);
  viennacl::vector<float> o_buf(static_cast<cl_mem>(out->mutable_data()), m);

  viennacl::matrix<float> A_buf;

  if (trans) {
    A_buf = viennacl::matrix<float>((const cl_mem)A->data(), n, m);
    A_buf = viennacl::trans(A_buf);
  } else {
    A_buf = viennacl::matrix<float>((const cl_mem)A->data(), m, n);
  }

  o_buf *= beta;
  o_buf += alpha * viennacl::linalg::prod(A_buf, v_buf);
}

/// multiply a matrix with a diagonal matrix constructed using values from 'v'.
/// if matrix_left_side is true, do M*v; else do v*M
template<>
void DGMM<float, lang::Opencl>(bool side_right,
		  const size_t nrow, const size_t ncol,
		  const Block *M, const Block *v, Block *out, Context *ctx) {

  viennacl::matrix<float> M_buf((const cl_mem)M->data(), nrow, ncol);
  viennacl::vector<float> v_buf((const cl_mem)v->data(), nrow);
  viennacl::matrix<float> out_buf(static_cast<cl_mem>(out->mutable_data()), nrow, ncol);

  auto diag = viennacl::diag(v_buf);

  if (side_right) {
    out_buf = viennacl::linalg::prod(M_buf, diag);
  } else {
    out_buf = viennacl::linalg::prod(diag, M_buf);
  }
}

/// C = alpha * A * B + beta * C.
template<>
void GEMM<float, lang::Opencl>(const bool transA, const bool transB,
		  const size_t nrowA, const size_t ncolB, const size_t ncolA,
		  const float alpha, const Block *A, const Block *B, const float beta,
		  Block *C, Context *ctx) {

  viennacl::matrix<float> A_buf, B_buf;
  viennacl::matrix<float> C_buf(static_cast<cl_mem>(C->mutable_data()), nrowA, ncolB);

  if (transA) {
    A_buf = viennacl::matrix<float>((const cl_mem)A->data(), ncolA, nrowA);
    A_buf = viennacl::trans(A_buf);
  } else {
    A_buf = viennacl::matrix<float>((const cl_mem)A->data(), nrowA, ncolA);
  }

  if (transB) {
    B_buf = viennacl::matrix<float>((const cl_mem)B->data(), ncolB, ncolA);
    B_buf = viennacl::trans(B_buf);
  } else {
    B_buf = viennacl::matrix<float>((const cl_mem)B->data(), ncolA, ncolB);
  }

  C_buf *= beta;
  C_buf += alpha * viennacl::linalg::prod(A_buf, B_buf);
}


template <>
void ComputeCrossEntropy<float, lang::Opencl>(bool int_target, const size_t batchsize,
                         const size_t dim, const Block *p, const Block *t,
                         Block *loss, Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_crossentropy");

  viennacl::vector<float> p_buf((const cl_mem)p->data(), batchsize);
  viennacl::vector<float> t_buf((const cl_mem)t->data(), batchsize);
  viennacl::vector<float> loss_buf(static_cast<cl_mem>(loss->mutable_data()), batchsize);

  enqueue(kernel((cl_uint)batchsize, (cl_uint)dim, p_buf, t_buf, loss_buf));
}


template <>
void SoftmaxCrossEntropyBwd<float, lang::Opencl>(bool int_target, const size_t batchsize, const size_t dim,
                            const Block *p, const Block *t, Block *grad,
                            Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_softmaxentropy");

  viennacl::vector<float> p_buf((const cl_mem)p->data(), batchsize);
  viennacl::vector<float> t_buf((const cl_mem)t->data(), batchsize);
  viennacl::vector<float> grad_buf(static_cast<cl_mem>(grad->mutable_data()), batchsize);

  enqueue(kernel((cl_uint)batchsize, (cl_uint)dim, p_buf, t_buf, grad_buf));
}


template<>
void RowMax<float, lang::Opencl>(const size_t nrow, const size_t ncol,
                                 const Block *in, Block *out, Context *ctx) {
  auto ocl_ctx = get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_tensor_math", "clkernel_rowmax");

//  kernel.global_work_size(0, nrow);

  viennacl::matrix<float> in_buf((const cl_mem)in->data(), nrow, ncol);
  viennacl::vector<float> outbuf(static_cast<cl_mem>(out->mutable_data()), nrow);

  enqueue(kernel((cl_uint)nrow, (cl_uint)ncol, in_buf, outbuf));
}

// **************************************
// Matrix functions
// **************************************
/*
template<>
void AddCol<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* A, const Block* v, Block* out, Context* ctx) {

}


template<>
void AddRow<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* A, const Block* v, Block* out, Context* ctx) {

}


template<>
void Outer<float, lang::Opencl>(const size_t m, const size_t n, const Block* lhs, const Block* rhs, Block* out, Context* ctx) {
  viennacl::vector<float> lhs_in((const cl_mem)lhs->data(), m);
  viennacl::vector<float> rhs_in((const cl_mem)rhs->data(), n);
  viennacl::matrix<float> out_buf(static_cast<cl_mem>(out->mutable_data()), m, n);

  out_buf = viennacl::linalg::outer_prod(lhs_in, rhs_in);
}


template<>
void SumColumns<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* in, Block* out, Context* ctx) {
  viennacl::matrix<float> m_in((const cl_mem)in->data(), nrow, ncol);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), nrow);

  v_out = viennacl::linalg::column_sum(m_in);
}


template<>
void SumRows<float, lang::Opencl>(const size_t nrow, const size_t ncol, const Block* in, Block* out, Context* ctx) {
  viennacl::matrix<float> m_in((const cl_mem)in->data(), nrow, ncol);
  viennacl::vector<float> v_out(static_cast<cl_mem>(out->mutable_data()), ncol);

  v_out = viennacl::linalg::column_sum(m_in);
}
*/

} // namespace singa

#endif // USE_OPENCL

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_OPENCL_H_v_in + x;
