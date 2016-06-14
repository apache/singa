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

// **************************************
// Element-wise functions
// **************************************

// Sum is basically reduction.
// This reduction code is serial reduction modified from AMD's example.
// http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
__kernel 
void clkernel_abs(const int num, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = fabs(input[i]);
}

__kernel
void clkernel_add_scalar(const int num, float x, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = input[i] + x;
}

__kernel
void clkernel_add(const int num, __global const float* in1, __global const float* in2, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = lhs[i] + rhs[i];
}

__kernel
void clkernel_clamp(const int num, float low, float high, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = clamp(input[i], low, high);
}

__kernel
void clkernel_divide_scalar_matx(const int num, __global const float* in1, const float x, __global const float* in2, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = lhs[i] / x;
}

__kernel
void clkernel_divide_scalar_xmat(const int num, const float x, __global const float* in1, __global const float* in2, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = x / lhs[i];
}

__kernel
void clkernel_divide(const int num, __global const float* in1, __global const float* in2, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = lhs[i] / rhs[i];
}

__kernel
void clkernel_eltmult_scalar(const int num, const float x, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = input[i] * x;
}

__kernel
void clkernel_eltmult(const int num, __global const float* in1, __global const float* in2, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = lhs[i] * rhs[i];
}

__kernel
void clkernel_exp(const int num, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = exp(input[i]);
}

__kernel
void clkernel_le(const const int num, __global const float* in, const float x, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  out[i] = (in[i] <= x) ? 1.0f : 0.0f;
}

__kernel
void clkernel_log(const int num, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = log(input[i]);
}

__kernel
void clkernel_lt(const const int num, __global const float* in, const float x, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  out[i] = (in[i] < x) ? 1.0f : 0.0f;
}

__kernel
void clkernel_ge(const const int num, __global const float* in, const float x, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  out[i] = (in[i] >= x) ? 1.0f : 0.0f;
}

__kernel
void clkernel_gt(const const int num, __global const float* in, const float x, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  out[i] = (in[i] > x) ? 1.0f : 0.0f;
}

__kernel
void clkernel_pow_scalar(const int num, const float x, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = pow(input[i], x);
}

__kernel
void clkernel_pow(const int num, __global const float* in1, __global const float* in2, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = pow(in1[i], in2[i]);
}

__kernel
void clkernel_relu(const int num, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = (input[i] > 0) ? input[i] : 0.0f;
}

__kernel
void clkernel_set(const int num, const float x, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = x;
}

__kernel
void clkernel_sigmoid(const int num, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = 1 / (1 + exp(-(input[i])));
}

__kernel
void clkernel_sign(const int num, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = sign(input[i]);
}

__kernel
void clkernel_sqrt(const int num, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = sqrt(input[i]);
}

// kernel for square is called pow(2).

__kernel
void clkernel_subtract_scalar(const int num, __global const float* in, const float x, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = lhs[i] - x;
}

__kernel
void clkernel_subtract(const int num, __global const float* in1, __global const float* in2, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = lhs[i] - rhs[i];
}

// reduce3 kernel from
// https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclReduction/oclReduction_kernel.cl
__kernel 
void clkernel_sum(const int num, __global const float* in, __global float* out, __local float* sdata) {
  const int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);
  const int tid = get_local_id(0);
  sdata[tid] = (i < count) ? input[i] : 0.0f;

  // Perform the first level of reduction.
  if (i + get_local_size(0) < count) {
	sdata[tid] += input[i + get_local_size(0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = get_local_size(0)/2; s > 0; s >>= 1) {
	if (tid > s) {
	  sdata[tid] += sdata[tid + s];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
	ret[get_group_id(0)] = sdata[0];
  }
}

__kernel
void clkernel_tanh(const int num, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = tanh(input[i]);
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

__kernel
void clkernel_amax(const int num, __global const float* in, __global int* ret, __local float* sdata) {
  const int tid = get_local_id(0);
  const int i = get_global_id(0);
  sdata[tid] = (i < count) ? input[i] : -INFINITY;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int s = get_local_size(0)/2; s > 0; s >>= 1) {
	if (tid < s) {
	  sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
	ret[0] = sdata[0];
  }
}

__kernel
void clkernel_amin(const int num, __global const float* in, __global int* ret, __local float* sdata) {
  const int tid = get_local_id(0);
  const int i = get_global_id(0);
  sdata[tid] = (i < count) ? input[i] : INFINITY;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int s = get_local_size(0)/2; s > 0; s >>= 1) {
	if (tid < s) {
	  sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
	ret[0] = sdata[0];
  }
}

__kernel
void clkernel_asum(const int num, __global const float* in, __global float* out, __local float* sdata) {
  const int tid = get_local_id(0);
  const int i = get_global_id(0);
  sdata[tid] = (i < count) ? input[i] : INFINITY;
  // Perform the first level of reduction.
  if (i + get_local_size(0) < count) {
	sdata[tid] += input[i + get_local_size(0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int s = get_local_size(0)/2; s > 0; s >>= 1) {
	if (tid < s) {
	  sdata[tid] = fabs(sdata[tid + s]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
	ret[0] = sdata[0];
  }
}

__kernel
void clkernel_axpy(const int num, float alpha, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = fma(alpha, input[i], ret[i]);
}

// TODO: NRM2

__kernel
void clkernel_scale(const int num, float x, __global float* out) {
  const int i = get_global_id(0);
  if (i >= count) return;
  ret[i] = x * ret[i];
}

__kernel
void clkernel_dot(const const int num, __global const float* in1, __global const float* in2, 
	  __global float* out, __local float* scratch) {
  const int i = get_global_id(0);
  if (i >= count) return;
  int offset = i << 2;
  scratch[i] = in1[offset] * in2[offset];
  
}

// TODO: GEMV

__kernel
void clkernel_dgmm_left(const int nrow, const int ncol,
	__global const float* M, __global const float* v, __global float* out) {
  //TODO
}

__kernel
void clkernel_dgmm_right(const int nrow, const int ncol,
	__global const float* M, __global const float* v, __global float* out) {
  //TODO
}

__kernel
void clkernel_gemm(const int nrowA, const int ncolB, const int ncolA, const float alpha,
	__global const float *A, __global const float* B, const float beta, __global float* C) {
  //TODO
}

// TODO: ComputeCrossEntropy

// TODO: SoftmaxCrossEntropyBwd


// **************************************
// Matrix functions
// **************************************
/*
__kernel
void clkernel_addcol(int nrow, int ncol, __global const float* A, __global const float* v, __global float* out) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  if (i >= nrow) return;
  if (j >= ncol) return;
  ret[j] = A[j + nrow * i] + v[j];
}

__kernel
void clkernel_addrow(int nrow, int ncol, __global const float* A, __global const float* v, __global float* out) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  if (i >= nrow) return;
  if (j >= ncol) return;
  ret[i] = A[i + ncol * j] + v[i];
}

__kernel
void clkernel_outerproduct(int m, const int n, __global const float* in1, __global const float* in2, __global float* out) {
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  
  // TODO: This
}

__kernel
void clkernel_sumcol(int nrow, int ncol, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= nrow) return;

  float sum = 0.0f;
  for (int j = 0; j < nrow; j++) {
	sum += input[nrow * i + j];
  }
  ret[i] = sum;
}

__kernel
void clkernel_sumrow(int nrow, int ncol, __global const float* in, __global float* out) {
  const int i = get_global_id(0);
  if (i >= nrow) return;
  
  float sum = 0.0f;
  for (int j = 0; j < ncol; j++) {
	sum += input[ncol * i + j];
  }
  ret[i] = sum;
}*/
