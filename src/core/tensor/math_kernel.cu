/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "singa/singa_config.h"
#ifdef USE_CUDA
#include <cmath>
#include <algorithm>
#include <cfloat>
#include "./math_kernel.h"

#define CU2DBLOCK_X 32
#define CU2DBLOCK_Y 32

#define CU1DBLOCK 1024
#define CU1DBLOCKF 1024.0

namespace singa {
// Cuda Kernel Functions
namespace cuda {
/*
wangwei: Not used due to error in the code.
__global__ void KernelSum(const size_t n, const float *in, float *out) {
  int THREADS = blockDim.x;

  __shared__ float aux[CU1DBLOCK];
  int steps = (n - 1) / THREADS + 1;
  aux[threadIdx.x] = in[threadIdx.x];

  for (int i = 1; i < steps; ++i) {
    if (threadIdx.x + i * THREADS < n) {
      aux[threadIdx.x] += in[threadIdx.x + i * THREADS];
    }
  }

  int total_threads = THREADS;
  __syncthreads();

  while (total_threads > 1) {
    int half_point = ((1 + total_threads) >> 1);
    if (threadIdx.x < half_point) {
      if (threadIdx.x + half_point < total_threads) {
        aux[threadIdx.x] += aux[threadIdx.x + half_point];
      }
    }
    __syncthreads();
    total_threads = ((total_threads + 1) >> 1);
  }

  __syncthreads();
  *out = aux[0];
}
*/

__global__ void KernelBroadcastTo(const size_t n, size_t nDim, const float *in,const float* shape, const float* stride, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    int shape_accu = n;
    size_t offset = 0;
    int remains = i;

    for (int k = 0; k < nDim; k++) {
      shape_accu = shape_accu/shape[k];
      int idx = remains/shape_accu;
      remains = remains%shape_accu;
      offset = offset + idx*stride[k];
    }
    out[i] = in[offset];
  }
}

__global__ void KernelAdd(const size_t n, const float *in1, const float *in2,
                          float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in1[i] + in2[i];
  }
}

__global__ void KernelAdd(const size_t n, const float *in, const float x,
                          float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] + x;
  }
}

__global__ void KernelSub(const size_t n, const float *in1, const float *in2,
                          float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in1[i] - in2[i];
  }
}

__global__ void KernelExp(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = std::exp(in[i]);
  }
}

__global__ void KernelCeil2(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = std::ceil(in[i]);
  }
}

__global__ void KernelLog(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = std::log(in[i]);
  }
}

__global__ void KernelSigmoid(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = 1.0f / (1.0f + expf(-in[i]));
  }
}
__global__ void KernelSign(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    if (in[i] > 0.0f)
      out[i] = 1.0f;
    else if (in[i] < 0.0f)
      out[i] = -1.0f;
    else
      out[i] = 0.0f;
  }
}

__global__ void KernelClamp(const size_t n, const float low, const float high,
                            const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    if (in[i] > high)
      out[i] = high;
    else if (in[i] < low)
      out[i] = low;
    else
      out[i] = in[i];
  }
}

__global__ void KernelRelu(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] > 0 ? in[i] : 0.0f;
  }
}

__global__ void KernelReLUBackward(const size_t n, const float *in1, const float *in2,
                         float *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in2[i] > 0 ? in1[i] : 0.0f;
  }
}

__global__ void KernelAbs(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] =  max(in[i], -in[i]);
  }
}

__global__ void KernelCastFloat2Int(const size_t n, const float *in, int *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = int(in[i]);
  }
}

__global__ void KernelCastInt2Float(const size_t n, const int *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = float(in[i]);
  }
}

__global__ void KernelSoftplus(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = logf(1 + expf(in[i]));
  }
}
  
__global__ void KernelSoftsign(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] / (max(in[i], -in[i]) + 1);
  }
}

__global__ void KernelSquare(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] * in[i];
  }
}
__global__ void KernelSqrt(const size_t n, const float *in, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = std::sqrt(in[i]);
  }
}

__global__ void KernelPow(const size_t n, const float *in1, const float *in2,
                          float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = std::pow(in1[i], in2[i]);
  }
}

__global__ void KernelPow(const size_t n, const float *in, const float x,
                          float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = std::pow(in[i], x);
  }
}

__global__ void KernelMult(const size_t n, const float *in1, const float *in2,
                           float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in1[i] * in2[i];
  }
}

__global__ void KernelMult(const size_t n, const float *in, const float x,
                           float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] * x;
  }
}

__global__ void KernelDiv(const size_t n, const float *in1, const float *in2,
                          float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in1[i] / in2[i];
  }
}
__global__ void KernelDiv(const size_t n, const float x, const float *in,
                          float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = x / in[i];
  }
}
__global__ static void KernelSet(const size_t n, const float x, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = x;
  }
}

__global__ void KernelThreshold(const size_t n, const float x, const float *in,
                                float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] < x ? 1.0f : 0.0f;
  }
}

__global__ void KernelGE(const size_t num, const float *in, const float x,
                         float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in[idx] >= x ? 1.0f : 0.0f;
  }
}

__global__ void KernelBGE(const size_t num, const float *in1, const float *in2,
                         float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in1[idx] >= in2[idx] ? 1.0f : 0.0f;
  }
}
__global__ void KernelGT(const size_t num, const float *in, const float x,
                         float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in[idx] > x ? 1.0f : 0.0f;
  }
}
__global__ void KernelBGT(const size_t num, const float *in1, const float *in2,
                         float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in1[idx] > in2[idx] ? 1.0f : 0.0f;
  }
}
__global__ void KernelLE(const size_t num, const float *in, const float x,
                         float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in[idx] <= x ? 1.0f : 0.0f;
  }
}
__global__ void KernelBLE(const size_t num, const float *in1, const float *in2,
                         float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in1[idx] <= in2[idx] ? 1.0f : 0.0f;
  }
}
__global__ void KernelLT(const size_t num, const float *in, const float x,
                         float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in[idx] < x ? 1.0f : 0.0f;
  }
}
__global__ void KernelBLT(const size_t num, const float *in1, const float *in2,
                         float *out) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in1[idx] < in2[idx] ? 1.0f : 0.0f;
  }
}
__global__ void KernelRowMax(const size_t nrow, const size_t ncol, const float *inPtr,
    float *outPtr) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nrow;
       idx += blockDim.x * gridDim.x) {
    int offset = idx * ncol;
    float maxval = inPtr[offset];
    for (size_t k = 1; k < ncol; k++) {
      maxval = max(maxval, inPtr[offset + k]);
    }
    outPtr[idx] = maxval;
  }
}
__global__ void KernelComputeCrossEntropy(const bool int_target, const size_t batchsize,
                                          const size_t dim, const float *p,
                                          const int *t, float *loss) {
  size_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  size_t num_threads = blockDim.x * gridDim.x;
  if (int_target) {
    for (; sample < batchsize; sample += num_threads) {
      float prob_of_truth = p[sample * dim + t[sample]];
      loss[sample] = -std::log(max(prob_of_truth, FLT_MIN));
    }
  } else {
    for (; sample < batchsize; sample += num_threads) {
      float sum = 0.f;
      for (size_t j = 0; j < dim; j++) {
        sum += t[sample * dim + j];
      }
      loss[sample] = 0;
      for (size_t j = 0, offset = sample * dim; j < dim; j++, offset++) {
        loss[sample] -= t[offset] / sum * std::log(max(p[offset], FLT_MIN));
      }
    }
  }
}

__global__ void KernelSoftmaxCrossEntropyBwd(const bool int_target, const size_t batchsize,
                                             const size_t dim, const float *p,
                                             const int *t, float *grad) {
  size_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  size_t num_threads = blockDim.x * gridDim.x;
  if (int_target) {
    for (; sample < batchsize; sample += num_threads) {
      size_t pos = sample * dim + t[sample];
      grad[pos] = p[pos] - 1.0f;  // TODO(wangwei) Consider p and grad are diff
    }
  } else {
    for (; sample < batchsize; sample += num_threads) {
      float sum = 0.f;
      for (size_t j = 0; j < dim; j++) {
        sum += t[sample * dim + j];
      }
      for (size_t j = 0, offset = sample * dim; j < dim; j++, offset++) {
        grad[offset] -= t[offset] / sum;
      }
    }
  }
}

__global__ void KernelFloat2Half(const size_t n, const float *in, __half *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = __float2half_rn(in[i]);
  }
}

__global__ void KernelHalf2Float(const size_t n, const __half *in, float *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = __half2float(in[i]);
  }
}

//kernal used by the threshold based sparsification
__global__ void KernelSparsAbs(const size_t n, const float threshold, const float *in, float *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = fabs(in[i]) >= threshold ? in[i] : 0.0f;
  }
}

//kernal used by the threshold based sparsification
__global__ void KernelSparsIndex(const size_t n, const float *in, int *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] == 0.0f ? 0 : i + 1;
  }
}

//kernal used by the topK based sparsification
__global__ void KernelGenerateIndex(const size_t n, int *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    out[i] = i + 1;
  }
}

//cuda unary elementwise ops kernel template 
#define GenUnaryCudaKernel(fn,kernelfn,cudafn)                                \
  __global__ void kernelfn(const size_t n, const float *in, float *out) {     \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                \
         i += blockDim.x * gridDim.x) {                                       \
      out[i] = cudafn(in[i]);                                                 \
    }                                                                         \
  }                                                                           \
  void fn(const size_t n, const float *in, float *out, cudaStream_t s) {      \
    kernelfn <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);             \
  }

GenUnaryCudaKernel(cos,KernelCos,cosf);
GenUnaryCudaKernel(cosh,KernelCosh,coshf);
GenUnaryCudaKernel(acos,KernelAcos,acosf);
GenUnaryCudaKernel(acosh,KernelAcosh,acoshf);
GenUnaryCudaKernel(sin,KernelSin,sinf);
GenUnaryCudaKernel(sinh,KernelSinh,sinhf);
GenUnaryCudaKernel(asin,KernelAsin,asinf);
GenUnaryCudaKernel(asinh,KernelAsinh,asinhf);
GenUnaryCudaKernel(tan,KernelTan,tanf);
GenUnaryCudaKernel(tanh,KernelTanh,tanhf);
GenUnaryCudaKernel(atan,KernelAtan,atanf);
GenUnaryCudaKernel(atanh,KernelAtanh,atanhf);


// ********************************
// Functions call kernels
// ********************************

void float2half(const size_t n, const float *in, __half *out, cudaStream_t s) {
  KernelFloat2Half <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void half2float(const size_t n, const __half *in, float *out, cudaStream_t s) {
  KernelHalf2Float <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void sparsabs(const size_t n, const float threshold, const float *in, float *out, cudaStream_t s) {
  KernelSparsAbs <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, threshold, in, out);
}

void sparsindex(const size_t n, const float *in, int *out, cudaStream_t s) {
  KernelSparsIndex <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void generateindex(const size_t n, int *out, cudaStream_t s) {
  KernelGenerateIndex <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, out);
}

//used by the threshold based sparsification
void removezeroval(const size_t n, float *in, cudaStream_t s) {
  thrust::remove(thrust::cuda::par.on(s), in, in + n, float(0));
}

//used by the threshold based sparsification
void removezeroidx(const size_t n, int *in, cudaStream_t s, int *address) {
  thrust::remove(thrust::cuda::par.on(s), in, in + n, int(0));  
  int a = thrust::count(thrust::cuda::par.on(s), in, in + n, int(0));
  *address = n - a;
}

struct absgreater : public thrust::binary_function<float,float,bool>
{
  thrust::maximum<int> max;
  __host__ __device__ bool operator()(const float &lhs, const float &rhs) const {
     return max(lhs, -lhs) > max(rhs, -rhs);
  }
};

//used by the topK based sparsification
void sortbykey(const size_t n, float *key, int *value, cudaStream_t s) {
  thrust::sort_by_key(thrust::cuda::par.on(s), key, key + n, value, absgreater());
}

void set(const size_t n, const float v, float *out, cudaStream_t s) {
  KernelSet <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, v, out);
}

void abs(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelAbs <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void cast_float_2_int(const size_t n, const float *src, int *dst, cudaStream_t s) {
  KernelCastFloat2Int <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, src, dst);
}

void cast_int_2_float(const size_t n, const int *src, float *dst, cudaStream_t s) {
  KernelCastInt2Float <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, src, dst);
}

void sign(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelSign <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void exp(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelExp <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void ceil2(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelCeil2 <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void log(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelLog <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void sqrt(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelSqrt <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void square(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelSquare <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void relu(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelRelu <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}
void sigmoid(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelSigmoid <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void softplus(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelSoftplus <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, out);
}

void softsign(const size_t n, const float *in, float *out, cudaStream_t s) {
  KernelSoftsign <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF>>> (n, in, out);
}

void clamp(const size_t n, const float low, const float high, const float *in,
           float *out, cudaStream_t s) {
  KernelClamp <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, low, high, in, out);
}

void pow(const size_t n, const float *in, const float x, float *out,
         cudaStream_t s) {
  KernelPow <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, x, out);
}

void add(const size_t n, const float *in, const float x, float *out,
         cudaStream_t s) {
  KernelAdd <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, x, out);
}

void broadcast_to(const size_t n, size_t nDim,const float *in,const float* shape, const float* stride, float *out, cudaStream_t s) {
  KernelBroadcastTo <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF>>> (n, nDim, in, shape, stride, out);
}

void mult(const size_t n, const float *in, const float x, float *out,
          cudaStream_t s) {
  KernelMult <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in, x, out);
}

void div(const size_t n, const float x, const float *in, float *out,
          cudaStream_t s) {
  KernelDiv <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, x, in, out);
}

void threshold(const size_t n, const float x, const float *in, float *out,
               cudaStream_t s) {
  KernelThreshold <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, x, in, out);
}

void relubackward(const size_t num, const float *in1, const float *in2, float *out,
        cudaStream_t s) {
  KernelReLUBackward <<<ceil(num / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (num, in1, in2, out);
}

void gt(const size_t num, const float *in, const float x, float *out,
        cudaStream_t s) {
  KernelGT <<<ceil(num / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (num, in, x, out);
}
void gt(const size_t num, const float *in1, const float *in2, float *out,
        cudaStream_t s) {
  KernelBGT <<<ceil(num / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (num, in1, in2, out);
}
void ge(const size_t num, const float *in, const float x, float *out,
        cudaStream_t s) {
  KernelGE <<<ceil(num / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (num, in, x, out);
}
void ge(const size_t num, const float *in1, const float *in2, float *out,
        cudaStream_t s) {
  KernelBGE <<<ceil(num / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (num, in1, in2, out);
}
void lt(const size_t num, const float *in, const float x, float *out,
        cudaStream_t s) {
  KernelLT <<<ceil(num / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (num, in, x, out);
}
void lt(const size_t num, const float *in1, const float *in2, float *out,
        cudaStream_t s) {
  KernelBLT <<<ceil(num / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (num, in1, in2, out);
}
void le(const size_t num, const float *in, const float x, float *out,
        cudaStream_t s) {
  KernelLE <<<ceil(num / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (num, in, x, out);
}
void le(const size_t num, const float *in1, const float *in2, float *out,
        cudaStream_t s) {
  KernelBLE <<<ceil(num / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (num, in1, in2, out);
}
void pow(const size_t n, const float *in1, const float *in2, float *out,
         cudaStream_t s) {
  KernelPow <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in1, in2, out);
}

void add(const size_t n, const float *in1, const float *in2, float *out,
         cudaStream_t s) {
  KernelAdd <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in1, in2, out);
}

void sub(const size_t n, const float *in1, const float *in2, float *out,
         cudaStream_t s) {
  KernelSub <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in1, in2, out);
}

void mult(const size_t n, const float *in1, const float *in2, float *out,
          cudaStream_t s) {
  KernelMult <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in1, in2, out);
}

void div(const size_t n, const float *in1, const float *in2, float *out,
         cudaStream_t s) {
  KernelDiv <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (n, in1, in2, out);
}

/*
void sum(const size_t n, const float *in, float *out, cudaStream_t s) {
  int threads_per_block = n > CU1DBLOCK ? CU1DBLOCK : n;
  //  here, we only need one block
  int num_blocks = 1;
  KernelSum <<<num_blocks, threads_per_block>>> (n, in, out);
}
*/

void ComputeCrossEntropy(const bool int_target, size_t batchsize, const size_t dim, const float *p,
                         const int *t, float *loss, cudaStream_t stream) {
  KernelComputeCrossEntropy <<<ceil(batchsize / CU1DBLOCKF), CU1DBLOCKF, 0, stream>>>
      (int_target, batchsize, dim, p, t, loss);
}

void SoftmaxCrossEntropyBwd(const bool int_target, size_t batchsize, const size_t dim, const float *p,
                            const int *t, float *grad, cudaStream_t stream) {
  KernelSoftmaxCrossEntropyBwd <<<ceil(batchsize / CU1DBLOCKF), CU1DBLOCKF, 0, stream>>>
      (int_target, batchsize, dim, p, t, grad);
}

void RowMax(const size_t nrow, const size_t ncol, const float *inPtr,
    float *outPtr, cudaStream_t stream) {
  KernelRowMax <<<ceil(nrow / CU1DBLOCKF), CU1DBLOCKF, 0, stream>>>(nrow, ncol, inPtr, outPtr);
}

/*
void square_grad(int n, const float *in, float *out, cudaStream_t s) {
  kernel_square_grad <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (in, out, n);
}

void tanh_grad(int n, const float *in, float *out, cudaStream_t s) {
  kernel_tanh_grad <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (in, out, n);
}


void relu_grad(int n, const float *in, float *out, cudaStream_t s) {
  kernel_relu_grad <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (in, out, n);
}


void sigmoid_grad(int n, const float *in, float *out, cudaStream_t s) {
  kernel_sigmoid_grad <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (in, out, n);
}

void softplus_grad(int n, const float *in, float *out, cudaStream_t s) {
  kernel_softplus_grad <<<ceil(n / CU1DBLOCKF), CU1DBLOCKF, 0, s>>> (in, out, n);
}


__global__ void kernel_sum_col(const float *src_mat_data, float *dst_vec_data,
                               int rows, int cols, int stride) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < rows; index += num_threads) {
    dst_vec_data[index] = 0.0f;
    for (int k = 0; k < cols; k++) {
      dst_vec_data[index] += src_mat_data[index * stride + k];
    }
  }
}

__global__ void kernel_sum_row(const float *src_mat_data, float *dst_vec_data,
                               int rows, int cols, int stride) {
  int j = blockIdx.x;
  int THREADS = blockDim.x;
  if (j >= cols) {
    return;
  }

  __shared__ float aux[CU1DBLOCK];
  int steps = (rows - 1) / THREADS + 1;
  aux[threadIdx.x] = src_mat_data[j + threadIdx.x * stride];
  for (int i = 1; i < steps; ++i) {
    if (threadIdx.x + i * THREADS < rows) {
      aux[threadIdx.x] +=
          src_mat_data[j + (threadIdx.x + i * THREADS) * stride];
    }
  }

  int total_threads = THREADS;
  __syncthreads();
  while (total_threads > 1) {
    int half_point = ((1 + total_threads) >> 1);
    if (threadIdx.x < half_point) {
      if (threadIdx.x + half_point < total_threads) {
        aux[threadIdx.x] += aux[threadIdx.x + half_point];
      }
    }
    __syncthreads();
    total_threads = ((total_threads + 1) >> 1);
  }

  __syncthreads();
  dst_vec_data[j] = aux[0];
}


__global__ void kernel_add_vec_row(const float *src_vec_data,
                                   const float *src_mat_data,
                                   float *des_mat_data, int rows, int cols,
                                   int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int num_threads_x = blockDim.x * gridDim.x;
  int num_threads_y = blockDim.y * gridDim.y;
  int index = 0;
  for (; i < cols && j < rows; i += num_threads_x, j += num_threads_y) {
    index = j * stride + i;
    des_mat_data[index] = src_mat_data[index] + src_vec_data[i];
  }
}

__global__ void kernel_sigmoid_grad(const float *src_data, float *des_data,
                                    int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = src_data[index] * (1.0f - src_data[index]);
  }
}


__global__ void kernel_relu_grad(const float *src_data, float *des_data,
                                 int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = src_data[index] > 0.0f ? 1.0f : 0.0f;
  }
}

__global__ void kernel_tanh_grad(const float *src_data, float *des_data,
                                 int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = (1.0f - src_data[index] * src_data[index]);
  }
}

__global__ void kernel_softplus_grad(const float *src_data, float *des_data,
                                     int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = 1.0f / (1.0f + expf(-src_data[index]));
  }
}
__global__ void KernelSquareGrad(const float *src_data, float *des_data,
                                   int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = 2 * src_data[index];
  }
}
__global__ void kernel_softmax_loss(const float *prob, const size_t *label,
                                    float *loss, int n, int dim) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    float prob_of_truth = prob[index * dim + label[index]];
    loss[index] -= std::log(max(prob_of_truth, FLT_MIN));
  }
}
__global__ void kernel_softmax_gradient(float *grad, const size_t *label, int n,
                                        int dim, float scale) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    int pos = index * dim + label[index];
    grad[pos] = (grad[pos] - 1.0f) * scale;
  }
}
*/


}  // namespace cuda
}  // namespace singa

#endif  // USE_CUDA
