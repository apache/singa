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
#include <cmath>
#include <algorithm>
#include "singa/utils/math_kernel.h"
#include "mshadow/tensor.h"  // FLT_MIN?

#define CU2DBLOCK_X 32
#define CU2DBLOCK_Y 32

#define CU1DBLOCK 1024
#define CU1DBLOCKF 1024.0

// Cuda Kernel Functions

__global__
void kernel_softmax_loss(const float *prob, const int *label , float *loss,
    int n, int dim) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    float prob_of_truth = prob[index * dim + label[index]];
    loss[index] -= log(max(prob_of_truth, FLT_MIN));
  }
}

__global__
void kernel_softmax_gradient(float *grad, const int *label ,
    int n, int dim, float scale) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    int pos = index * dim + label[index];
    grad[pos] = (grad[pos] - 1.0f) * scale;
  }
}

__global__
void kernel_sum_vec(float *data, float *sum , int n) {
  int THREADS = blockDim.x;

  __shared__ float aux[CU1DBLOCK];
  int steps = (n - 1) / THREADS + 1;
  aux[threadIdx.x] = data[threadIdx.x];

  for (int i = 1; i < steps; ++i) {
    if (threadIdx.x + i * THREADS < n) {
      aux[threadIdx.x] += data[threadIdx.x+i*THREADS];
    }
  }

  int total_threads = THREADS;
  __syncthreads();

  while (total_threads > 1) {
    int half_point = ((1+total_threads) >> 1);
    if (threadIdx.x < half_point) {
      if (threadIdx.x+half_point < total_threads) {
        aux[threadIdx.x] += aux[threadIdx.x + half_point];
      }
    }
    __syncthreads();
    total_threads = ((total_threads+1) >> 1);
  }

  __syncthreads();
  *sum = aux[0];
}

__global__
void kernel_sum_col(const float *src_mat_data,
    float *dst_vec_data, int rows, int cols, int stride) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < rows; index += num_threads) {
    dst_vec_data[index] = 0.0f;
    for (int k = 0; k < cols; k++) {
      dst_vec_data[index] += src_mat_data[index * stride + k];
    }
  }
}

__global__
void kernel_sum_row(const float *src_mat_data,
    float *dst_vec_data, int rows, int cols, int stride) {
  int j = blockIdx.x;
  int THREADS = blockDim.x;
  if (j >= cols) {
    return;
  }

  __shared__ float aux[CU1DBLOCK];
  int steps = (rows - 1) / THREADS + 1;
  aux[threadIdx.x] = src_mat_data[j+threadIdx.x*stride];
  for (int i = 1; i < steps; ++i) {
    if (threadIdx.x+i*THREADS < rows) {
      aux[threadIdx.x] += src_mat_data[j+(threadIdx.x+i*THREADS)*stride];
    }
  }

  int total_threads = THREADS;
  __syncthreads();
  while (total_threads > 1) {
    int half_point = ((1+total_threads) >> 1);
    if (threadIdx.x < half_point) {
      if (threadIdx.x+half_point < total_threads) {
        aux[threadIdx.x] += aux[threadIdx.x + half_point];
      }
    }
    __syncthreads();
    total_threads = ((total_threads+1) >> 1);
  }

  __syncthreads();
  dst_vec_data[j] = aux[0];
}

__global__
void kernel_add_vec_row(const float *src_vec_data, const float *src_mat_data,
    float* des_mat_data, int rows, int cols, int stride) {
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

__global__
void kernel_exp(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = exp(src_data[index]);
  }
}

__global__
void kernel_log(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = log(src_data[index]);
  }
}

__global__
void kernel_sigmoid(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = 1.0f / (1.0f + expf(-src_data[index]));
  }
}

__global__
void kernel_sigmoid_grad(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = src_data[index] * (1.0f - src_data[index]);
  }
}

__global__
void kernel_relu(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = max(src_data[index], 0.0f);
  }
}

__global__
void kernel_relu_grad(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = src_data[index] > 0.0f ? 1.0f : 0.0f;
  }
}

__global__
void kernel_tanh(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = tanhf(src_data[index]);
  }
}

__global__
void kernel_tanh_grad(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = (1.0f - src_data[index] * src_data[index]);
  }
}

__global__
void kernel_softplus(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = logf(1 + expf(src_data[index]));
  }
}

__global__
void kernel_softplus_grad(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = 1.0f / (1.0f + expf(-src_data[index]));
  }
}

__global__
void kernel_square(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = src_data[index] * src_data[index];
  }
}

__global__
void kernel_square_grad(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = 2 * sqrt(src_data[index]);
  }
}

__global__
void kernel_sqrt(const float *src_data, float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = sqrt(src_data[index]);
  }
}

__global__
void kernel_pow(const float *src_data_a, const float *src_data_b,
    float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = pow(src_data_a[index], src_data_b[index]);
  }
}

__global__
void kernel_mult(const float *src_data_a, const float *src_data_b,
    float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = src_data_a[index] * src_data_b[index];
  }
}

__global__
void kernel_div(const float *src_data_a, const float *src_data_b,
    float *des_data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = src_data_a[index] / src_data_b[index];
  }
}

__global__ static
void kernel_set_value(float *data, float value, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    data[index] = value;
  }
}

__global__
void kernel_threshold(const float *src_data, float *des_data,
    float alpha, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (; index < n; index += num_threads) {
    des_data[index] = src_data[index] < alpha ? 1.0f : 0.0f;
  }
}

//
namespace singa {

void singa_gpu_softmaxloss_forward(int n, int dim, const float *prob,
    const int *label, float *loss) {
  kernel_softmax_loss<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(prob, label, loss, n,
      dim);
}

void singa_gpu_softmaxloss_backward(int n, int dim, float scale,
    const int *label, float *grad) {
  kernel_softmax_gradient<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(grad, label, n,
      dim, scale);
}

void singa_gpu_sum_vec(float *data, float *sum , int n) {
  int threads_per_block = n > CU1DBLOCK ? CU1DBLOCK : n;
  //  here, we only need one block
  int num_blocks = 1;

  kernel_sum_vec<<<num_blocks, threads_per_block>>>(data, sum, n);
}

void singa_gpu_sum_row(const float *src_mat_data, float *dst_vec_data,
    int rows, int cols, int stride) {
  int threads_per_block = rows > CU1DBLOCK ? CU1DBLOCK : rows;
  int num_blocks = cols;

  kernel_sum_row<<<num_blocks, threads_per_block>>>(src_mat_data,
      dst_vec_data, rows, cols, stride);
}

void singa_gpu_sum_col(const float *src_mat_data, float *dst_vec_data,
    int rows, int cols, int stride) {
  int threads_per_block = cols > CU1DBLOCK ? CU1DBLOCK : cols;
  int num_blocks = rows;

  kernel_sum_col<<<num_blocks, threads_per_block>>>(src_mat_data,
      dst_vec_data, rows, cols, stride);
}

void singa_gpu_add_vec_row(const float *src_vec_data, const float *src_mat_data,
    float *des_mat_data , int rows, int cols, int stride) {
  dim3 threads_per_block(CU2DBLOCK_X, CU2DBLOCK_Y);
  dim3 num_blocks(cols/threads_per_block.x +
    (cols%threads_per_block.x == 0 ? 0 : 1),
    rows/threads_per_block.y + (rows%threads_per_block.y == 0 ? 0 : 1));
  kernel_add_vec_row<<<num_blocks, threads_per_block>>>
    (src_vec_data, src_mat_data, des_mat_data, rows, cols, stride);
}

void singa_gpu_exp(const float *src_data, float *des_data, int n) {
  kernel_exp<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_log(const float *src_data, float *des_data, int n) {
  kernel_log<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_sigmoid(const float *src_data, float *des_data, int n) {
  kernel_sigmoid<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_sigmoid_grad(const float *src_data, float *des_data,
    int n) {
  kernel_sigmoid_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>
    (src_data, des_data, n);
}

void singa_gpu_relu(const float *src_data, float *des_data, int n) {
  kernel_relu<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_relu_grad(const float *src_data, float *des_data, int n) {
  kernel_relu_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_tanh(const float *src_data, float *des_data, int n) {
  kernel_tanh<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_tanh_grad(const float *src_data, float *des_data, int n) {
  kernel_tanh_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_softplus(const float *src_data, float *des_data, int n) {
  kernel_softplus<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_softplus_grad(const float *src_data, float *des_data, int n) {
  kernel_softplus_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>
    (src_data, des_data, n);
}

void singa_gpu_square(const float *src_data, float *des_data, int n) {
  kernel_square<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_square_grad(const float *src_data, float *des_data, int n) {
  kernel_square_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_sqrt(const float *src_data, float *des_data, int n) {
  kernel_sqrt<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, n);
}

void singa_gpu_pow(const float *src_data_a, const float *src_data_b,
    float *des_data, int n) {
  kernel_pow<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>
    (src_data_a, src_data_b, des_data, n);
}

void singa_gpu_mult(const float *src_data_a, const float *src_data_b,
    float *des_data, int n) {
  kernel_mult<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>
    (src_data_a, src_data_b, des_data, n);
}

void singa_gpu_div(const float *src_data_a, const float *src_data_b,
    float *des_data, int n) {
  kernel_div<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>
    (src_data_a, src_data_b, des_data, n);
}

void singa_gpu_set_value(float *data, float value, int n) {
  kernel_set_value<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(data, value, n);
}

void singa_gpu_threshold(const float *src_data, float *des_data,
    float alpha, int n) {
  kernel_threshold<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>
    (src_data, des_data, alpha, n);
}

}  // namespace singa
