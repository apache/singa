#include <cmath>
#include "singa/utils/math_kernel.h"

#define CU2DBLOCK_X 32
#define CU2DBLOCK_Y 32

#define CU1DBLOCK 1024
#define CU1DBLOCKF 1024.0


//Cuda Kernel Functions

__global__
void kernel_sum_vec(float *data, float *sum , long n)
{
	int THREADS = blockDim.x;

	__shared__ float aux[CU1DBLOCK];
	int steps = (n - 1) / THREADS + 1;
	aux[threadIdx.x] = data[threadIdx.x];

	for(int i=1; i<steps; ++i) {
		if(threadIdx.x+i*THREADS < n) {
			aux[threadIdx.x] += data[threadIdx.x+i*THREADS];
		}
	}

	int total_threads = THREADS;
	__syncthreads();

	while(total_threads > 1) {
		int half_point = ((1+total_threads) >> 1);
		if (threadIdx.x < half_point) {
			if(threadIdx.x+half_point < total_threads) {
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
void kernel_sum_col(const float *src_mat_data, float *dst_vec_data, long rows, long cols, long stride)
{
	    int j = blockIdx.x;
		int THREADS = blockDim.x;
		if(j >= cols) {
			return;
		}

		__shared__ float aux[CU1DBLOCK];
		int steps = (rows - 1) / THREADS + 1;
		aux[threadIdx.x] = src_mat_data[j+threadIdx.x*stride];
		for(int i=1; i<steps; ++i) {
			if(threadIdx.x+i*THREADS < rows) {
				aux[threadIdx.x] += src_mat_data[j+(threadIdx.x+i*THREADS)*stride];
			}
		}

		int total_threads = THREADS;
		__syncthreads();
		while(total_threads > 1) {
			int half_point = ((1+total_threads) >> 1);
			if (threadIdx.x < half_point) {
				if(threadIdx.x+half_point < total_threads) {
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
void kernel_add_vec_row(const float *src_vec_data, const float *src_mat_data, float* des_mat_data,long rows, long cols, long stride)
{
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	long j = blockIdx.y * blockDim.y + threadIdx.y;
	long num_threads_x = blockDim.x * gridDim.x;
	long num_threads_y = blockDim.y * gridDim.y;
	long index = 0;
	for(; i<cols && j<rows; i+=num_threads_x, j+=num_threads_y) {
		index = j * stride + i;
		des_mat_data[index] = src_mat_data[index] + src_vec_data[i];
	}
}

__global__ static
void kernel_set_value(float *data, float value, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		data[index] = value;
	}
}

__global__
void kernel_scale(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data[index] * alpha;
	}
}

__global__
void kernel_scale_grad(float *data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		data[index] = alpha;
	}
}

__global__
void kernel_exp(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = pow(-src_data[index],alpha);
	}
}

__global__
void kernel_exp_grad(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data[index] * log(alpha);
	}
}

__global__
void kernel_sigmoid(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = 1.0f / (1.0f + expf(-src_data[index]) * alpha);
	}
}

__global__
void kernel_sigmoid_grad(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data[index] * (1.0f - src_data[index]) * alpha;
	}
}

__global__
void kernel_relu(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = 1.0f / ( 1 - alpha ) * max( src_data[index], 0.0f ) + alpha * src_data[index];
	}
}

__global__
void kernel_relu_grad(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data[index] > 0.0f ? 1.0f : alpha;
	}
}


__global__
void kernel_tanh(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = tanhf( src_data[index] * alpha );
	}
}

__global__
void kernel_tanh_grad(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = alpha * (1.0f - src_data[index] * src_data[index] );
	}
}

__global__
void kernel_softplus(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = logf(1 + expf(src_data[index]));
	}
}

__global__
void kernel_softplus_grad(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = 1.0f / (1.0f + expf(-src_data[index]));
	}
}

__global__
void kernel_square(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data[index] * src_data[index];
	}
}

__global__
void kernel_square_grad(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = 2 * sqrt(src_data[index]);
	}
}

__global__
void kernel_sqrt(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = sqrt(src_data[index]);
	}
}

__global__
void kernel_threshold(const float *src_data, float *des_data, float alpha, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data[index] < alpha ? 1.0f : 0.0f;
	}
}

__global__
void kernel_add(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data_a[index] + src_data_b[index];
	}
}

__global__
void kernel_sub(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data_a[index] - src_data_b[index];
	}
}

__global__
void kernel_mult(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data_a[index] * src_data_b[index];
	}
}

__global__
void kernel_div(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long num_threads = blockDim.x * gridDim.x;
	for(; index<n; index+=num_threads) {
		des_data[index] = src_data_a[index] / src_data_b[index];
	}
}

//
namespace singa{

void singa_gpu_sum_vec(float *data, float *sum , long n)
{
	long threads_per_block = n > CU1DBLOCK ? CU1DBLOCK : n;
	// here, we only need one block
	long num_blocks = 1;

	kernel_sum_vec<<<num_blocks, threads_per_block>>>(data, sum, n);
}

void singa_gpu_sum_col(const float *src_mat_data, float *dst_vec_data, long rows, long cols, long stride)
{
	long threads_per_block = rows > CU1DBLOCK ? CU1DBLOCK : rows;
	long num_blocks = cols;

	kernel_sum_col<<<num_blocks, threads_per_block>>>(src_mat_data, dst_vec_data, rows, cols, stride);
}

void singa_gpu_add_vec_row(const float *src_vec_data, const float *src_mat_data, float *des_mat_data ,long rows, long cols, long stride)
{
	dim3 threads_per_block(CU2DBLOCK_X, CU2DBLOCK_Y);
	dim3 num_blocks(cols/threads_per_block.x + (cols%threads_per_block.x == 0 ? 0 : 1), rows/threads_per_block.y + (rows%threads_per_block.y == 0 ? 0 : 1));
	kernel_add_vec_row<<<num_blocks, threads_per_block>>>(src_vec_data, src_mat_data, des_mat_data,rows, cols, stride);
}

void singa_gpu_set_value(float *data, float value, long n)
{
	kernel_set_value<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(data, value, n);
}

void singa_gpu_scale(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_scale<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_scale_grad(float *data, float alpha, long n)
{
	kernel_scale_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(data, alpha, n);
}

void singa_gpu_exp(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_exp<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_exp_grad(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_exp_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_sigmoid(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_sigmoid<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_sigmoid_grad(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_sigmoid_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_relu(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_relu<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_relu_grad(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_relu_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_tanh(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_tanh<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_tanh_grad(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_tanh_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_softplus(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_softplus<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_softplus_grad(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_softplus_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_square(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_square<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_square_grad(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_square_grad<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_sqrt(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_sqrt<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_threshold(const float *src_data, float *des_data, float alpha, long n)
{
	kernel_threshold<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data, des_data, alpha, n);
}

void singa_gpu_add(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n)
{
	kernel_add<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data_a, src_data_b, des_data, alpha, beta, n);
}

void singa_gpu_sub(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n)
{
	kernel_sub<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data_a, src_data_b, des_data, alpha, beta, n);
}

void singa_gpu_mult(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n)
{
	kernel_mult<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data_a, src_data_b, des_data, alpha, beta, n);
}

void singa_gpu_div(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n)
{
	kernel_div<<<ceil(n/CU1DBLOCKF), CU1DBLOCKF>>>(src_data_a, src_data_b, des_data, alpha, beta, n);
}


}//namespace singa_gpu
