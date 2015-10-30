#include "singa/blob/math_kernel.h"

#define CU2DBLOCK_X 32
#define CU2DBLOCK_Y 32

#define CU1DBLOCK 1024 
#define CU1DBLOCKF 1024.0 


//Cuda Kernel Functions
__global__
void kernel_sum_col(float *src_mat_data, float *dst_vec_data, long rows, long cols, long stride)
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
void kernel_add_vec_row(float *src_vec_data, float *src_mat_data, float* des_mat_data,long rows, long cols, long stride)
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

//
namespace singa{

void singa_sum_col(float *src_mat_data, float *dst_vec_data, long rows, long cols, long stride)
{
	long threads_per_block = rows > CU1DBLOCK ? CU1DBLOCK : rows;
	long num_blocks = cols;

	kernel_sum_col<<<num_blocks, threads_per_block>>>(src_mat_data, dst_vec_data, rows, cols, stride);
}

void singa_add_vec_row(float *src_vec_data, float *src_mat_data, float *des_mat_data ,long rows, long cols, long stride)
{
	dim3 threads_per_block(CU2DBLOCK_X, CU2DBLOCK_Y);
	dim3 num_blocks(cols/threads_per_block.x + (cols%threads_per_block.x == 0 ? 0 : 1), rows/threads_per_block.y + (rows%threads_per_block.y == 0 ? 0 : 1));
	kernel_add_vec_row<<<num_blocks, threads_per_block>>>(src_vec_data, src_mat_data, des_mat_data,rows, cols, stride);
}

}//namespace singa
