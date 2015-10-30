extern "C"
{
   #include <cblas.h>
}

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "singa/blob/math_addr.h"
#include "singa/blob/singa_op.h"

namespace singa{

const float * cpu_uni_vec(const int n)
{
	float * res = new float[n];
	for(int i = 0; i < n; i++)
		res[i] = 1.0;
	return res;
}

void cpu_gemm(const float * A, const float * B, const int m, const int n, const int k, const float alpha, const float beta, const bool TranA, const bool TranB, float * C)
{
	int lda, ldb;
	CBLAS_TRANSPOSE tA, tB;
	lda = TranA ? m : k;
	ldb = TranB ? k : n;
	tA = TranA ? CblasTrans : CblasNoTrans;
	tB = TranB ? CblasTrans : CblasNoTrans;
	cblas_sgemm(CblasRowMajor, tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, n);
}

void cpu_gemv(const float * A, const float * B, const int m, const int n, const float alpha, const float beta, const bool TranA, float * C)
{
	CBLAS_TRANSPOSE tA;
	tA = TranA ? CblasTrans : CblasNoTrans;
	cblas_sgemv(CblasRowMajor, tA, m, n, alpha, A, n, B, 1, beta, C, 1);
}

void cpu_axpy(const float * A, const int n, const float alpha, float * B)
{
	cblas_saxpy(n, alpha, A, 1, B, 1);
}

float cpu_dot(const float * A, const float * B, const int n)
{
	float sum = 0;
	for(int i = 0 ; i < n ; i++)
		sum += A[i]*B[i];
	return sum;
}

//Trick: swap A and B
//
void gpu_gemm(const float * A, const float * B, const int m, const int n, const int k, const float alpha, const float beta, const bool TranA, const bool TranB, float * C)
{
    int lda = TranA ? m : k;
    int ldb = TranB ? k : n;
    int ldc = n;

    cublasOperation_t tA= (TranA==false) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t tB= (TranB==false) ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle, tB, tA, n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc);

    cublasDestroy(handle);
}

void gpu_gemv(const float * A, const float * B, const int m, const int n, const float alpha, const float beta, const bool TranA, float * C)
{
	int lda = n ;
	cublasOperation_t tA= (TranA==true) ? CUBLAS_OP_N : CUBLAS_OP_T;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemv(handle, tA , n , m ,&alpha , A , lda , B , 1 ,&beta , C , 1);

	cublasDestroy(handle);

}


void gpu_axpy(const float * A, const int n, const float alpha, float * B)
{

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSaxpy(handle,n,&alpha,A,1,B,1);

	cublasDestroy(handle);

}


float gpu_dot(const float * A, const float * B, const int n)
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	float result=0.0;

	cublasSdot(handle,n,A,1,B,1,&result);

	cublasDestroy(handle);

	return result;

}

} // namespace singa
