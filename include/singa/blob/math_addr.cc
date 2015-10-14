extern "C"
{
   #include <cblas.h>
}

#include "math_addr.h"
#include "singa_op.h"

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



