#ifndef MATH_ADDR_H
#define MATH_ADDR_H

namespace singa{

const float * cpu_uni_vec(const int n);

void cpu_gemm(const float * A, const float * B, const int m, const int n, const int k, const float alpha, const float beta, const bool TranA, const bool TranB, float * C);

void cpu_gemv(const float * A, const float * B, const int m, const int n, const float alpha, const float beta, const bool TranA, float * C);
// should be very careful : m is the length of B, and n is the length of C , A is a n*m matrix

void cpu_axpy(const float * A, const int n, const float alpha, float * B);

float cpu_dot(const float * A, const float * B, const int n);

/*
//element-wise
template<typename Op> void cpu_e_f(const int n, const float alpha, float * A);
template<typename Op> void cpu_e_f(const int n,const float * A,const float alpha, float * B);
template<typename Op> void cpu_e_f(const int n,const float * A,const float * B,const float alpha, const float beta,float * C);
// element-wise generalized operation defined in Op
*/

//element-wise
template<typename Op> void cpu_e_f(const int n, const float alpha, float * A)
{
                for(int i = 0 ; i < n ; i++)
                {
                                Op::Map(alpha, A[i]);
                }
}

template<typename Op> void cpu_e_f(const int n,const float * A,const float alpha, float * B)
{
                for(int i = 0 ; i < n ; i++)
                {
                                Op::Map(alpha, A[i], B[i]);
                }
}

template<typename Op> void cpu_e_f(const int n,const float * A,const float * B,const float alpha, const float beta,float * C)
{
                for(int i = 0 ; i < n ; i++)
                {
                                Op::Map(alpha, beta, A[i], B[i], C[i]);
                }
}
// element-wise generalized operation defined in Op

/*
//matrix/vector expand/reduce

template<typename Op> void cpu_reduce_f(const float * A,const int m, const int n, float * B);
//reduce each row of A to an element of B e.g. the sum operation in softmax
template<typename Op> void cpu_expand_f(const float * A,const int m, const int n, float * B);
//expand each element in A into a row of B
*/

//matrix/vector expand/reduce

template<typename Op> void cpu_reduce_f(const float * A,const int m, const int n, float * B)
{
                for(int i = 0 ; i < m ; i++)
                {
                                Op::Map(A+i*n, n, B[i]);
                }
}
//reduce each row of A to an element of B e.g. the sum operation in softmax
template<typename Op> void cpu_expand_f(const float * A,const int m, const int n, float * B)
{
                for(int i = 0 ; i < m ; i++)
                {
                                Op::Map(A[i], n, B+i*n);
                }
}
//expand each element in A into a row of B

void gpu_gemm(const float * A, const float * B, const int m, const int n, const int k, const float alpha, const float beta, const bool TranA, const bool TranB, float * C);
void gpu_gemv(const float * A, const float * B, const int m, const int n, const float alpha, const float beta, const bool TranA, float * C);
void gpu_axpy(const float * A, const int n, const float alpha, float * B);
float gpu_dot(const float * A, const float * B, const int n);

//element-wise
template<typename Op> void gpu_e_f(const int n, const float alpha, float * A)
{
	Op::CudaMap(alpha, A, n);
}

template<typename Op> void gpu_e_f(const int n,const float * A,const float alpha, float * B)
{
	Op::CudaMap(alpha, A, B, n);
}

template<typename Op> void gpu_e_f(const int n,const float * A,const float * B,const float alpha, const float beta,float * C)
{
	Op::CudaMap(alpha, beta, A, B, C, n);
}
// element-wise generalized operation defined in Op

//matrix/vector expand/reduce

template<typename Op> void gpu_reduce_f(const float * A,const int m, const int n, float * B)
{
                for(int i = 0 ; i < m ; i++)
                {
                                Op::CudaMap(A+i*n, n, B[i]);
                }
}
//reduce each row of A to an element of B e.g. the sum operation in softmax
template<typename Op> void gpu_expand_f(const float * A,const int m, const int n, float * B)
{
                for(int i = 0 ; i < m ; i++)
                {
                                Op::CudaMap(A[i], n, B+i*n);
                }
}
//expand each element in A into a row of B


}  // namespace singa
#endif // MATH_ADDR_H
