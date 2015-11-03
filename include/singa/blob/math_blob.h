#ifndef MATH_BLOB_H
#define MATH_BLOB_H

#include <vector>
#include "singa/utils/blob.h"
#include "singa/blob/singa_op.h"
#include "singa/blob/math_addr.h"


namespace singa{
/*********************Level-2 interface, called by user code*******************/
// c++ ususally use const & for input arguments, and * for output arguments.
// ww: maybe we represent Blob's shape using int s[4]+dim? currently we use a vector, which may
// not be convenient as int array.


int get_size(const std::vector<int>& shape);

template <typename Dtype>
bool check_shape_mv(const Blob<Dtype> & A, const Blob<Dtype> & B)
{
	if(A.shape().size() != 2) return false;
	if(B.shape().size() != 1) return false;
	if(A.shape().at(0) != B.shape().at(0)) return false;
	return true;
}

template <typename Dtype>
bool check_shape_equal(const Blob<Dtype> & A, const Blob<Dtype> & B, const Blob<Dtype> & C)
{
  int asize, bsize, csize;
  asize = get_size(A.shape());
  bsize = get_size(B.shape());
  csize = get_size(C.shape());
  if(asize != bsize) return false;
  if(asize != csize) return false;
  return true;
}

template <typename Dtype>
bool check_shape_mmm(const Blob<Dtype> & A, const Blob<Dtype> & B, const Blob<Dtype> & C)
{
  if(A.shape().size() != 2) return false;
  if(B.shape().size() != 2) return false;
  if(C.shape().size() != 2) return false;
  int a1, a2, b1, b2, c1, c2;
  if(C.isTranspose()) return false;
  a1 = A.isTranspose() ? A.shape().at(1) : A.shape().at(0);
  a2 = A.isTranspose() ? A.shape().at(0) : A.shape().at(1);
  b1 = B.isTranspose() ? B.shape().at(1) : B.shape().at(0);
  b2 = B.isTranspose() ? B.shape().at(0) : B.shape().at(1);
  c1 = C.shape().at(0);
  c2 = C.shape().at(1);
  if(a2 != b1) return false;
  if(a1 != c1) return false;
  if(b2 != c2) return false;
  return true;
}

template <typename Dtype>
bool check_shape_vvm(const Blob<Dtype> & A, const Blob<Dtype> & B, const Blob<Dtype> & C)
{
  if(A.shape().size() != 1) return false;
  if(B.shape().size() != 1) return false;
  if(C.shape().size() != 2) return false;
  int a1, b1, c1, c2;
  if(C.isTranspose()) return false;
  a1 = A.shape().at(0);
  b1 = B.shape().at(0);
  c1 = C.shape().at(0);
  c2 = C.shape().at(1);
  if(a1 != c2) return false;
  if(b1 != c1) return false;
  return true;
}

template <typename Dtype>
bool check_shape_mvv(const Blob<Dtype> & A, const Blob<Dtype> & B, const Blob<Dtype> & C)
{
  if(A.shape().size() != 2) return false;
  if(B.shape().size() != 1) return false;
  if(C.shape().size() != 1) return false;
  int a1, a2, b1, c1;
  a1 = A.isTranspose() ? A.shape().at(1) : A.shape().at(0);
  a2 = A.isTranspose() ? A.shape().at(0) : A.shape().at(1);
  b1 = B.shape().at(0);
  c1 = C.shape().at(0);
  if(a2 != b1) return false;
  if(a1 != c1) return false;
  return true;
}

/**********************************************************************************/
// blob transformation

template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, const std::vector<int>& shape)
{
  Blob<Dtype>* res = new Blob<Dtype>();
  res->Mirror(A);
  res->Reshape(shape);
  return res;
}

// the current reshape in blob.h is: void Reshape(const std::vector<int>& shape);

template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, int dim1)
{
	std::vector<int> tmpshape;
	tmpshape.push_back(dim1);
	return Reshape(A, tmpshape);
}

template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, int dim1, int dim2)
{
	std::vector<int> tmpshape;
	tmpshape.push_back(dim1);
	tmpshape.push_back(dim2);;
	return Reshape(A, tmpshape);
}

template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, int dim1, int dim2, int dim3)
{
	std::vector<int> tmpshape;
	tmpshape.push_back(dim1);
	tmpshape.push_back(dim2);
	tmpshape.push_back(dim3);
	return Reshape(A, tmpshape);
}

template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, int dim1, int dim2, int dim3, int dim4)
{
	std::vector<int> tmpshape;
	tmpshape.push_back(dim1);
	tmpshape.push_back(dim2);
	tmpshape.push_back(dim3);
	tmpshape.push_back(dim4);
	return Reshape(A, tmpshape);
}

template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, int dim1, int dim2, int dim3, int dim4, int dim5)
{
	std::vector<int> tmpshape;
	tmpshape.push_back(dim1);
	tmpshape.push_back(dim2);
	tmpshape.push_back(dim3);
	tmpshape.push_back(dim4);
	tmpshape.push_back(dim5);
	return Reshape(A, tmpshape);
}

template <typename Dtype>
Blob<Dtype>* Transpose(const Blob<Dtype> & A)
{
	Blob<Dtype>* res = new Blob<Dtype>();
	res->Mirror(A);
	res->setTranspose();
	return res;
}
// return A^T


/**********************************************************************************/
// class1 matrix operation


void MMDot(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C);
// A,B and C are matrix


void MVDot(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C);
// A is matrix,B and C are vector


void VVDot(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C);
// C is matrix,A and B are vector


float VVdot(XPU xpu, const Blob<float> & A, const Blob<float> & B);
//A and B are vectors


void GEMM(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C, float alpha = 1, float beta = 1);
//C = alpha*A*B+beta*C, A, B and C are matrix



/**********************************************************************************/
// class2 element-wise operation

// element-wise generalized operation defined in Op


template<typename Op> 
void E_Func(XPU xpu, Blob<float> * A, float alpha)
{
	if(xpu == cpu)
	{
		int n = get_size(A->shape());
		cpu_e_f<Op>(n, alpha, A->mutable_cpu_data());
	}
	if(xpu == gpu)
	{
		//gpu part
		int n = get_size(A->shape());
		gpu_e_f<Op>(n, alpha, A->mutable_gpu_data());
	}
}

template<typename Op>
void E_Func(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha)
{
	if(check_shape_equal(A, *B, *B))
	{
			int n = get_size(A.shape());
			if(xpu == cpu)
			{
				cpu_e_f<Op>(n, A.cpu_data(), alpha, B->mutable_cpu_data());
			}

			if(xpu == gpu)
			{
				//gpu part
				gpu_e_f<Op>(n, A.gpu_data(), alpha, B->mutable_gpu_data());
			}
	}
	else{
			// report errors here
	}	
}

template<typename Op>
void E_Func(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C, float alpha, float beta)
{
	if(check_shape_equal(A, B, *C))
	{
		int n = get_size(A.shape());

		if(xpu == cpu)
		{
			cpu_e_f<Op>(n, A.cpu_data(), B.cpu_data(), alpha, beta, C->mutable_cpu_data());
		}
		if(xpu == gpu)
		{
			//gpu part
			gpu_e_f<Op>(n, A.gpu_data(), B.gpu_data(), alpha, beta, C->mutable_gpu_data());
		}
	}
	else{
			// report errors here
	}
}


inline void Set(XPU xpu, Blob<float> * A,float alpha)
{
	E_Func<singa_op::Set>(xpu, A, alpha);
}
// element-wise operation: Ai = alpha


inline void Scale(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha)
{
	E_Func<singa_op::Scale>(xpu, A, B, alpha);
}
// element-wise operation: Bi = alpha*Ai

inline void Exp(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha = 2.71)
{
	E_Func<singa_op::Exp>(xpu, A, B, alpha);
}
// element-wise operation: Bi = alpha^Ai

inline void Exp_grad(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha = 2.71)
{
	E_Func<singa_op::Exp_grad>(xpu, A, B, alpha);
}
// element-wise operation: Bi = Ai*log(alpha)

inline void Gsigmoid(XPU xpu, const Blob<float> & A, Blob<float> * B,float alpha)
{
	E_Func<singa_op::Gsigmoid>(xpu, A, B, alpha);
}
// element-wise operation: b = 1.0f / (1.0f + expf(-a * alpha));

inline void Gsigmoid_grad(XPU xpu, const Blob<float> & A, Blob<float> * B,float alpha)
{
	E_Func<singa_op::Gsigmoid_grad>(xpu, A, B, alpha);
}
// element-wise operation: b = alpha * a * ( 1.0f - a );

inline void Grelu(XPU xpu, const Blob<float> & A, Blob<float> * B,float alpha = 0)
{
	E_Func<singa_op::Grelu>(xpu, A, B, alpha);
}
// element-wise operation: b = ( 1 - alpha ) * std::max( a, 0.0f ) + alpha * a;

inline void Grelu_grad(XPU xpu, const Blob<float> & A, Blob<float> * B,float alpha = 0)
{
	E_Func<singa_op::Grelu_grad>(xpu, A, B, alpha);
}
// element-wise operation: b = a > 0.0f ? 1.0f : alpha;

inline void Gtanh(XPU xpu, const Blob<float> & A, Blob<float> * B,float alpha)
{
	E_Func<singa_op::Gtanh>(xpu, A, B, alpha);
}
// element-wise operation: b = tanhf( a * alpha );

inline void Gtanh_grad(XPU xpu, const Blob<float> & A, Blob<float> * B,float alpha)
{
	E_Func<singa_op::Gtanh_grad>(xpu, A, B, alpha);
}
// element-wise operation: b = alpha * ( 1.0f - a * a );
        
inline void Softplus(XPU xpu, const Blob<float> & A, Blob<float> * B)
{
	E_Func<singa_op::Softplus>(xpu, A, B, 0);
}
// element-wise operation: b = logf(1 + expf(a));

inline void Softplus_grad(XPU xpu, const Blob<float> & A, Blob<float> * B)
{
	E_Func<singa_op::Softplus_grad>(xpu, A, B, 0);
}
// element-wise operation: b = 1.0f / (1.0f + expf(-a));

inline void Square(XPU xpu, const Blob<float> & A, Blob<float> * B)
{
	E_Func<singa_op::Square>(xpu, A, B, 0);
}
// element-wise operation: b = a * a;

inline void Square_grad(XPU xpu, const Blob<float> & A, Blob<float> * B)
{
	E_Func<singa_op::Square_grad>(xpu, A, B, 0);
}
// element-wise operation: b = 2 * sqrt(a);

inline void Sqrt(XPU xpu, const Blob<float> & A, Blob<float> * B)
{
	E_Func<singa_op::Sqrt>(xpu, A, B, 0);
}
// element-wise operation: b = sqrt(a);

inline void Threshold(XPU xpu, const Blob<float> & A, float alpha, Blob<float> * B)
{
	E_Func<singa_op::Threshold>(xpu, A, B, alpha);
}
// element-wise operation: b =  a < alpha ? 1.0f : 0.0f;

inline void Add(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C)
{
	E_Func<singa_op::Add>(xpu, A, B, C, 0, 0);
}
// element-wise operation: Ci = Ai+Bi  A,B and C should have the same size

inline void Sub(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C)
{
	E_Func<singa_op::Sub>(xpu, A, B, C, 0, 0);
}
// element-wise operation: Ci = Ai-Bi  A,B and C should have the same size

inline void Mult(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C)
{
	E_Func<singa_op::Mult>(xpu, A, B, C, 0, 0);
}
// element-wise operation: Ci = Ai*Bi  A,B and C should have the same size

inline void Div(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C)
{
	E_Func<singa_op::Div>(xpu, A, B, C, 0, 0);
}
// element-wise operation: Ci = Ai/Bi  A,B and C should have the same size


void AXPY(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha);
// element-wise operation: Bi = alpha*Ai+Bi  A and B should have the same size

//todo: random part
/*
void Gaussian(XPU xpu, Blob & A, float mu, float sigma);
// element-wise operation: initialize each element in A following distribution Gaussian(mu, sigma)

void Uniform(XPU xpu, Blob & A, float low, float high);
// element-wise operation: initialize each element in A following uniform distribution from low to high

void Bernoulli(XPU xpu, Blob & A, float p, int n = 1);
// element-wise operation: initialize each element in A following distribution Bernoulli(n,p)
*/

/**********************************************************************************/
//class3 matrix-vector expand/reduce operation

template<typename Op> 
void Reduce_F(XPU xpu, const Blob<float> & A, Blob<float> * B)
{
	if(check_shape_mv(A, *B))
	{
		int m = get_size(B->shape());
		int n = get_size(A.shape()) / m;

		if(xpu == cpu)
		{
			cpu_reduce_f<Op>(A.cpu_data(), m, n, B->mutable_cpu_data());
		}
		if(xpu == gpu)
		{
			//gpu part
			gpu_reduce_f<Op>(A.gpu_data(), m, n, B->mutable_gpu_data());
		}
	}
	else{
		// report errors here
	}
}
//reduce each row of A to an element of B e.g. the sum operation in softmax

template<typename Op> 
void Expand_F(XPU xpu, const Blob<float> & A, Blob<float> * B)
{
	if(check_shape_mv(*B, A))
	{
		int m = get_size(A.shape());
		int n = get_size(B->shape()) / m;

		if(xpu == cpu)
		{
			cpu_expand_f<Op>(A.cpu_data(), m, n, B->mutable_cpu_data());
		}
		if(xpu == gpu)
		{
			//gpu part
			gpu_expand_f<Op>(A.gpu_data(), m, n, B->mutable_gpu_data());
		}
	}
	else{
		// report errors here
	}
}
//expand each element in A into a row of B

void Repmat(XPU xpu, const Blob<float> & A, Blob<float> * B);
// A is a vector, B is a matrix , let each row of B to be A

void MVAdd(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha, float beta);
// A is a vector, B is a matrix , Bij = alpha*Ai+beta*Bij
// will use gemm. faster than general expand_f

void MVSum(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha, float beta);
// A is a vector, B is a matrix , Ai = \sigma_j_{alpha*Bij}+beta*Ai
// will use gemm. faster than general reduce_f


} // end of namespace singa

#endif // MATH_BLOB_H
