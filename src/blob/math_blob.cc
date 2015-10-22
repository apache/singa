#include "singa/blob/math_blob.h"

namespace singa {

/**********************************************************************************/
// shape_check function

int get_size(const std::vector<int>& shape)
{
  int sum = 1;
  for(unsigned int i = 0; i < shape.size(); i++) sum *= shape[i];
  return sum; 
}

/**********************************************************************************/
// class1 matrix operation


void GEMM(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C, float alpha, float beta)
{
	if(xpu == cpu)
	{
	  if(check_shape_mmm(A, B, *C))
	  {
	    int m = C->shape().at(0);
	    int n = C->shape().at(1);
	    int k = A.isTranspose() ? A.shape().at(0) : A.shape().at(1);
	    bool TranA = A.isTranspose();
	    bool TranB = B.isTranspose();
	    cpu_gemm(A.cpu_data(), B.cpu_data(), m, n, k, alpha, beta, TranA, TranB, C->mutable_cpu_data());
	  }
	  else{
	  // report errors here
	  }
	}
	if(xpu == gpu)
	{
	  //gpu part
	}
}
//C = alpha*A*B+beta*C, A, B and C are matrix

 
void MMDot(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C)
{
  	GEMM(xpu, A, B, C, 1, 0);
}
// A,B and C are matrix


void MVDot(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C)
{
	if(xpu == cpu)
	{
		if(check_shape_mvv(A, B, *C))
		{
			int m = B.shape().at(0);
			int n = C->shape().at(0);
			bool TranA = A.isTranspose();
			cpu_gemv(A.cpu_data(), B.cpu_data(), m, n, 1, 0, TranA, C->mutable_cpu_data());
		}
		else{
			// report errors here
		}
	}
	if(xpu == gpu)
	{
	  //gpu part
	}
	
}
// A is matrix,B and C are vector

 
void VVDot(XPU xpu, const Blob<float> & A, const Blob<float> & B, Blob<float> * C)
{
	if(xpu == cpu)
	{
		if(check_shape_vvm(A, B, *C))
		{
			int m = C->shape().at(0);
			int n = C->shape().at(1);
			cpu_gemm(A.cpu_data(), B.cpu_data(), m, n, 1, 1, 0, false, false, C->mutable_cpu_data());
		}
		else{
		// report errors here
		}
	}
	if(xpu == gpu)
	{
	  //gpu part
	}
}
// C is matrix,A and B are vector

 
float VVdot(XPU xpu, const Blob<float> & A, const Blob<float> & B)
{
	float res = 0;
	if(xpu == cpu)
	{
		if(check_shape_equal(A, B, B))
		{
			int n = get_size(A.shape());
			res = cpu_dot(A.cpu_data(), B.cpu_data(), n);
		}
		else{
		// report errors here
		}
	}
	if(xpu == gpu)
	{
	  //gpu part
	}
	return res;
}
//A and B are vectors

void AXPY(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha)
{
	if(xpu == cpu)
	{
		if(check_shape_equal(A, *B, *B))
		{
			cpu_axpy(A.cpu_data(), get_size(A.shape()), alpha, B->mutable_cpu_data());
		}
		else{
		// report errors here
		}
	}
	if(xpu == gpu)
	{
	  //gpu part
	}
}
// element-wise operation: Bi = alpha*Ai+Bi  A and B should have the same size

inline void Repmat(XPU xpu, const Blob<float> & A, Blob<float> * B)
{
	MVAdd(xpu, A, B, 1, 0);
}
// A is a vector, B is a matrix , let each row of B to be A

void MVAdd(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha, float beta)
{
	if(xpu == cpu)
	{
		if(check_shape_mv(*B, A))
		{
			int m = get_size(A.shape());
			int n = get_size(B->shape()) / m;
			const float * univ = cpu_uni_vec(n);
			cpu_gemm(A.cpu_data(), univ, m, n, 1, alpha, beta, false, false, B->mutable_cpu_data());
			delete univ;
		}
		else{
		// report errors here
		}
	}
	if(xpu == gpu)
	{
	  //gpu part
	}	
}
// A is a vector, B is a matrix , Bij = alpha*Ai+beta*Bij
// will use gemm. faster than general expand_f

void MVSum(XPU xpu, const Blob<float> & A, Blob<float> * B, float alpha, float beta)
{
	if(xpu == cpu)
	{
		if(check_shape_mv(A, *B))
		{
			int m = get_size(B->shape());
			int n = get_size(A.shape()) / m;
			const float * univ = cpu_uni_vec(n);
			cpu_gemm(A.cpu_data(), univ, m, 1, n, alpha, beta, false, false, B->mutable_cpu_data());
			delete univ;
		}
		else{
		// report errors here
		}
	}
	if(xpu == gpu)
	{
	  //gpu part
	}
}
// B is a vector, A is a matrix , Bi = \sigma_j_{alpha*Aij}+beta*Bi
// will use gemm. faster than general reduce_f

} // namespace singa

