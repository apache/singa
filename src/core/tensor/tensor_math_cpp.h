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
#ifndef SINGA_CORE_TENSOR_TENSOR_MATH_CPP_H_
#define SINGA_CORE_TENSOR_TENSOR_MATH_CPP_H_
#include "./tensor_math.h"
#include "singa/core/common.h"
#include <math.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

/// TODO(wangwei) Clean the implementations following the comments in
/// tensor_math.h.
namespace singa {

template<>
void Abs<float, lang::Cpp>(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = fabs(inPtr[i]);
  }
}

template <>
void Set<float, lang::Cpp>(const size_t num, const float x, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  for (size_t i = 0; i < num; i++) outPtr[i] = x;
}

// sum all elements of input into out
// TODO(wangwei) optimize using omp
template <>
void Sum<float, lang::Cpp>(const size_t num, const Blob *in, float *out, Context *ctx) {
  float s = 0.f;
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    s += inPtr[i];
  }
  *out = s;
}

template <>
void Sign<float, lang::Cpp>(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float*>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = inPtr[i] > 0 ? 1.0f : 0.0f; 
  }
}

template <>
void Exp<float, lang::Cpp>(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = exp(inPtr[i]);
  }
}

template <>
void Log<float, lang::Cpp>(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    CHECK_GT(inPtr[i], 0.f);
    outPtr[i] = log(inPtr[i]);
  }
}

template <>
void Sqrt<float, lang::Cpp>(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    CHECK_GT(inPtr[i], 0.f);
    outPtr[i] = sqrt(inPtr[i]);
  }
}

template <>
void Square<float, lang::Cpp>(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = inPtr[i] * inPtr[i];
  }
}

template <>
void Tanh<float, lang::Cpp>(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = tanh(inPtr[i]);
  }
}

template <>
void ReLU<float, lang::Cpp>(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = (inPtr[i] >= 0.f) ? inPtr[i] : 0.f;
  }
}

template <>
void Sigmoid<float, lang::Cpp>(const size_t num, const Blob *in, Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = 1.f / (1.f + exp(-inPtr[i]));
  }
}

template <>
void Softmax<float, lang::Cpp>(const size_t nrow, const size_t ncol, const Blob *in,
             Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
	float *bPtr = new float[ncol];
  for (size_t r = 0; r < nrow; r++) {
    size_t offset = r * ncol;
		float denom = 0.f;
    for (size_t c = 0; c < ncol; c++) {
			bPtr[c] = exp(inPtr[offset + c]);
			denom += bPtr[c];
    }
		for (size_t c = 0; c < ncol; c++) {
			size_t idx = offset + c;
			outPtr[idx] = bPtr[c] / denom;
		}
  }
	delete bPtr;
}

template <>
void SumRows<float, lang::Cpp>(const size_t nrow, const size_t ncol, const Blob *in,
             Blob *out, Context *ctx) {
	float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());                              
	for (size_t r = 0; r < nrow; r++) {
		size_t offset = r * ncol;
		outPtr[r] = 0.f;
		for (size_t c = 0; c < ncol; c++) {
			outPtr[r] += inPtr[offset + c];
		}
	}
}

template <>
void SumColumns<float, lang::Cpp>(const size_t nrow, const size_t ncol, const Blob *in, Blob *out, Context *ctx) {
	float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());                              
	for (size_t c = 0; c < ncol; c++) {
		outPtr[c] = 0.f;
	}
	for (size_t r = 0; r < nrow; r++) {
		size_t offset = r * ncol;
		for (size_t c = 0; c < ncol; c++) {
				outPtr[c] += inPtr[offset + c];
		}
	}
}

template <>
void AddRow<float, lang::Cpp>(const size_t nrow, const size_t ncol, const Blob *A, const Blob *v,
            Blob *out, Context *ctx) {
	float *outPtr = static_cast<float *>(out->mutable_data());
  const float *APtr = static_cast<const float *>(A->data());                              
  const float *vPtr = static_cast<const float *>(v->data());                              
	for (size_t r = 0; r < nrow; r++) {
		size_t offset = r * ncol;
		for (size_t c = 0; c < ncol; c++) {
			outPtr[offset + c] = APtr[offset + c] + vPtr[c];
		}
	}
}

template <>
void AddCol<float, lang::Cpp>(const size_t nrow, const size_t ncol, const Blob *A, const Blob *v,
            Blob *out, Context *ctx) {
	float *outPtr = static_cast<float *>(out->mutable_data());
  const float *APtr = static_cast<const float *>(A->data());                              
  const float *vPtr = static_cast<const float *>(v->data());                              
	for (size_t r = 0; r < nrow; r++) {
		size_t offset = r * ncol;
		for (size_t c = 0; c < ncol; c++) {
			outPtr[offset + c] = APtr[offset + c] + vPtr[r];
		}
	}
}

template <>
void Pow<float, lang::Cpp>(const size_t num, const Blob *in, const float x, Blob *out, Context *ctx) {
	float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());                              
	for (size_t i = 0; i < num; i++) {
    outPtr[i] = pow(inPtr[i], x);
  }
}

template <>
void Pow<float, lang::Cpp>(const size_t num, const Blob *in1, const Blob *in2,
                           Blob *out, Context *ctx) {
  float *outPtr= static_cast<float *>(out->mutable_data());
  const float *in1Ptr= static_cast<const float *>(in1->data());
  const float *in2Ptr = static_cast<const float *>(in2->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = pow(in1Ptr[i], in2Ptr[i]);
  }
}

template <>
void Clamp<float, lang::Cpp>(const size_t num, const float low, const float high, const Blob *in,
														 Blob *out, Context *ctx) {
	float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());                              
	for (size_t i = 0; i < num; i++) {
		if (inPtr[i] > high) {
			outPtr[i] = high;
		}
		else if (inPtr[i] < low) {
			outPtr[i] = low;
		}
		else {
			outPtr[i] = inPtr[i];			
		}
	}
}

template <>
void Add<float, lang::Cpp>(const size_t num, const Blob *in, const float x, 
													 Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = inPtr[i] + x;
  }
}

template <>
void Add<float, lang::Cpp>(const size_t num, const Blob *in1, const Blob *in2,
                           Blob *out, Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  float *outPtr= static_cast<float *>(out->mutable_data());
  const float *in1Ptr = static_cast<const float *>(in1->data());
  const float *in2Ptr = static_cast<const float *>(in2->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = in1Ptr[i] + in2Ptr[i];
  }
}

template <>
void Sub<float, lang::Cpp>(const size_t num, const Blob *in1, const Blob *in2,
                           Blob *out, Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  float *outPtr= static_cast<float *>(out->mutable_data());
  const float *in1Ptr = static_cast<const float *>(in1->data());
  const float *in2Ptr = static_cast<const float *>(in2->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = in1Ptr[i] - in2Ptr[i];
  }
}

template <>
void EltwiseMult<float, lang::Cpp>(const size_t num, const Blob *in, const float x,
                                   Blob *out, Context *ctx) {
  float *outPtr= static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = inPtr[i] * x;
  }
}

template <>
void EltwiseMult<float, lang::Cpp>(const size_t num, const Blob *in1, const Blob *in2,
                                   Blob *out, Context *ctx) {
  float *outPtr= static_cast<float *>(out->mutable_data());
  const float *in1Ptr = static_cast<const float *>(in1->data());
  const float *in2Ptr = static_cast<const float *>(in2->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = in1Ptr[i] * in2Ptr[i];
  }
}

template <>
void Div<float, lang::Cpp>(const size_t num, const Blob *in1, const Blob *in2,
                                   Blob *out, Context *ctx) {
  float *outPtr= static_cast<float *>(out->mutable_data());
  const float *in1Ptr = static_cast<const float *>(in1->data());
  const float *in2Ptr = static_cast<const float *>(in2->data());
  for (size_t i = 0; i < num; i++) {
		CHECK_NE(in2Ptr[i],0.f);
    outPtr[i] = in1Ptr[i] / in2Ptr[i];
  }
}

template <>
void Div<float, lang::Cpp>(const size_t num, const float x, const Blob *in, 
         								  Blob *out, Context *ctx) {
	float *outPtr= static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
		CHECK_NE(inPtr[i],0.f);
    outPtr[i] = x / inPtr[i];
  }
}

template <>
void Outer<float, lang::Cpp>(const size_t m, const size_t n, const Blob *in1, const Blob *in2,
           Blob *out, Context *ctx) {
	float *outPtr= static_cast<float *>(out->mutable_data());
	const float *in1Ptr = static_cast<const float *>(in1->data());
  const float *in2Ptr = static_cast<const float *>(in2->data());
	for (size_t r = 0; r < m ; r++) {
		size_t offset = r * n;
		for (size_t c = 0; c < n; c++) {
			outPtr[offset + c] = in1Ptr[r] * in2Ptr[c];
		}
	}
}

template <>
void LT<float, lang::Cpp>(const size_t num, const Blob *in, const float x,
                          Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = (inPtr[i] < x) ? 1.f : 0.f;
  }
}

template <>
void LE<float, lang::Cpp>(const size_t num, const Blob *in, const float x,
                          Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = (inPtr[i] <= x) ? 1.f : 0.f;
  }
}

template <>
void GT<float, lang::Cpp>(const size_t num, const Blob *in, const float x,
                          Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = (inPtr[i] > x) ? 1.f : 0.f;
  }
}

template <>
void GE<float, lang::Cpp>(const size_t num, const Blob *in, const float x,
                          Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = (inPtr[i] >= x) ? 1.f : 0.f;
  }
}

template <>
void Amax<float, lang::Cpp>(const size_t num, const Blob *in, size_t *out, Context *ctx) {
	size_t maxPos = 0;
	float maxVal = 0;
  const float *inPtr = static_cast<const float *>(in->data());
	for (size_t i = 0; i < num; i++) {
		if (i == 0) {
			maxVal = inPtr[i]; 
		}
		else if (inPtr[i] > maxVal) {
			maxVal = inPtr[i];
			maxPos = i;
		}
	}
	*out = maxPos;
}

template <>
void Amin<float, lang::Cpp>(const size_t num, const Blob *in, size_t *out, Context *ctx) {
	size_t minPos = 0;
	float minVal = 0;
  const float *inPtr = static_cast<const float *>(in->data());
	for (size_t i = 0; i < num; i++) {
		if (i == 0) {
			minVal = inPtr[i]; 
		}
		else if (inPtr[i] > minVal) {
			minVal = inPtr[i];
			minPos = i;
		}
	}
	*out = minPos;
}

template <>
void Asum<float, lang::Cpp>(const size_t num, const Blob *in, float *out, Context *ctx) {
	float sum = 0;
	const float *inPtr = static_cast<const float *>(in->data());
	for (size_t i = 0; i < num; i++) {
		sum += fabs(inPtr[i]);
	}
}

template <>
void Axpy<float, lang::Cpp>(const size_t num, const float alpha, const Blob *in,
          							 	  Blob *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in->data());
	for (size_t i = 0; i < num; i++) {	
		outPtr[i] += alpha * inPtr[i];
	}
}

template <>
void Scale<float, lang::Cpp>(const size_t num, const float x, Blob *out, Context *ctx) {
	float *outPtr = static_cast<float *>(out->mutable_data());
	for (size_t i = 0; i < num; i++) {
		outPtr[i] *= x;
	}
}

//template <>
//void Dot<float, lang::Cpp>(const size_t num, const Blob *in1, const Blob *in2,
//         									 float *out, Context *ctx) {
//	float sum = 0;
//	const float *in1Ptr = static_cast<const float *>(in1->data());
//	const float *in2Ptr = static_cast<const float *>(in2->data());
//	for (size_t i = 0; i < num; i++) {
//		sum += in1Ptr[i] * in2Ptr[i];
//	}
//}

template <>
void GEMV<float, lang::Cpp>(bool trans, const size_t m, const size_t n, const float alpha,
          const Blob *A, const Blob *v, const float beta, 
					Blob *out, Context *ctx) {
	float *outPtr = static_cast<float *>(out->mutable_data());
	const float* APtr = static_cast<const float *>(A->data());
	const float* vPtr = static_cast<const float *>(v->data());
	for (size_t r = 0; r < m; r++) {
		float sum = 0; 
		for (size_t c = 0; c < n; c++) {
			size_t idx = trans ? c * m + r : r * n + c;	
			sum += APtr[idx] * vPtr[c];
		}
		outPtr[r] = alpha * sum + beta * outPtr[r];
	}
}

template <>
void DGMM<float, lang::Cpp>(const bool side_right, const size_t nrow,
                            const size_t ncol, const Blob *M, const Blob *v,
                            Blob *out, Context *ctx) {
  const float *MPtr = static_cast<const float *>(M->data());
  const float *vPtr = static_cast<const float *>(v->data());
  float *outPtr = static_cast<float *>(out->mutable_data());
  if (side_right) {
    for (size_t r = 0; r < nrow; r++) {
      size_t offset = r * ncol;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[offset + c] = MPtr[offset + c] * vPtr[c];
      }
    }
  } else {
    for (size_t r = 0; r < nrow; r++) {
      size_t offset = r * ncol;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[offset + c] = MPtr[offset + c] * vPtr[r];
      }
    }
  }
}

template <>
void Bernoulli<float, lang::Cpp>(const size_t num, const float p, Blob *out, Context *ctx) {
  std::bernoulli_distribution distribution(p);
  float *outPtr = static_cast<float *>(out->mutable_data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = distribution(ctx->random_generator) ? 1.0f : 0.0f;
  }
}

template <>
void Uniform<float, lang::Cpp>(const size_t num, const float low, const float high, Blob *out,
                               Context *ctx) {
  std::uniform_real_distribution<float> distribution(low, high);
  float *outPtr= static_cast<float *>(out->mutable_data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}

template <>
void Gaussian<float, lang::Cpp>(const size_t num, const float mean, const float std, Blob *out,
                                Context *ctx) {
  std::normal_distribution<float> distribution(mean, std);
  float *outPtr = static_cast<float *>(out->mutable_data());
  for (size_t i = 0; i < num; i++) {
    outPtr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}


#ifdef USE_CBLAS
template <>
void Dot<float, lang::Cpp>(const size_t num, const Blob *in1, const Blob *in2,
                           float *out, Context *ctx) {
  const float *in1Ptr = static_cast<const float *>(in1->data());
  const float *in2Ptr = static_cast<const float *>(in2->data());
  *out = cblas_sdot(num, in1Ptr, 1, in2Ptr, 1);
}

template <>
void GEMM<float, lang::Cpp>(const bool transA, const bool transB,
                            const size_t nrowA, const size_t ncolB,
                            const size_t ncolA, const float alpha,
                            const Blob *A, const Blob *B, const float beta,
                            Blob *C, Context *ctx) {
  auto transa = transA ? CblasTrans : CblasNoTrans;
  auto transb = transB ? CblasTrans : CblasNoTrans;
  auto lda = transA ? nrowA : ncolA;
  auto ldb = transB ? ncolA : ncolB;
  auto ldc = ncolB;
  const float *APtr = static_cast<const float *>(A->data());
  const float *BPtr = static_cast<const float *>(B->data());
  float *CPtr = static_cast<float *>(C->mutable_data());
  cblas_sgemm(CblasRowMajor, transa, transb, nrowA, ncolB, ncolA, alpha, APtr,
              lda, BPtr, ldb, beta, CPtr, ldc);
}

#endif  // USE_CBLAS
}  // namespace singa

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CPP_H_
