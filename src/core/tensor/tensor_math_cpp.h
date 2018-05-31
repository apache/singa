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
#include <cfloat>
#include "singa/core/common.h"
#include "singa/core/tensor.h"
#include <math.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

namespace singa {

// ===================== Helper Functions =============================

// generate a traversal_info vector based on the tensor's shape for the
// traverse_next function to work
vector<int> generate_traversal_info(const Tensor& x) {
  vector<int> traversal_info = {};
  for (size_t n = 0; n < (x.shape().size() + 2); ++n) {
    traversal_info.push_back(0);
  }
  return traversal_info;
};

//generate shape multipliers
//for e.g. tensor of shape (3,3), stride (1,3) will have shape multipliers of (3,1)
//for e.g. tensor of shape (3,3), stride (3,1) will also have shape multipliers of (3,1)
//this means that the 3rd, 6th, and 9th index of the array will always be the starting element of their respective rows
//so we need to need use the inner stride when jumping from 1st->2nd element, and outer stride when jumping from 2nd->3rd
vector<int> generate_shape_multipliers(const Tensor& x) {
  Shape y_shape = x.shape();
  if (y_shape.size() == 0) {
    return {1};
  }
  vector<int> shape_multipliers = {1};
  int cumulative_product = 1;

  for (size_t n = 0; n < (y_shape.size() - 1); ++n) {
    cumulative_product = cumulative_product * y_shape[y_shape.size() - 1 - n];
    shape_multipliers.insert(shape_multipliers.begin(), cumulative_product);
  }
  return shape_multipliers;
};

// ******************************************************************************************
// CPP traversal operations (works on const declarations without modifying tensor variables)
// ******************************************************************************************

//this function checks whether the next index falls on a special multiplier of the outer shape
//so the algorithm knows when to jump over/back to a starting element of the outer shape
//for e.g. in [[1,4,7], [2,5,8], [3,6,9]], elements 1,2,3 are the starting elements of their respective rows
//this additional check only has 1 loop for 2d matrix
//but runtime performance might degrade to O(nlog(n)) for higher dimensional tensors
int determine_order(vector<int>& shape_multipliers, int counter) {
  for (size_t n = 0; n < (shape_multipliers.size() - 1); ++n) {
    if ((counter % shape_multipliers[n]) == 0) {
      return ((shape_multipliers.size()) - 1 - n);
    }
  }
  return 0;
};

//this function updates the base indexes with the current index after every single traversal step,
//can be generalized beyond 2d cases
void update_base_index(const Tensor& x, vector<int>& traversal_info) {
  for (int n = 0; n < (traversal_info[x.shape().size() + 1] + 1); ++n) {
    traversal_info[n] = traversal_info[x.shape().size()];
  }
};

//function to traverse a const strided tensor object
//it requires an additional vector, traversal_info {0,0,0,0 ...}, comprising (x.shape().size()+2) elements of 0
//for e.g. 2d matrix:
//index 0 and 1 store the base row and column index respectively
//index 2 stores the current index of the traversal
//index 3 stores the order of the traversal for e.g. if the order is 0,
//it means the next element can be navigated to using the innermost stride
void traverse_next(const Tensor& x,
                   vector<int>& shape_multipliers,
                   vector<int>& traversal_info,
                   int counter) {

  update_base_index(x, traversal_info);
  traversal_info[x.shape().size() + 1] = determine_order(shape_multipliers, counter);
  traversal_info[x.shape().size()] = traversal_info[traversal_info[x.shape().size() + 1]] +
                                     x.strides()[x.strides().size() - traversal_info[x.shape().size() + 1] - 1];
};

template <typename DType>
void TraverseUnary(const Tensor &in, Tensor* out, std::function<DType(DType)> func) {
  DType *outPtr = static_cast<DType *>(out->block()->mutable_data());
  const DType *inPtr = static_cast<const DType *>(in.block()->data());
  vector<int> traversal_info = generate_traversal_info(in);
  vector<int> shape_multipliers = generate_shape_multipliers(in);

  for (size_t i = 0; i < in.Size(); i++) {
    outPtr[i] = func(inPtr[traversal_info[in.shape().size()]]);
    traverse_next(in, shape_multipliers, traversal_info, i + 1);
  }
}

template <typename DType>
void TraverseBinary(const Tensor &in1, const Tensor &in2, Tensor* out,
                    std::function<DType(DType, DType)> func) {
  DType *outPtr = static_cast<DType *>(out->block()->mutable_data());
  const DType *in1Ptr = static_cast<const DType *>(in1.block()->data());
  const DType *in2Ptr = static_cast<const DType *>(in2.block()->data());
  vector<int> traversal_info_in1 = generate_traversal_info(in1);
  vector<int> traversal_info_in2 = generate_traversal_info(in2);
  vector<int> shape_multipliers_in1 = generate_shape_multipliers(in1);
  vector<int> shape_multipliers_in2 = generate_shape_multipliers(in2);

  for (size_t i = 0; i < in1.Size(); i++) {
    outPtr[i] = func(in1Ptr[traversal_info_in1[in1.shape().size()]],
                     in2Ptr[traversal_info_in2[in2.shape().size()]]);
    traverse_next(in1, shape_multipliers_in1, traversal_info_in1, i + 1);
    traverse_next(in2, shape_multipliers_in2, traversal_info_in2, i + 1);
  }
}

// ******************************************************************************************
// traversal operations end
// ******************************************************************************************

// ===================== CUDA Functions =============================

template <>
void Abs<float, lang::Cpp>(const Tensor& in, Tensor* out, Context *ctx) {
  TraverseUnary<float>(in, out, [](float x) {return fabs(x);});
}

template <>
void Add<float, lang::Cpp>(const Tensor& in, const float x, Tensor* out, Context *ctx) {
  auto add_lambda = [&x](float a) {
    return (a + x);
  };
  TraverseUnary<float>(in, out, add_lambda);
}

template <>
void Add<float, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out, Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  auto add_lambda_binary = [](float a, float b) {
    return (a + b);
  };
  TraverseBinary<float>(in1, in2, out, add_lambda_binary);

}

template <>
void Clamp<float, lang::Cpp>(const float low, const float high,
                             const Tensor& in, Tensor* out,
                             Context *ctx) {
  auto clamp_lambda = [&low, &high](float a) {
    if (a < low) {return low;}
    else if (a > high) {return high;}
    else {return a;}
  };
  TraverseUnary<float>(in, out, clamp_lambda);
}

template <>
void Div<float, lang::Cpp>(const float x, const Tensor& in, Tensor* out,
                           Context *ctx) {
  const float *inPtr = static_cast<const float *>(in.block()->data());
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  vector<int> traversal_info = generate_traversal_info(in);
  vector<int> shape_multipliers = generate_shape_multipliers(in);

  for (size_t i = 0; i < in.Size(); i++) {
    CHECK_NE(inPtr[traversal_info[in.shape().size()]], 0.f);
    outPtr[i] = x / inPtr[traversal_info[in.shape().size()]];
    traverse_next(in, shape_multipliers, traversal_info, i + 1);
  }
}

template <>
void Div<float, lang::Cpp>(const Tensor& in1, const Tensor& in2,
                           Tensor* out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const float *in1Ptr = static_cast<const float *>(in1.block()->data());
  const float *in2Ptr = static_cast<const float *>(in2.block()->data());
  vector<int> traversal_info_in1 = generate_traversal_info(in1);
  vector<int> traversal_info_in2 = generate_traversal_info(in2);
  vector<int> shape_multipliers_in1 = generate_shape_multipliers(in1);
  vector<int> shape_multipliers_in2 = generate_shape_multipliers(in2);

  for (size_t i = 0; i < in1.Size(); i++) {
    CHECK_NE(in2Ptr[traversal_info_in2[in2.shape().size()]], 0.f);
    outPtr[i] = in1Ptr[traversal_info_in1[in1.shape().size()]] / in2Ptr[traversal_info_in2[in2.shape().size()]];
    traverse_next(in1, shape_multipliers_in1, traversal_info_in1, i + 1);
    traverse_next(in2, shape_multipliers_in2, traversal_info_in2, i + 1);
  }
}

template <>
void EltwiseMult<float, lang::Cpp>(const Tensor& in, const float x, Tensor* out,
                                   Context *ctx) {
  auto eltwisemult_lambda = [&x](float a) {
    return (a * x);
  };
  TraverseUnary<float>(in, out, eltwisemult_lambda);
}

template <>
void EltwiseMult<float, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                                   Context *ctx) {
  auto eltwisemult_lambda_binary = [](float a, float b) {
    return (a * b);
  };
  TraverseBinary<float>(in1, in2, out, eltwisemult_lambda_binary);
}

template <>
void Exp<float, lang::Cpp>(const Tensor& in, Tensor *out, Context *ctx) {
  TraverseUnary<float>(in, out, [](float x) {return exp(x);});
}

template <>
void GE<float, lang::Cpp>(const Tensor& in, const float x, Tensor* out,
                          Context *ctx) {
  auto ge_lambda = [&x](float a) {
    return (a >= x) ? 1.f : 0.f;
  };
  TraverseUnary<float>(in, out, ge_lambda);
}

template <>
void GE<float, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                          Context *ctx) {
  auto ge_lambda_binary = [](float a, float b) {
    return (a >= b) ? 1.f : 0.f;
  };
  TraverseBinary<float>(in1, in2, out, ge_lambda_binary);
}

template <>
void GT<float, lang::Cpp>(const Tensor& in, const float x, Tensor* out,
                          Context *ctx) {
  auto gt_lambda = [&x](float a) {
    return (a > x) ? 1.f : 0.f;
  };
  TraverseUnary<float>(in, out, gt_lambda);
}

template <>
void GT<float, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                          Context *ctx) {
  auto gt_lambda_binary = [](float a, float b) {
    return (a > b) ? 1.f : 0.f;
  };
  TraverseBinary<float>(in1, in2, out, gt_lambda_binary);
}

template <>
void LE<float, lang::Cpp>(const Tensor& in, const float x, Tensor* out,
                          Context *ctx) {
  auto le_lambda = [&x](float a) {
    return (a <= x) ? 1.f : 0.f;
  };
  TraverseUnary<float>(in, out, le_lambda);
}

template <>
void LE<float, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                          Context *ctx) {
  auto le_lambda_binary = [](float a, float b) {
    return (a <= b) ? 1.f : 0.f;
  };
  TraverseBinary<float>(in1, in2, out, le_lambda_binary);
}

template <>
void Log<float, lang::Cpp>(const Tensor& in, Tensor* out,
                           Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const float *inPtr = static_cast<const float *>(in.block()->data());
  vector<int> traversal_info = generate_traversal_info(in);
  vector<int> shape_multipliers = generate_shape_multipliers(in);

  for (size_t i = 0; i < in.Size(); i++) {
    CHECK_GT(inPtr[traversal_info[in.shape().size()]], 0.f);
    outPtr[i] = log(inPtr[traversal_info[in.shape().size()]]);
    traverse_next(in, shape_multipliers, traversal_info, i + 1);
  }
}

template <>
void LT<float, lang::Cpp>(const Tensor& in, const float x, Tensor* out,
                          Context *ctx) {
  auto lt_lambda = [&x](float a) {
    return (a < x) ? 1.f : 0.f;
  };
  TraverseUnary<float>(in, out, lt_lambda);
}


template <>
void LT<float, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                          Context *ctx) {
  auto lt_lambda_binary = [](float a, float b) {
    return (a < b) ? 1.f : 0.f;
  };
  TraverseBinary<float>(in1, in2, out, lt_lambda_binary);
}

template <>
void Pow<float, lang::Cpp>(const Tensor& in, const float x, Tensor *out, Context *ctx) {
  TraverseUnary<float>(in, out, [x](float y) {return pow(y, x);});
}

template <>
void Pow<float, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                           Context *ctx) {
  auto pow_lambda_binary = [](float a, float b) {
    return pow(a, b);
  };
  TraverseBinary<float>(in1, in2, out, pow_lambda_binary);
}

template <>
void ReLU<float, lang::Cpp>(const Tensor& in, Tensor* out,
                            Context *ctx) {
  auto relu_lambda = [](float a) {
    return (a >= 0.f) ? a : 0.f;
  };
  TraverseUnary<float>(in, out, relu_lambda);
}

template <>
void Set<float, lang::Cpp>(const float x, Tensor* out,
                           Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) outPtr[i] = x;
}

template <>
void Set<int, lang::Cpp>(const int x, Tensor* out,
                         Context *ctx) {
  int *outPtr = static_cast<int *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) outPtr[i] = x;
}

template <>
void Sigmoid<float, lang::Cpp>(const Tensor& in, Tensor* out,
                               Context *ctx) {
  auto sigmoid_lambda = [](float a) {
    return 1.f / (1.f + exp(-a));
  };
  TraverseUnary<float>(in, out, sigmoid_lambda);
}

template <>
void Sign<float, lang::Cpp>(const Tensor& in, Tensor* out,
                            Context *ctx) {
  auto sign_lambda = [](float a) {
    return (a > 0) - (a < 0);
  };
  TraverseUnary<float>(in, out, sign_lambda);
}

template <>
void Sqrt<float, lang::Cpp>(const Tensor& in, Tensor* out,
                            Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const float *inPtr = static_cast<const float *>(in.block()->data());
  vector<int> traversal_info = generate_traversal_info(in);
  vector<int> shape_multipliers = generate_shape_multipliers(in);

  for (size_t i = 0; i < in.Size(); i++) {
    CHECK_GE(inPtr[traversal_info[in.shape().size()]], 0.f);
    outPtr[i] = sqrt(inPtr[traversal_info[in.shape().size()]]);
    traverse_next(in, shape_multipliers, traversal_info, i + 1);
  }
}

template <>
void Sub<float, lang::Cpp>(const Tensor& in1, const Tensor& in2,
                           Tensor* out, Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  auto sub_lambda_binary = [](float a, float b) {
    return (a - b);
  };
  TraverseBinary<float>(in1, in2, out, sub_lambda_binary);
}

// sum all elements of input into out
// TODO(wangwei) optimize using omp
template <>
void Sum<float, lang::Cpp>(const Tensor& in, float *out,
                           Context *ctx) {
  float s = 0.f;
  const float *inPtr = static_cast<const float *>(in.block()->data());
  for (size_t i = 0; i < in.Size(); i++) {
    s += inPtr[i];
  }
  *out = s;
}

template <>
void Tanh<float, lang::Cpp>(const Tensor& in, Tensor* out,
                            Context *ctx) {
  auto tanh_lambda = [](float a) {
    return tanh(a);
  };
  TraverseUnary<float>(in, out, tanh_lambda);
}

template <>
void Transform<float, lang::Cpp>(const Tensor& in, Tensor* out,
                            Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const float *inPtr = static_cast<const float *>(in.block()->data());
  vector<int> traversal_info = generate_traversal_info(in);
  vector<int> shape_multipliers = generate_shape_multipliers(in);

  for (size_t i = 0; i < in.Size(); i++) {
    outPtr[i] = inPtr[traversal_info[in.shape().size()]];
    traverse_next(in, shape_multipliers, traversal_info, i + 1);
  }
}

template <>
void Bernoulli<float, lang::Cpp>(const float p, Tensor* out,
                                 Context *ctx) {
  std::bernoulli_distribution distribution(p);
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] = distribution(ctx->random_generator) ? 1.0f : 0.0f;
  }
}

template <>
void Gaussian<float, lang::Cpp>(const float mean,
                                const float std, Tensor* out, Context *ctx) {
  std::normal_distribution<float> distribution(mean, std);
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}

template <>
void Uniform<float, lang::Cpp>(const float low,
                               const float high, Tensor* out, Context *ctx) {
  std::uniform_real_distribution<float> distribution(low, high);
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}

// ====================Blas operations======================================

//warning, this function has block M overwritting to block M itself
template <>
void DGMM<float, lang::Cpp>(const bool side_right,
                            const Tensor& M, const Tensor& v,
                            Tensor* out, Context *ctx) {
  const float *MPtr = static_cast<const float *>(M.block()->data());
  const float *vPtr = static_cast<const float *>(v.block()->data());
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const size_t nrow = M.shape(0);
  const size_t ncol = M.shape(1);
  vector<int> traversal_info = generate_traversal_info(M);
  vector<int> shape_multipliers = generate_shape_multipliers(M);

  if (side_right) {
    for (size_t r = 0; r < nrow; r++) {
      size_t offset = r * ncol;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[traversal_info[M.shape().size()]] = MPtr[traversal_info[M.shape().size()]] * vPtr[c];
        traverse_next(M, shape_multipliers, traversal_info, offset + c + 1);
      }
    }
  } else {
    for (size_t r = 0; r < nrow; r++) {
      size_t offset = r * ncol;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[traversal_info[M.shape().size()]] = MPtr[traversal_info[M.shape().size()]] * vPtr[r];
        traverse_next(M, shape_multipliers, traversal_info, offset + c + 1);
      }
    }
  }
}


#ifdef USE_CBLAS
template <>
void Amax<float, lang::Cpp>(const Tensor& in, size_t *out,
                            Context *ctx) {
  const float *inPtr = static_cast<const float *>(in.block()->data());
  *out = cblas_isamax(in.Size(), inPtr, 1); //not using strided traversal
}

template <>
void Asum<float, lang::Cpp>(const Tensor& in, float *out,
                            Context *ctx) {
  const float *inPtr = static_cast<const float *>(in.block()->data());
  *out = cblas_sasum(in.Size(), inPtr, 1); //not using strided traversal
}

template <>
void Axpy<float, lang::Cpp>(const float alpha,
                            const Tensor& in, Tensor *out, Context *ctx) {
  //check input tensor for strides first
  if (in.strides() == out->strides()) {
    const float *inPtr = static_cast<const float *>(in.block()->data());
    float *outPtr = static_cast<float *>(out->block()->mutable_data());
    cblas_saxpy(in.Size(), alpha, inPtr, 1, outPtr, 1);
  } else {
    LOG(FATAL) << "Axpy, input and output strides do not match." ;
  }
}

template <>
void Dot<float, lang::Cpp>(const Tensor& in1, const Tensor& in2,
                           float *out, Context *ctx) {
  //check input tensor for strides first
  if (!(in1.transpose()) && !(in2.transpose())) {
    const float *in1Ptr = static_cast<const float *>(in1.block()->data());
    const float *in2Ptr = static_cast<const float *>(in2.block()->data());
    *out = cblas_sdot(in1.Size(), in1Ptr, 1, in2Ptr, 1);
  } else {
    LOG(FATAL) << "Dot, one of the input is tranposed. Not implemented yet." ;
  }
}

template <>
void Scale<float, lang::Cpp>(const float x, Tensor *out,
                             Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  cblas_sscal(out->Size(), x, outPtr, 1); //not using strided traversal
}

template <>
void Nrm2<float, lang::Cpp>(const Tensor& in, float *out,
                            Context *ctx) {
  const float *inPtr = static_cast<const float *>(in.block()->data());
  *out = cblas_snrm2(in.Size(), inPtr, 1); //not using strided traversal
}

template <>
void GEMV<float, lang::Cpp>(const float alpha, const Tensor& A, const Tensor& v,
                            const float beta, Tensor *out, Context *ctx) {
  const float *APtr = static_cast<const float *>(A.block()->data());
  const float *vPtr = static_cast<const float *>(v.block()->data());
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const size_t m = A.shape()[0];
  const size_t n = A.shape()[1];
  if (A.transpose()) {
    cblas_sgemv(CblasRowMajor, CblasTrans, n, m, alpha, APtr, m, vPtr, 1, beta,
                outPtr, 1);
  } else {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, APtr, n, vPtr, 1,
                beta, outPtr, 1);
  }
}

template <>
void GEMM<float, lang::Cpp>(const float alpha,
                            const Tensor& A, const Tensor& B, const float beta,
                            Tensor *C, Context *ctx) {
  auto transA = A.transpose();
  auto transa = transA ? CblasTrans : CblasNoTrans;
  auto transB = B.transpose();
  auto transb = transB ? CblasTrans : CblasNoTrans;
  const size_t nrowA = A.shape()[0];
  const size_t ncolA = A.shape()[1];
  const size_t ncolB = B.shape()[1];
  auto lda = transA ? nrowA : ncolA;
  auto ldb = transB ? ncolA : ncolB;
  auto ldc = ncolB;
  const float *APtr = static_cast<const float *>(A.block()->data());
  const float *BPtr = static_cast<const float *>(B.block()->data());
  float *CPtr = static_cast<float *>(C->block()->mutable_data());
  cblas_sgemm(CblasRowMajor, transa, transb, nrowA, ncolB, ncolA, alpha, APtr,
              lda, BPtr, ldb, beta, CPtr, ldc);
}

#else

template <>
void Amax<float, lang::Cpp>(const Tensor& in, size_t *out,
                            Context *ctx) {
  size_t maxPos = 0;
  float maxVal = 0;
  const float *inPtr = static_cast<const float *>(in.block()->data());
  for (size_t i = 0; i < in.Size(); i++) { //not using strided traversal
    if (i == 0) {
      maxVal = inPtr[i];
    } else if (inPtr[i] > maxVal) {
      maxVal = inPtr[i];
      maxPos = i;
    }
  }
  *out = maxPos;
}
template <>
void Amin<float, lang::Cpp>(const Tensor& in, size_t *out,
                            Context *ctx) {
  size_t minPos = 0;
  float minVal = 0;
  const float *inPtr = static_cast<const float *>(in.block()->data());
  for (size_t i = 0; i < in.Size(); i++) { //not using strided traversal
    if (i == 0) {
      minVal = inPtr[i];
    } else if (inPtr[i] > minVal) {
      minVal = inPtr[i];
      minPos = i;
    }
  }
  *out = minPos;
}

template <>
void Asum<float, lang::Cpp>(const Tensor& in, float *out,
                            Context *ctx) {
  float sum = 0;
  const float *inPtr = static_cast<const float *>(in.block()->data());
  for (size_t i = 0; i < in.Size(); i++) {
    sum += fabs(inPtr[i]); //not using strided traversal
  }
}

template <>
void Axpy<float, lang::Cpp>(const float alpha,
                            const Tensor& in, Tensor *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const float *inPtr = static_cast<const float *>(in.block()->data());
  vector<int> traversal_info = generate_traversal_info(in);
  vector<int> shape_multipliers = generate_shape_multipliers(in);

  for (size_t i = 0; i < in.Size(); i++) {
    outPtr[i] += alpha * inPtr[traversal_info[in.shape().size()]];
    traverse_next(in, shape_multipliers, traversal_info, i + 1);
  }
}

template <>
void Scale<float, lang::Cpp>(const float x, Tensor *out,
                             Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] *= x; //not using strided traversal
  }
}

template <>
void Dot<float, lang::Cpp>(const Tensor& in1, const Tensor& in2,
                           float *out, Context *ctx) {
  float sum = 0;
  // const float *in1Ptr = static_cast<const float *>(in1.data());
  // const float *in2Ptr = static_cast<const float *>(in2.data());
  // for (size_t i = 0; i < in.Size(); i++) {
  //   sum += in1Ptr[i] * in2Ptr[i];
  // }
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const float *in1Ptr = static_cast<const float *>(in1.block()->data());
  const float *in2Ptr = static_cast<const float *>(in2.block()->data());
  vector<int> traversal_info_in1 = generate_traversal_info(in1);
  vector<int> traversal_info_in2 = generate_traversal_info(in2);
  vector<int> shape_multipliers_in1 = generate_shape_multipliers(in1);
  vector<int> shape_multipliers_in2 = generate_shape_multipliers(in2);

  for (size_t i = 0; i < in1.Size(); i++) {
    sum += in1Ptr[traversal_info_in1[in1.shape().size()]] * in2Ptr[traversal_info_in2[in2.shape().size()]];
    traverse_next(in1, shape_multipliers_in1, traversal_info_in1, i + 1);
    traverse_next(in2, shape_multipliers_in2, traversal_info_in2, i + 1);
  }
}

template <>
void GEMV<float, lang::Cpp>(const float alpha, const Tensor& A, const Tensor& v,
                            const float beta, Tensor *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const float *APtr = static_cast<const float *>(A.block()->data());
  const float *vPtr = static_cast<const float *>(v.block()->data());
  bool trans = A.transpose();
  const size_t m = A.shape(0);
  const size_t n = A.shape(1);
  for (size_t r = 0; r < m; r++) {
    float sum = 0;
    for (size_t c = 0; c < n; c++) {
      size_t idx = trans ? c * m + r : r * n + c;
      sum += APtr[idx] * vPtr[c];
    }
    outPtr[r] = alpha * sum + beta * outPtr[r];
  }
}

#endif  // USE_CBLAS
template <>
void ComputeCrossEntropy<float, lang::Cpp>(bool int_target,
    const size_t batchsize,
    const size_t dim, const Block *p,
    const Block *t, Block *loss,
    Context *ctx) {
  const float *pPtr = static_cast<const float *>(p->data());
  const int *tPtr = static_cast<const int *>(t->data());
  float *lossPtr = static_cast<float *>(loss->mutable_data());
  if (int_target) {
    for (size_t i = 0; i < batchsize; i++) {
      int truth_idx = tPtr[i];
      CHECK_GE(truth_idx, 0);
      float prob_of_truth = pPtr[i * dim + truth_idx];
      lossPtr[i] = -std::log((std::max)(prob_of_truth, FLT_MIN));
    }
  } else {
    for (size_t i = 0; i < batchsize; i++) {
      float sum = 0.f;
      for (size_t j = 0; j < dim; j++) {
        sum += tPtr[i * dim + j];
      }
      float loss = 0.f;
      for (size_t j = 0, offset = i * dim; j < dim; j++, offset++) {
        loss -= tPtr[offset] / sum * std::log((std::max)(pPtr[offset], FLT_MIN));
      }
      lossPtr[i] = loss;
    }
  }
}

template <>
void SoftmaxCrossEntropyBwd<float, lang::Cpp>(bool int_target,
    const size_t batchsize,
    const size_t dim, const Block *p,
    const Block *t, Block *grad,
    Context *ctx) {
  CHECK_EQ(p, grad) << "Use the same pointer to optimize performance";
  // const float* pPtr = static_cast<const float*>(p->data());
  const int *tPtr = static_cast<const int *>(t->data());
  float *gradPtr = static_cast<float *>(grad->mutable_data());

  if (int_target) {
    for (size_t i = 0; i < batchsize; i++) {
      int truth_idx = static_cast<int>(tPtr[i]);
      CHECK_GE(truth_idx, 0);
      gradPtr[i * dim + truth_idx] -= 1.0;
    }
  } else {
    for (size_t i = 0; i < batchsize; i++) {
      float sum = 0.f;
      for (size_t j = 0; j < dim; j++) {
        sum += tPtr[i * dim + j];
      }
      for (size_t j = 0, offset = i * dim; j < dim; j++, offset++) {
        gradPtr[offset] -= tPtr[offset] / sum;
      }
    }
  }
}

template <>
void RowMax<float, lang::Cpp>(const Tensor& in, Tensor *out, Context *ctx) {
  const float *inPtr = static_cast<const float *>(in.block()->data());
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const size_t nrow = in.shape()[0];
  const size_t ncol = in.shape()[1];
  vector<int> traversal_info = generate_traversal_info(in);
  vector<int> shape_multipliers = generate_shape_multipliers(in);

  for (size_t r = 0; r < nrow; r++) {
    int counter_offset = (r * ncol);
    float maxval = 0;
    for (size_t c = 0; c < ncol; c++) {
      maxval = (std::max)(maxval, inPtr[traversal_info[in.shape().size()]]);
      traverse_next(in, shape_multipliers, traversal_info, counter_offset + c + 1);
    }
    outPtr[r] = maxval;
  }
}

// =========Matrix operations ================================================
/*
template <>
void AddCol<float, lang::Cpp>(const size_t nrow, const size_t ncol,
                              const Tensor& A, const Tensor& v, Tensor* out,
                              Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *APtr = static_cast<const float *>(A.data());
  const float *vPtr = static_cast<const float *>(v.data());
  for (size_t r = 0; r < nrow; r++) {
    size_t offset = r * ncol;
    for (size_t c = 0; c < ncol; c++) {
      outPtr[offset + c] = APtr[offset + c] + vPtr[r];
    }
  }
}

template <>
void AddRow<float, lang::Cpp>(const size_t nrow, const size_t ncol,
                              const Tensor& A, const Tensor& v, Tensor* out,
                              Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *APtr = static_cast<const float *>(A.data());
  const float *vPtr = static_cast<const float *>(v.data());
  for (size_t r = 0; r < nrow; r++) {
    size_t offset = r * ncol;
    for (size_t c = 0; c < ncol; c++) {
      outPtr[offset + c] = APtr[offset + c] + vPtr[c];
    }
  }
}
template <>
void Outer<float, lang::Cpp>(const size_t m, const size_t n, const Tensor& in1,
                             const Tensor& in2, Tensor* out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *in1Ptr = static_cast<const float *>(in1.data());
  const float *in2Ptr = static_cast<const float *>(in2.data());
  for (size_t r = 0; r < m; r++) {
    size_t offset = r * n;
    for (size_t c = 0; c < n; c++) {
      outPtr[offset + c] = in1Ptr[r] * in2Ptr[c];
    }
  }
}
template <>
void Softmax<float, lang::Cpp>(const size_t nrow, const size_t ncol,
                               const Tensor& in, Tensor* out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in.data());
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
void SumColumns<float, lang::Cpp>(const size_t nrow, const size_t ncol,
                                  const Tensor& in, Tensor* out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in.data());
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
void SumRows<float, lang::Cpp>(const size_t nrow, const size_t ncol,
                               const Tensor& in, Tensor* out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->mutable_data());
  const float *inPtr = static_cast<const float *>(in.data());
  for (size_t r = 0; r < nrow; r++) {
    size_t offset = r * ncol;
    outPtr[r] = 0.f;
    for (size_t c = 0; c < ncol; c++) {
      outPtr[r] += inPtr[offset + c];
    }
  }
}
*/
}  // namespace singa

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CPP_H_
