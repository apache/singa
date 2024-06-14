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
// #include "./stacktrace.h"
#include <math.h>

#include <algorithm>
#include <cfloat>
#include <iostream>
#include <iterator>
#include <sstream>

#include "singa/core/common.h"
#include "singa/core/tensor.h"

#ifdef USE_CBLAS
#include <cblas.h>
#endif

namespace singa {

// ===================== Helper Functions =============================

// generate a traversal_info vector based on the tensor's shape for the
// traverse_next function to work
vector<int> generate_traversal_info(const Tensor &x) {
  vector<int> traversal_info = {};
  for (size_t n = 0; n < (x.shape().size() + 2); ++n) {
    traversal_info.push_back(0);
  }
  return traversal_info;
};

// generate shape multipliers
// for e.g. tensor of shape (3,3), stride (1,3) will have shape multipliers of
// (3,1)
// for e.g. tensor of shape (3,3), stride (3,1) will also have shape multipliers
// of (3,1)
// this means that the 3rd, 6th, and 9th index of the array will always be the
// starting element of their respective rows
// so we need to need use the inner stride when jumping from 1st->2nd element,
// and outer stride when jumping from 2nd->3rd
vector<int> generate_shape_multipliers(const Tensor &x) {
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
// CPP traversal operations (works on const declarations without modifying
// tensor variables)
// ******************************************************************************************

// this function checks whether the next index falls on a special multiplier of
// the outer shape
// so the algorithm knows when to jump over/back to a starting element of the
// outer shape
// for e.g. in [[1,4,7], [2,5,8], [3,6,9]], elements 1,2,3 are the starting
// elements of their respective rows
// this additional check only has 1 loop for 2d matrix
// but runtime performance might degrade to O(nlog(n)) for higher dimensional
// tensors
int determine_order(vector<int> &shape_multipliers, int counter) {
  for (size_t n = 0; n < (shape_multipliers.size() - 1); ++n) {
    if ((counter % shape_multipliers[n]) == 0) {
      return ((shape_multipliers.size()) - 1 - n);
    }
  }
  return 0;
};

// this function updates the base indexes with the current index after every
// single traversal step,
// can be generalized beyond 2d cases
void update_base_index(const Tensor &x, vector<int> &traversal_info) {
  for (int n = 0; n < (traversal_info[x.shape().size() + 1] + 1); ++n) {
    traversal_info[n] = traversal_info[x.shape().size()];
  }
};

// function to traverse a const strided tensor object
// it requires an additional vector, traversal_info {0,0,0,0 ...}, comprising
// (x.shape().size()+2) elements of 0
// for e.g. 2d matrix:
// index 0 and 1 store the base row and column index respectively
// index 2 stores the current index of the traversal
// index 3 stores the order of the traversal for e.g. if the order is 0,
// it means the next element can be navigated to using the innermost stride
void traverse_next(const Tensor &x, vector<int> &shape_multipliers,
                   vector<int> &traversal_info, int counter) {
  update_base_index(x, traversal_info);
  traversal_info[x.shape().size() + 1] =
      determine_order(shape_multipliers, counter);
  traversal_info[x.shape().size()] =
      traversal_info[traversal_info[x.shape().size() + 1]] +
      x.stride()[x.stride().size() - traversal_info[x.shape().size() + 1] - 1];
};

inline int next_offset(int offset, const vector<size_t> &shape,
                       const vector<int> &stride, vector<int> *index) {
  for (int k = shape.size() - 1; k >= 0; k--) {
    if (index->at(k) + 1 < int(shape.at(k))) {
      offset += stride.at(k);
      index->at(k) += 1;
      break;
    }
    index->at(k) = 0;
    offset -= stride.at(k) * (shape.at(k) - 1);
  }
  return offset;
}

template <typename DType>
void traverse_unary(const Tensor &in, Tensor *out,
                    std::function<DType(DType)> func) {
  DType *outPtr = static_cast<DType *>(out->block()->mutable_data());
  const DType *inPtr = static_cast<const DType *>(in.block()->data());
  /*
  vector<int> traversal_info = generate_traversal_info(in);
  vector<int> shape_multipliers = generate_shape_multipliers(in);

  for (size_t i = 0; i < in.Size(); i++) {
    outPtr[i] = func(inPtr[traversal_info[in.shape().size()]]);
    traverse_next(in, shape_multipliers, traversal_info, i + 1);
  }
  */
  CHECK(in.shape() == out->shape());
  if (in.stride() == out->stride()) {
    for (size_t i = 0; i < in.Size(); i++) outPtr[i] = func(inPtr[i]);
  } else {
    // LOG(INFO) << "not equal stride";
    size_t in_offset = 0, out_offset = 0;
    vector<int> in_idx(in.nDim(), 0), out_idx(out->nDim(), 0);
    for (size_t i = 0; i < Product(in.shape()); i++) {
      outPtr[out_offset] = func(inPtr[in_offset]);
      out_offset =
          next_offset(out_offset, out->shape(), out->stride(), &out_idx);
      in_offset = next_offset(in_offset, in.shape(), in.stride(), &in_idx);
    }
  }
}

template <typename DType>
void traverse_binary(const Tensor &in1, const Tensor &in2, Tensor *out,
                     std::function<DType(DType, DType)> func) {
  DType *outPtr = static_cast<DType *>(out->block()->mutable_data());
  const DType *in1Ptr = static_cast<const DType *>(in1.block()->data());
  const DType *in2Ptr = static_cast<const DType *>(in2.block()->data());
  /*
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
  */
  auto prod = Product(in1.shape());
  CHECK(in1.shape() == out->shape());
  CHECK(in2.shape() == out->shape());
  if ((in1.stride() == out->stride()) && (in2.stride() == in1.stride())) {
    for (size_t i = 0; i < prod; i++) outPtr[i] = func(in1Ptr[i], in2Ptr[i]);
  } else {
    /*
    LOG(INFO) << "not equal stride";
    std::ostringstream s1, s2, s3, s4, s5, s6;
    std::copy(in1.stride().begin(), in1.stride().end(),
    std::ostream_iterator<int>(s1, ", "));
    std::copy(in2.stride().begin(), in2.stride().end(),
    std::ostream_iterator<int>(s2, ", "));
    std::copy(out->stride().begin(), out->stride().end(),
    std::ostream_iterator<int>(s3, ", "));

    std::copy(in1.shape().begin(), in1.shape().end(),
    std::ostream_iterator<int>(s4, ", "));
    std::copy(in2.shape().begin(), in2.shape().end(),
    std::ostream_iterator<int>(s5, ", "));
    std::copy(out->shape().begin(), out->shape().end(),
    std::ostream_iterator<int>(s6, ", "));

    LOG(INFO) << s1.str() << ": " << s4.str();
    LOG(INFO) << s2.str() << ": " << s5.str();
    LOG(INFO) << s3.str() << ": " << s6.str();
    LOG(INFO) << Backtrace();
    */

    size_t in1_offset = 0, in2_offset = 0, out_offset = 0;
    vector<int> in1_idx(in1.nDim(), 0), in2_idx(in2.nDim(), 0),
        out_idx(out->nDim(), 0);
    for (size_t i = 0; i < prod; i++) {
      outPtr[out_offset] = func(in1Ptr[in1_offset], in2Ptr[in2_offset]);
      out_offset =
          next_offset(out_offset, out->shape(), out->stride(), &out_idx);
      in1_offset = next_offset(in1_offset, in1.shape(), in1.stride(), &in1_idx);
      in2_offset = next_offset(in2_offset, in2.shape(), in2.stride(), &in2_idx);
      // LOG(INFO) <<  in1_offset << ", " << in2_offset << ", " << out_offset;
    }
  }
}

// ******************************************************************************************
// traversal operations end
// ******************************************************************************************

// ===================== CUDA Functions =============================

template <>
void Abs<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  traverse_unary<float>(in, out, [](float x) { return fabs(x); });
}

template <>
void Erf<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  traverse_unary<float>(in, out, [](float x) { return erff(x); });
}

template <>
void CastCopy<float, half_float::half, lang::Cpp>(const Tensor *src,
                                                  Tensor *dst, Context *ctx) {
  half_float::half *dst_array =
      static_cast<half_float::half *>(dst->block()->mutable_data());
  const float *src_array = static_cast<const float *>(src->block()->data());
  for (int i = 0; i < dst->Size(); ++i)
    dst_array[i] = static_cast<half_float::half>(src_array[i]);
}

template <>
void CastCopy<half_float::half, float, lang::Cpp>(const Tensor *src,
                                                  Tensor *dst, Context *ctx) {
  float *dst_array = static_cast<float *>(dst->block()->mutable_data());
  const half_float::half *src_array =
      static_cast<const half_float::half *>(src->block()->data());
  for (int i = 0; i < dst->Size(); ++i)
    dst_array[i] = static_cast<float>(src_array[i]);
}

template <>
void CastCopy<float, int, lang::Cpp>(const Tensor *src, Tensor *dst,
                                     Context *ctx) {
  int *dst_array = static_cast<int *>(dst->block()->mutable_data());
  const float *src_array = static_cast<const float *>(src->block()->data());
  for (int i = 0; i < dst->Size(); ++i) dst_array[i] = (int)src_array[i];
}

template <>
void CastCopy<int, float, lang::Cpp>(const Tensor *src, Tensor *dst,
                                     Context *ctx) {
  float *dst_array = static_cast<float *>(dst->block()->mutable_data());
  const int *src_array = static_cast<const int *>(src->block()->data());
  for (int i = 0; i < dst->Size(); ++i) dst_array[i] = (float)src_array[i];
}

template <>
void Ceil<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  traverse_unary<float>(in, out, [](float x) { return std::ceil(x); });
}

template <>
void Floor<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  traverse_unary<float>(in, out, [](float x) { return std::floor(x); });
}

template <>
void Round<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  traverse_unary<float>(in, out, [](float x) { return std::round(x); });
}

template <>
void RoundE<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  traverse_unary<float>(in, out, [](float x) {
    float doub = x * 2;
    if (ceilf(doub) == doub) {
      return std::round(x / 2) * 2;
    } else {
      return std::round(x);
    }
  });
}

#ifdef USE_DNNL
template <>
void SoftMax<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto md = dnnl::memory::desc({static_cast<long long>(in.shape()[0]),
                                static_cast<long long>(in.shape()[1])},
                               dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::ab);
  auto in_mem = dnnl::memory(md, ctx->dnnl_engine, in.block()->mutable_data());
  auto out_mem =
      dnnl::memory(md, ctx->dnnl_engine, out->block()->mutable_data());

  auto softmax_desc =
      dnnl::softmax_forward::desc(dnnl::prop_kind::forward_scoring, md, 1);
  auto softmax_prim_desc =
      dnnl::softmax_forward::primitive_desc(softmax_desc, ctx->dnnl_engine);
  auto softmax = dnnl::softmax_forward(softmax_prim_desc);
  softmax.execute(ctx->dnnl_stream,
                  {{DNNL_ARG_SRC, in_mem}, {DNNL_ARG_DST, out_mem}});
  ctx->dnnl_stream.wait();
}

template <>
void SoftMaxBackward<float, lang::Cpp>(const Tensor &in, Tensor *out,
                                       const Tensor &fdout, Context *ctx) {
  auto md = dnnl::memory::desc({static_cast<long long>(in.shape()[0]),
                                static_cast<long long>(in.shape()[1])},
                               dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::ab);
  auto in_mem = dnnl::memory(md, ctx->dnnl_engine, in.block()->mutable_data());
  auto fdout_mem =
      dnnl::memory(md, ctx->dnnl_engine, fdout.block()->mutable_data());
  auto out_mem =
      dnnl::memory(md, ctx->dnnl_engine, out->block()->mutable_data());

  auto softmax_desc =
      dnnl::softmax_forward::desc(dnnl::prop_kind::forward_scoring, md, 1);
  auto softmax_prim_desc =
      dnnl::softmax_forward::primitive_desc(softmax_desc, ctx->dnnl_engine);

  auto softmaxbwd_desc = dnnl::softmax_backward::desc(md, md, 1);
  auto softmaxbwd_prim_desc = dnnl::softmax_backward::primitive_desc(
      softmaxbwd_desc, ctx->dnnl_engine, softmax_prim_desc);
  auto softmaxbwd = dnnl::softmax_backward(softmaxbwd_prim_desc);
  softmaxbwd.execute(ctx->dnnl_stream, {{DNNL_ARG_DIFF_SRC, out_mem},
                                        {DNNL_ARG_DIFF_DST, in_mem},
                                        {DNNL_ARG_DST, fdout_mem}});
  ctx->dnnl_stream.wait();
}
#else
// native Softmax without DNNL
template <>
void SoftMax<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  CHECK_LE(in.nDim(), 2u)
      << "Axis is required for SoftMax on multi dimemsional tensor";
  out->CopyData(in);
  size_t nrow = 1, ncol = in.Size(), size = ncol;
  if (in.nDim() == 2u) {
    nrow = in.shape(0);
    ncol = size / nrow;
    out->Reshape(Shape{nrow, ncol});
  }
  Tensor tmp = RowMax(*out);
  SubColumn(tmp, out);
  Exp(*out, out);

  SumColumns(*out, &tmp);
  DivColumn(tmp, out);
  out->Reshape(in.shape());
}
#endif  // USE_DNNL

template <>
void Add<float, lang::Cpp>(const Tensor &in, const float x, Tensor *out,
                           Context *ctx) {
  auto add_lambda = [&x](float a) { return (a + x); };
  traverse_unary<float>(in, out, add_lambda);
}

template <>
void Add<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                           Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  auto add_lambda_binary = [](float a, float b) { return (a + b); };
  traverse_binary<float>(in1, in2, out, add_lambda_binary);
}

template <>
void Clamp<float, lang::Cpp>(const float low, const float high,
                             const Tensor &in, Tensor *out, Context *ctx) {
  auto clamp_lambda = [&low, &high](float a) {
    if (a < low) {
      return low;
    } else if (a > high) {
      return high;
    } else {
      return a;
    }
  };
  traverse_unary<float>(in, out, clamp_lambda);
}

template <>
void Div<float, lang::Cpp>(const float x, const Tensor &in, Tensor *out,
                           Context *ctx) {
  auto const_div = [&x](float a) {
    CHECK_NE(a, 0.f);
    return x / a;
  };
  traverse_unary<float>(in, out, const_div);
}

template <>
void Div<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                           Context *ctx) {
  auto binary_div = [](float a, float b) {
    CHECK_NE(b, 0.f);
    return a / b;
  };
  traverse_binary<float>(in1, in2, out, binary_div);
}

template <>
void EltwiseMult<float, lang::Cpp>(const Tensor &in, const float x, Tensor *out,
                                   Context *ctx) {
  auto eltwisemult_lambda = [&x](float a) { return (a * x); };
  traverse_unary<float>(in, out, eltwisemult_lambda);
}

template <>
void EltwiseMult<float, lang::Cpp>(const Tensor &in1, const Tensor &in2,
                                   Tensor *out, Context *ctx) {
  auto eltwisemult_lambda_binary = [](float a, float b) { return (a * b); };
  traverse_binary<float>(in1, in2, out, eltwisemult_lambda_binary);
}

template <>
void ReLUBackward<float, lang::Cpp>(const Tensor &in1, const Tensor &in2,
                                    Tensor *out, Context *ctx) {
  auto relubackward_lambda = [](float a, float b) { return (b > 0) ? a : 0.f; };
  traverse_binary<float>(in1, in2, out, relubackward_lambda);
}

template <>
void Exp<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  traverse_unary<float>(in, out, [](float x) { return exp(x); });
}

template <>
void GE<float, lang::Cpp>(const Tensor &in, const float x, Tensor *out,
                          Context *ctx) {
  auto ge_lambda = [&x](float a) { return (a >= x) ? 1.f : 0.f; };
  traverse_unary<float>(in, out, ge_lambda);
}

template <>
void GE<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                          Context *ctx) {
  auto ge_lambda_binary = [](float a, float b) { return (a >= b) ? 1.f : 0.f; };
  traverse_binary<float>(in1, in2, out, ge_lambda_binary);
}

template <>
void GE<int, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                        Context *ctx) {
  auto ge_lambda_binary = [](int a, int b) { return (a >= b) ? 1.f : 0.f; };
  traverse_binary<int>(in1, in2, out, ge_lambda_binary);
}

template <>
void GT<float, lang::Cpp>(const Tensor &in, const float x, Tensor *out,
                          Context *ctx) {
  auto gt_lambda = [&x](float a) { return (a > x) ? 1.f : 0.f; };
  traverse_unary<float>(in, out, gt_lambda);
}

template <>
void GT<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                          Context *ctx) {
  auto gt_lambda_binary = [](float a, float b) { return (a > b) ? 1.f : 0.f; };
  traverse_binary<float>(in1, in2, out, gt_lambda_binary);
}

template <>
void GT<int, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                        Context *ctx) {
  auto gt_lambda_binary = [](int a, int b) { return (a > b) ? 1.f : 0.f; };
  traverse_binary<int>(in1, in2, out, gt_lambda_binary);
}

template <>
void LE<float, lang::Cpp>(const Tensor &in, const float x, Tensor *out,
                          Context *ctx) {
  auto le_lambda = [&x](float a) { return (a <= x) ? 1.f : 0.f; };
  traverse_unary<float>(in, out, le_lambda);
}

template <>
void LE<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                          Context *ctx) {
  auto le_lambda_binary = [](float a, float b) { return (a <= b) ? 1.f : 0.f; };
  traverse_binary<float>(in1, in2, out, le_lambda_binary);
}

template <>
void LE<int, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                        Context *ctx) {
  auto le_lambda_binary = [](int a, int b) { return (a <= b) ? 1.f : 0.f; };
  traverse_binary<int>(in1, in2, out, le_lambda_binary);
}

template <>
void Log<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto ulog = [](float a) {
    CHECK_GT(a, 0.f);
    return log(a);
  };
  traverse_unary<float>(in, out, ulog);
}

template <>
void LT<float, lang::Cpp>(const Tensor &in, const float x, Tensor *out,
                          Context *ctx) {
  auto lt_lambda = [&x](float a) { return (a < x) ? 1.f : 0.f; };
  traverse_unary<float>(in, out, lt_lambda);
}

template <>
void LT<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                          Context *ctx) {
  auto lt_lambda_binary = [](float a, float b) { return (a < b) ? 1.f : 0.f; };
  traverse_binary<float>(in1, in2, out, lt_lambda_binary);
}

template <>
void LT<int, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                        Context *ctx) {
  auto lt_lambda_binary = [](int a, int b) { return (a < b) ? 1.f : 0.f; };
  traverse_binary<int>(in1, in2, out, lt_lambda_binary);
}

template <>
void EQ<float, lang::Cpp>(const Tensor &in, const float x, Tensor *out,
                          Context *ctx) {
  auto eq_lambda = [&x](float a) { return (a == x) ? 1.f : 0.f; };
  traverse_unary<float>(in, out, eq_lambda);
}

template <>
void EQ<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                          Context *ctx) {
  auto eq_lambda_binary = [](float a, float b) { return (a == b) ? 1.f : 0.f; };
  traverse_binary<float>(in1, in2, out, eq_lambda_binary);
}

template <>
void EQ<int, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                        Context *ctx) {
  auto eq_lambda_binary = [](int a, int b) { return (a == b) ? 1.f : 0.f; };
  traverse_binary<int>(in1, in2, out, eq_lambda_binary);
}

template <>
void Pow<float, lang::Cpp>(const Tensor &in, const float x, Tensor *out,
                           Context *ctx) {
  traverse_unary<float>(in, out, [x](float y) { return pow(y, x); });
}

template <>
void Pow<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                           Context *ctx) {
  auto pow_lambda_binary = [](float a, float b) { return pow(a, b); };
  traverse_binary<float>(in1, in2, out, pow_lambda_binary);
}

template <>
void ReLU<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto relu_lambda = [](float a) { return (a >= 0.f) ? a : 0.f; };
  traverse_unary<float>(in, out, relu_lambda);
}

template <>
void Set<float, lang::Cpp>(const float x, Tensor *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) outPtr[i] = x;
}

template <>
void Set<int, lang::Cpp>(const int x, Tensor *out, Context *ctx) {
  int *outPtr = static_cast<int *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) outPtr[i] = x;
}

template <>
void Set<half_float::half, lang::Cpp>(const half_float::half x, Tensor *out,
                                      Context *ctx) {
  half_float::half *outPtr =
      static_cast<half_float::half *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) outPtr[i] = x;
}

template <>
void Sigmoid<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto sigmoid_lambda = [](float a) { return 1.f / (1.f + exp(-a)); };
  traverse_unary<float>(in, out, sigmoid_lambda);
}

template <>
void Sign<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto sign_lambda = [](float a) { return (a > 0) - (a < 0); };
  traverse_unary<float>(in, out, sign_lambda);
}

template <>
void SoftPlus<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto softplus_lambda = [](float a) { return log(1.f + exp(a)); };
  traverse_unary<float>(in, out, softplus_lambda);
}

template <>
void SoftSign<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto softsign_lambda = [](float a) { return a / (1.f + fabs(a)); };
  traverse_unary<float>(in, out, softsign_lambda);
}

template <>
void Sqrt<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto usqrt = [](float a) {
    CHECK_GE(a, 0.f);
    return sqrt(a);
  };
  traverse_unary<float>(in, out, usqrt);
}

template <>
void Sub<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                           Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  auto sub_lambda_binary = [](float a, float b) { return (a - b); };
  traverse_binary<float>(in1, in2, out, sub_lambda_binary);
}

// sum all elements of input into out
// TODO(wangwei) optimize using omp
template <>
void Sum<float, lang::Cpp>(const Tensor &in, float *out, Context *ctx) {
  float s = 0.f;
  const float *inPtr = static_cast<const float *>(in.block()->data());
  for (size_t i = 0; i < in.Size(); i++) {
    s += inPtr[i];
  }
  *out = s;
}

#define GenUnaryTensorCppFn(fn, cppfn)                                     \
  template <>                                                              \
  void fn<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) { \
    auto fn_lambda = [](float a) { return cppfn(a); };                     \
    traverse_unary<float>(in, out, fn_lambda);                             \
  }

GenUnaryTensorCppFn(Cos, cos);
GenUnaryTensorCppFn(Cosh, cosh);
GenUnaryTensorCppFn(Acos, acos);
GenUnaryTensorCppFn(Acosh, acosh);
GenUnaryTensorCppFn(Sin, sin);
GenUnaryTensorCppFn(Sinh, sinh);
GenUnaryTensorCppFn(Asin, asin);
GenUnaryTensorCppFn(Asinh, asinh);
GenUnaryTensorCppFn(Tan, tan);
GenUnaryTensorCppFn(Tanh, tanh);
GenUnaryTensorCppFn(Atan, atan);
GenUnaryTensorCppFn(Atanh, atanh);

template <>
void Transform<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto identity = [](float a) { return a; };
  traverse_unary<float>(in, out, identity);
}

template <>
void Transform<int, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
  auto identity = [](int a) { return a; };
  traverse_unary<int>(in, out, identity);
}

template <>
void Transform<half_float::half, lang::Cpp>(const Tensor &in, Tensor *out,
                                            Context *ctx) {
  auto identity = [](half_float::half a) { return a; };
  traverse_unary<half_float::half>(in, out, identity);
}

template <>
void Bernoulli<float, lang::Cpp>(const float p, Tensor *out, Context *ctx) {
  std::bernoulli_distribution distribution(p);
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] = distribution(ctx->random_generator) ? 1.0f : 0.0f;
  }
}

template <>
void Gaussian<float, lang::Cpp>(const float mean, const float std, Tensor *out,
                                Context *ctx) {
  std::normal_distribution<float> distribution(mean, std);
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}

template <>
void Gaussian<half_float::half, lang::Cpp>(const half_float::half mean,
                                           const half_float::half std,
                                           Tensor *out, Context *ctx) {
  Tensor tmp(out->shape(), out->device(), kFloat32);
  Gaussian<float, lang::Cpp>(static_cast<float>(mean), static_cast<float>(std),
                             &tmp, ctx);
  CastCopy<float, half_float::half, lang::Cpp>(&tmp, out, ctx);
}

template <>
void Uniform<float, lang::Cpp>(const float low, const float high, Tensor *out,
                               Context *ctx) {
  std::uniform_real_distribution<float> distribution(low, high);
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] = static_cast<float>(distribution(ctx->random_generator));
  }
}

// ====================Blas operations======================================

// warning, this function has block M overwritting to block M itself
template <>
void DGMM<float, lang::Cpp>(const bool side_right, const Tensor &M,
                            const Tensor &v, Tensor *out, Context *ctx) {
  const float *MPtr = static_cast<const float *>(M.block()->data());
  const float *vPtr = static_cast<const float *>(v.block()->data());
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const size_t nrow = M.shape(0);
  const size_t ncol = M.shape(1);

  if (side_right) {
    for (size_t r = 0; r < nrow; r++) {
      size_t in_offset = M.stride()[0] * r, out_offset = out->stride()[0] * r;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[out_offset] = MPtr[in_offset] * vPtr[c];
        in_offset += M.stride()[1];
        out_offset += out->stride()[1];
      }
    }
  } else {
    for (size_t r = 0; r < nrow; r++) {
      size_t in_offset = M.stride()[0] * r, out_offset = out->stride()[0] * r;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[out_offset] = MPtr[in_offset] * vPtr[r];
        in_offset += M.stride()[1];
        out_offset += out->stride()[1];
      }
    }
  }
}

#ifdef USE_CBLAS
template <>
void Amax<float, lang::Cpp>(const Tensor &in, size_t *out, Context *ctx) {
  const float *inPtr = static_cast<const float *>(in.block()->data());
  *out = cblas_isamax(in.Size(), inPtr, 1);  // not using strided traversal
}

template <>
void Asum<float, lang::Cpp>(const Tensor &in, float *out, Context *ctx) {
  const float *inPtr = static_cast<const float *>(in.block()->data());
  *out = cblas_sasum(in.Size(), inPtr, 1);  // not using strided traversal
}

template <>
void Axpy<float, lang::Cpp>(const float alpha, const Tensor &in, Tensor *out,
                            Context *ctx) {
  // check input tensor for strides first
  const float *inPtr = static_cast<const float *>(in.block()->data());
  float *outPtr = static_cast<float *>(out->block()->mutable_data());

  if (in.stride() == out->stride()) {
    cblas_saxpy(in.Size(), alpha, inPtr, 1, outPtr, 1);
  } else {
    // LOG(FATAL) << "Axpy, input and output strides do not match." ;
    Tensor t(in.shape(), in.device(), in.data_type());
    EltwiseMult<float, lang::Cpp>(in, alpha, &t, ctx);
    float *tPtr = static_cast<float *>(t.block()->mutable_data());
    cblas_saxpy(in.Size(), 1, tPtr, 1, outPtr, 1);
  }
}

template <>
void Axpy<float, lang::Cpp>(const Tensor &alpha, const Tensor &in, Tensor *out,
                            Context *ctx) {
  // check input tensor for strides first
  const float *inPtr = static_cast<const float *>(in.block()->data());
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  const float a = *static_cast<const float *>(alpha.block()->data());

  if (in.stride() == out->stride()) {
    cblas_saxpy(in.Size(), a, inPtr, 1, outPtr, 1);
  } else {
    // LOG(FATAL) << "Axpy, input and output strides do not match." ;
    Tensor t(in.shape(), in.device(), in.data_type());
    EltwiseMult<float, lang::Cpp>(in, a, &t, ctx);
    float *tPtr = static_cast<float *>(t.block()->mutable_data());
    cblas_saxpy(in.Size(), 1, tPtr, 1, outPtr, 1);
  }
}

template <>
void Dot<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, float *out,
                           Context *ctx) {
  // check input tensor for strides first
  if (!(in1.transpose()) && !(in2.transpose())) {
    const float *in1Ptr = static_cast<const float *>(in1.block()->data());
    const float *in2Ptr = static_cast<const float *>(in2.block()->data());
    *out = cblas_sdot(in1.Size(), in1Ptr, 1, in2Ptr, 1);
  } else {
    LOG(FATAL) << "Dot, one of the input is tranposed. Not implemented yet.";
  }
}
template <>
void Dot<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, Tensor *out,
                           Context *ctx) {
  // check input tensor for strides first
  if (!(in1.transpose()) && !(in2.transpose())) {
    const float *in1Ptr = static_cast<const float *>(in1.block()->data());
    const float *in2Ptr = static_cast<const float *>(in2.block()->data());
    float *outPtr = static_cast<float *>(out->block()->mutable_data());
    *outPtr = cblas_sdot(in1.Size(), in1Ptr, 1, in2Ptr, 1);
  } else {
    LOG(FATAL) << "Dot, one of the input is tranposed. Not implemented yet.";
  }
}

template <>
void Scale<float, lang::Cpp>(const float x, Tensor *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  cblas_sscal(out->Size(), x, outPtr, 1);  // not using strided traversal
}

template <>
void Nrm2<float, lang::Cpp>(const Tensor &in, float *out, Context *ctx) {
  const float *inPtr = static_cast<const float *>(in.block()->data());
  *out = cblas_snrm2(in.Size(), inPtr, 1);  // not using strided traversal
}

template <>
void GEMV<float, lang::Cpp>(const float alpha, const Tensor &A, const Tensor &v,
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
void GEMM<float, lang::Cpp>(const float alpha, const Tensor &A, const Tensor &B,
                            const float beta, Tensor *C, Context *ctx) {
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

/*
 * implement matmul for 3d 4d tensor
 *   simulate cblas_sgemm_batch();
 *   which is only available in intel cblas
 */
template <>
void GEMMBatched<float, lang::Cpp>(const float alpha, const Tensor &A,
                                   const Tensor &B, const float beta, Tensor *C,
                                   Context *ctx) {
  const float *APtr = static_cast<const float *>(A.block()->data());
  const float *BPtr = static_cast<const float *>(B.block()->data());
  float *CPtr = static_cast<float *>(C->block()->mutable_data());

  auto transA = A.transpose();
  auto transa = transA ? CblasTrans : CblasNoTrans;
  auto transB = B.transpose();
  auto transb = transB ? CblasTrans : CblasNoTrans;

  const size_t ncolB = B.shape().end()[-1];
  const size_t nrowA = A.shape().end()[-2];
  const size_t ncolA = A.shape().end()[-1];

  auto lda = transA ? nrowA : ncolA;
  auto ldb = transB ? ncolA : ncolB;
  auto ldc = ncolB;
  const int group_count = 1;

  size_t group_size = A.shape()[0];                // 3d
  if (A.nDim() == 4u) group_size *= A.shape()[1];  // 4d

  auto matrix_stride_A = A.shape().end()[-1] * A.shape().end()[-2];
  auto matrix_stride_B = B.shape().end()[-1] * B.shape().end()[-2];
  auto matrix_stride_C = C->shape().end()[-1] * C->shape().end()[-2];
  auto offset_A = 0;
  auto offset_B = 0;
  auto offset_C = 0;

  for (int i = 0; i < group_size; i++) {
    cblas_sgemm(CblasRowMajor, transa, transb, nrowA, ncolB, ncolA, alpha,
                APtr + offset_A, lda, BPtr + offset_B, ldb, beta,
                CPtr + offset_C, ldc);
    offset_A += matrix_stride_A;
    offset_B += matrix_stride_B;
    offset_C += matrix_stride_C;
  }
}

#else

template <>
void Amax<float, lang::Cpp>(const Tensor &in, size_t *out, Context *ctx) {
  size_t maxPos = 0;
  float maxVal = 0;
  const float *inPtr = static_cast<const float *>(in.block()->data());
  for (size_t i = 0; i < in.Size(); i++) {  // not using strided traversal
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
void Amin<float, lang::Cpp>(const Tensor &in, size_t *out, Context *ctx) {
  size_t minPos = 0;
  float minVal = 0;
  const float *inPtr = static_cast<const float *>(in.block()->data());
  for (size_t i = 0; i < in.Size(); i++) {  // not using strided traversal
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
void Asum<float, lang::Cpp>(const Tensor &in, float *out, Context *ctx) {
  float sum = 0;
  const float *inPtr = static_cast<const float *>(in.block()->data());
  for (size_t i = 0; i < in.Size(); i++) {
    sum += fabs(inPtr[i]);  // not using strided traversal
  }
}

template <>
void Axpy<float, lang::Cpp>(const float alpha, const Tensor &in, Tensor *out,
                            Context *ctx) {
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
void Scale<float, lang::Cpp>(const float x, Tensor *out, Context *ctx) {
  float *outPtr = static_cast<float *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] *= x;  // not using strided traversal
  }
}

template <>
void Dot<float, lang::Cpp>(const Tensor &in1, const Tensor &in2, float *out,
                           Context *ctx) {
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
    sum += in1Ptr[traversal_info_in1[in1.shape().size()]] *
           in2Ptr[traversal_info_in2[in2.shape().size()]];
    traverse_next(in1, shape_multipliers_in1, traversal_info_in1, i + 1);
    traverse_next(in2, shape_multipliers_in2, traversal_info_in2, i + 1);
  }
}

template <>
void GEMV<float, lang::Cpp>(const float alpha, const Tensor &A, const Tensor &v,
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
                                           const size_t dim, const Tensor &p,
                                           const Tensor &t, Tensor *loss,
                                           Context *ctx) {
  const float *pPtr = static_cast<const float *>(p.block()->data());
  const int *tPtr = static_cast<const int *>(t.block()->data());
  float *lossPtr = static_cast<float *>(loss->block()->mutable_data());
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
      float loss_value = 0.f;
      for (size_t j = 0, offset = i * dim; j < dim; j++, offset++) {
        loss_value -=
            tPtr[offset] / sum * std::log((std::max)(pPtr[offset], FLT_MIN));
      }
      lossPtr[i] = loss_value;
    }
  }
}

template <>
void SoftmaxCrossEntropyBwd<float, lang::Cpp>(bool int_target,
                                              const size_t batchsize,
                                              const size_t dim, const Tensor &p,
                                              const Tensor &t, Tensor *grad,
                                              Context *ctx) {
  CHECK_EQ(p.block(), grad->block())
      << "Use the same pointer to optimize performance";
  // const float* pPtr = static_cast<const float*>(p->data());
  const int *tPtr = static_cast<const int *>(t.block()->data());
  float *gradPtr = static_cast<float *>(grad->block()->mutable_data());

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
void RowMax<float, lang::Cpp>(const Tensor &in, Tensor *out, Context *ctx) {
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
      traverse_next(in, shape_multipliers, traversal_info,
                    counter_offset + c + 1);
    }
    outPtr[r] = maxval;
  }
}

}  // namespace singa

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_CPP_H_
