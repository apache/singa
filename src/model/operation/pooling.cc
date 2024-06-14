/*********************************************************
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 ************************************************************/
#include "pooling.h"

#include <cmath>

namespace singa {

PoolingHandle::PoolingHandle(const Tensor &input,
                             const std::vector<int> &kernel_size,
                             const std::vector<int> &stride,
                             const std::vector<int> &padding,
                             const bool is_max) {
  kernel_h = kernel_size[0];
  kernel_w = kernel_size[1];

  pad_h = padding[0];
  pad_w = padding[1];

  stride_h = stride[0];
  stride_w = stride[1];

  batchsize = input.shape(0);
  channels = input.shape(1);
  height = input.shape(2);
  width = input.shape(3);

  pooled_height = 1;

  if (stride_h > 0)
    pooled_height =
        std::floor(((height + 2 * pad_h - kernel_h) / stride_h)) + 1;
  pooled_width = std::floor(((width + 2 * pad_w - kernel_w) / stride_w)) + 1;
  is_max_pooling = is_max;

#ifdef USE_DNNL
  if (input.device()->lang() == kCpp) {
    auto x_dims =
        dnnl::memory::dims(input.shape().begin(), input.shape().end());
    auto y_dims =
        dnnl::memory::dims({batchsize, channels, pooled_height, pooled_width});
    auto s_dims = dnnl::memory::dims(stride.begin(), stride.end());
    auto k_dims = dnnl::memory::dims(kernel_size.begin(), kernel_size.end());

    auto p_dims = dnnl::memory::dims(padding.begin(), padding.end());

    auto dtype_ = dnnl::memory::data_type::f32;
    auto format_tag_ = get_dnnl_format_tag(input);
    x_md = dnnl::memory::desc({x_dims}, dtype_, format_tag_);
    y_md = dnnl::memory::desc({y_dims}, dtype_, format_tag_);

    // allow max or avg (follow cudnn implementation convention)
    auto pooling_algo = dnnl::algorithm::pooling_avg_exclude_padding;
    if (is_max_pooling) pooling_algo = dnnl::algorithm::pooling_max;

    auto pool_fwd_d = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_training, pooling_algo, x_md, y_md, s_dims,
        k_dims, p_dims, p_dims);
    auto pool_bwd_d = dnnl::pooling_backward::desc(
        pooling_algo, x_md, y_md, s_dims, k_dims, p_dims, p_dims);

    auto eng = input.device()->context(0)->dnnl_engine;
    pool_fwd_pd = dnnl::pooling_forward::primitive_desc(pool_fwd_d, eng);
    pool_bwd_pd =
        dnnl::pooling_backward::primitive_desc(pool_bwd_d, eng, pool_fwd_pd);

    auto ws_md = pool_fwd_pd.workspace_desc();
    ws_mem = dnnl::memory(ws_md, eng);
  }
#endif  // USE_DNNL
}

PoolingHandle::~PoolingHandle() {}

#ifdef USE_DNNL

Tensor CpuPoolingForward(const PoolingHandle &ph, const Tensor &x) {
  CHECK_EQ(x.device()->lang(), kCpp);
  Tensor y({(unsigned long)ph.batchsize, (unsigned long)ph.channels,
            (unsigned long)ph.pooled_height, (unsigned long)ph.pooled_width},
           x.device(), x.data_type());

  y.device()->Exec(
      [y, x, &ph](Context *ctx) mutable {
        auto eng = ctx->dnnl_engine;
        using namespace dnnl;

        memory x_mem(ph.x_md, eng, x.block()->mutable_data());
        memory y_mem(ph.y_md, eng, y.block()->mutable_data());

        pooling_forward(ph.pool_fwd_pd)
            .execute(ctx->dnnl_stream, {{DNNL_ARG_SRC, x_mem},
                                        {DNNL_ARG_DST, y_mem},
                                        {DNNL_ARG_WORKSPACE, ph.ws_mem}});
        ctx->dnnl_stream.wait();
      },
      {x.block()}, {y.block()}, "CpuPoolingForward");

  return y;
}

Tensor CpuPoolingBackward(const PoolingHandle &ph, const Tensor &grad,
                          const Tensor &x, const Tensor &y) {
  CHECK_EQ(x.device()->lang(), kCpp);
  CHECK_EQ(grad.device()->lang(), kCpp);
  CHECK_EQ(y.device()->lang(), kCpp);
  Tensor in_grad;
  in_grad.ResetLike(x);

  in_grad.device()->Exec(
      [x, y, in_grad, grad, &ph](Context *ctx) mutable {
        auto eng = ctx->dnnl_engine;
        using namespace dnnl;

        memory dx_mem(ph.x_md, eng, in_grad.block()->mutable_data());
        memory dy_mem(ph.y_md, eng, grad.block()->mutable_data());

        pooling_backward(ph.pool_bwd_pd)
            .execute(ctx->dnnl_stream, {{DNNL_ARG_DIFF_DST, dy_mem},
                                        {DNNL_ARG_DIFF_SRC, dx_mem},
                                        {DNNL_ARG_WORKSPACE, ph.ws_mem}});
        ctx->dnnl_stream.wait();
      },
      {x.block(), y.block(), grad.block()}, {in_grad.block()},
      "CpuPoolingBackward");

  return in_grad;
}
#endif  // USE_DNNL

#ifdef USE_CUDNN

CudnnPoolingHandle::CudnnPoolingHandle(const Tensor &input,
                                       const std::vector<int> &kernel_size,
                                       const std::vector<int> &stride,
                                       const std::vector<int> &padding,
                                       const bool is_max)
    : PoolingHandle(input, kernel_size, stride, padding, is_max) {
  // nan_prop = CUDNN_NOT_PROPAGATE_NAN;

  DataType dtype = input.data_type();

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize,
                                         channels, height, width));
  // LOG(ERROR) << batchsize << " " << channels << " " << pooled_height << " "
  // << pooled_width;
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      y_desc, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize, channels,
      pooled_height, pooled_width));
  auto pool_method = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  if (is_max) pool_method = CUDNN_POOLING_MAX;

  CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc, pool_method, nan_prop,
                                          kernel_h, kernel_w, pad_h, pad_w,
                                          stride_h, stride_w));
};

CudnnPoolingHandle::~CudnnPoolingHandle() {
  if (pool_desc != nullptr)
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
  if (x_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  if (y_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
}

Tensor GpuPoolingForward(const CudnnPoolingHandle &cph, const Tensor &x) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(x.nDim(), 4u);

  Tensor output = Tensor(
      Shape({cph.batchsize, cph.channels, cph.pooled_height, cph.pooled_width}),
      x.device(), x.data_type());

  output.device()->Exec(
      [output, x, &cph](Context *ctx) mutable {
        float alpha = 1.0f, beta = 0.0f;
        cudnnPoolingForward(ctx->cudnn_handle, cph.pool_desc, &alpha,
                            cph.x_desc, x.block()->data(), &beta, cph.y_desc,
                            output.block()->mutable_data());
      },
      {x.block()}, {output.block()}, "GpuPoolingForward");

  return output;
}

Tensor GpuPoolingBackward(const CudnnPoolingHandle &cph, const Tensor &dy,
                          const Tensor &x, const Tensor &y) {
  CHECK_EQ(dy.device()->lang(), kCuda);
  CHECK_EQ(dy.nDim(), 4u);

  Tensor dx;
  dx.ResetLike(x);

  dx.device()->Exec(
      [dx, dy, x, y, &cph](Context *ctx) mutable {
        float alpha = 1.0f, beta = 0.0f;
        cudnnPoolingBackward(ctx->cudnn_handle, cph.pool_desc, &alpha,
                             cph.y_desc, y.block()->data(), cph.y_desc,
                             dy.block()->data(), cph.x_desc, x.block()->data(),
                             &beta, cph.x_desc, dx.block()->mutable_data());
      },
      {dy.block(), y.block(), x.block()}, {dx.block()}, "GpuPoolingBackward");

  return dx;
};
#endif  // USE_CUDNN

}  // namespace singa
