/*
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
#include "./cudnn_pooling.h"
#ifdef USE_CUDNN

#include <cudnn.h>
#include <chrono>

#include "./cudnn_utils.h"
#include "singa/utils/logging.h"

namespace singa {
RegisterLayerClass(CudnnPooling);
CudnnPooling::~CudnnPooling() {
  if (pool_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
  if (x_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_));
  if (y_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_));
}

void CudnnPooling::Setup(const Shape& in_sample, const LayerConf &conf) {
  Pooling::Setup(in_sample, conf);
  PoolingConf pool_conf = conf.pooling_conf();
  if (pool_conf.nan_prop())
    nan_prop_ = CUDNN_PROPAGATE_NAN;
  else
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
}

void CudnnPooling::InitCudnn(const Tensor &input) {
  CHECK(!has_init_cudnn_);
  DataType dtype = input.data_type();
  size_t batchsize = input.shape(0);
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize,
                                         channels_, height_, width_));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      y_desc_, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize, channels_,
      pooled_height_, pooled_width_));
  auto pool_method = CUDNN_POOLING_MAX;
  if (pool_ == PoolingConf_PoolMethod_MAX)
    pool_method = CUDNN_POOLING_MAX;
  else if (pool_ == PoolingConf_PoolMethod_AVE)
    pool_method = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  else
    LOG(FATAL) << "Not implemented!";

#if CUDNN_VERSION_MAJOR == 5
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc_, pool_method, nan_prop_,
                                          kernel_h_, kernel_w_, pad_h_, pad_w_,
                                          stride_h_, stride_w_));
#elif CUDNN_VERSION_MAJOR == 4
  CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(pool_desc_, pool_method, nan_prop_,
                                             kernel_h_, kernel_w_, pad_h_,
                                             pad_w_, stride_h_, stride_w_));
#else
  LOG(FATAL) << "Not supported CUDNN version = " << CUDNN_VERSION_MAJOR;
#endif
  has_init_cudnn_ = true;
}

const Tensor CudnnPooling::Forward(int flag, const Tensor &input) {
  CHECK(buf_.empty());
  CHECK_EQ(input.device()->lang(), kCuda);
  CHECK_EQ(input.nDim(), 4u);
  size_t batchsize = input.shape(0);
  DataType dtype = input.data_type();
  auto dev = input.device();
  if (!has_init_cudnn_) InitCudnn(input);

  Shape shape{batchsize, channels_, pooled_height_, pooled_width_};
  Tensor output = Tensor(shape, dev, dtype);
  output.device()->Exec([input, output, this](Context *ctx) {
    Block *inblock = input.block(), *outblock = output.block();
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingForward(ctx->cudnn_handle, this->pool_desc_, &alpha,
                        this->x_desc_, inblock->data(), &beta, this->y_desc_,
                        outblock->mutable_data());
  }, {input.block()}, {output.block()});
  if (flag & kTrain) {
    buf_.push(input);
    buf_.push(output);
  }
  return output;
}

const std::pair<Tensor, vector<Tensor>> CudnnPooling::Backward(
    int flag, const Tensor &grad) {
  CHECK_EQ(grad.device()->lang(), kCuda);
  CHECK_EQ(grad.nDim(), 4u);
  vector<Tensor> param_grad;
  CHECK(!buf_.empty());
  Tensor y = buf_.top();
  buf_.pop();
  Tensor x = buf_.top();
  buf_.pop();
  Tensor dx;
  dx.ResetLike(x);

  dx.device()->Exec([dx, grad, x, y, this](Context *ctx) {
    Block *dyblock = grad.block(), *dxblock = dx.block(), *yblock = y.block(),
          *xblock = x.block();
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingBackward(ctx->cudnn_handle, this->pool_desc_, &alpha,
                         this->y_desc_, yblock->data(), this->y_desc_,
                         dyblock->data(), this->x_desc_, xblock->data(), &beta,
                         this->x_desc_, dxblock->mutable_data());
  }, {grad.block(), y.block(), x.block()}, {dx.block()});

  return std::make_pair(dx, param_grad);
}
}  // namespace singa
#endif  // USE_CUDNN
