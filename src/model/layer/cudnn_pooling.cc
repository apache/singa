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
CudnnPooling::~CudnnPooling() {
  if (pool_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
  if (x_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_));
  if (y_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_));
}

void CudnnPooling::Setup(const LayerConf &conf) {
  Pooling::Setup(conf);
  PoolingConf pool_conf = conf.pooling_conf();
  if (pool_conf.nan_prop())
    nan_prop_ = CUDNN_PROPAGATE_NAN;
  else
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
}

void CudnnPooling::InitCudnn(DataType dtype) {
  CHECK(!has_init_cudnn_);
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize_,
                                         channels_, height_, width_));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      y_desc_, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize_,
      channels_, pooled_height_, pooled_width_));
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
  CHECK_EQ(input.device()->lang(), kCuda);
  CHECK_EQ(input.shape().size(), 4);
  buf_.push(input);
  batchsize_ = input.shape()[0];
  DataType dtype = input.data_type();
  Device *dev = input.device();
  float alpha = 1.0f, beta = 0.0f;
  if (!has_init_cudnn_) InitCudnn(dtype);

  Shape shape{batchsize_, channels_, pooled_height_, pooled_width_};
  Tensor output = Tensor(shape, dev, dtype);
  output.device()->Exec(
      [input, output, alpha, beta, this](Context *ctx) {
        Blob *inblob = input.blob(), *outblob = output.blob();
        cudnnPoolingForward(ctx->cudnn_handle, this->pool_desc_, &alpha,
                            this->x_desc_, inblob->data(), &beta, this->y_desc_,
                            outblob->mutable_data());
      },
      {input.blob()}, {output.blob()});
  buf_.push(output);
  return output;
}

const std::pair<Tensor, vector<Tensor>> CudnnPooling::Backward(
    int flag, const Tensor &grad) {
  CHECK_EQ(grad.device()->lang(), kCuda);
  CHECK_EQ(grad.shape().size(), 4);
  vector<Tensor> param_grad;
  Tensor dx;
  Tensor data = buf_.top();
  buf_.pop();
  Tensor src_data = buf_.top();
  buf_.pop();
  dx.ResetLike(src_data);

  float alpha = 1.0f, beta = 0.0f;
  dx.device()->Exec(
      [dx, grad, src_data, data, alpha, beta, this](Context *ctx) {
        Blob *dyblob = grad.blob(), *dxblob = dx.blob(),
             *yblob = data.blob(), *xblob = src_data.blob();
        cudnnPoolingBackward(ctx->cudnn_handle, this->pool_desc_, &alpha,
                             this->y_desc_, yblob->data(), this->y_desc_,
                             dyblob->data(), this->x_desc_, xblob->data(),
                             &beta, this->x_desc_, dxblob->mutable_data());
      },
      {grad.blob(), data.blob(), src_data.blob()}, {dx.blob()});

  return std::make_pair(dx, param_grad);
}
}  // namespace singa
#endif  // USE_CUDNN
