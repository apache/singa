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
#include "./cudnn_convolution.h"
#ifdef USE_CUDNN
#include <cudnn.h>
#include <chrono>
#include "./cudnn_utils.h"
#include "singa/utils/logging.h"

namespace singa {
CudnnConvolution::~CudnnConvolution() {
  if (bias_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
  if (filter_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
  if (conv_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  if (x_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_));
  if (y_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_));
}

void CudnnConvolution::Setup(const LayerConf &conf) {
  Convolution::Setup(conf);
  ConvolutionConf conv_conf = conf.convolution_conf();
  // convert MB to bytes
  workspace_byte_limit_ = conv_conf.workspace_byte_limit() << 20;
  pref_ = conv_conf.algo_pref();
  CHECK(pref_ == "fastest" || pref_ == "limited_workspace" ||
        pref_ == "no_workspace")
      << "CudnnConvolution only supports three algorithm preferences: fastest, "
         "limited_workspace and no_workspace";
}

void CudnnConvolution::ToDevice(Device *device) {
  weight_.ToDevice(device);
  bias_.ToDevice(device);
  workspace_.ToDevice(device);
}

void CudnnConvolution::InitCudnn(DataType dtype, Device *dev, Context *ctx) {
  CHECK(!has_init_cudnn_);
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize_,
                                         channels_, height_, width_));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      y_desc_, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize_,
      num_filters_, conv_height_, conv_width_));
  if (bias_term_)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW,
                                           GetCudnnDataType(dtype), 1, 1,
                                           num_filters_, 1));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_, pad_h_, pad_w_,
                                              stride_h_, stride_w_, 1, 1,
                                              CUDNN_CROSS_CORRELATION));
#if CUDNN_VERSION_MAJOR == 5
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, GetCudnnDataType(dtype),
                                         CUDNN_TENSOR_NCHW, num_filters_,
                                         channels_, kernel_h_, kernel_w_));
#elif CUDNN_VERSION_MAJOR == 4
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(
      filter_desc_, GetCudnnDataType(dtype), CUDNN_TENSOR_NCHW, num_filters_,
      channels_, kernel_h_, kernel_w_));
#else
  LOG(FATAL) << "Not supported CUDNN version = " << CUDNN_VERSION_MAJOR;
#endif

  cudnnConvolutionFwdPreference_t fwd_pref;
  cudnnConvolutionBwdFilterPreference_t bwd_filt_pref;
  cudnnConvolutionBwdDataPreference_t bwd_data_pref;
  if (pref_ == "fastest") {
    fwd_pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
    bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
  } else if (pref_ == "limited_workspace") {
    fwd_pref = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
    bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
    bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
  } else if (pref_ == "no_workspace") {
    fwd_pref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
    bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
    bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
  } else {
    LOG(FATAL) << "Algorithm preference is not implemented!";
  }
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
      ctx->cudnn_handle, x_desc_, filter_desc_, conv_desc_, y_desc_, fwd_pref,
      workspace_byte_limit_, &fp_alg_));

  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
      ctx->cudnn_handle, x_desc_, y_desc_, conv_desc_, filter_desc_,
      bwd_filt_pref, workspace_byte_limit_, &bp_filter_alg_));
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
      ctx->cudnn_handle, filter_desc_, y_desc_, conv_desc_, x_desc_,
      bwd_data_pref, workspace_byte_limit_, &bp_data_alg_));

  size_t fp_byte, bp_data_byte, bp_filter_byte;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      ctx->cudnn_handle, x_desc_, filter_desc_, conv_desc_, y_desc_, fp_alg_,
      &fp_byte));
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
      ctx->cudnn_handle, filter_desc_, y_desc_, conv_desc_, x_desc_,
      bp_data_alg_, &bp_data_byte));
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      ctx->cudnn_handle, x_desc_, y_desc_, conv_desc_, filter_desc_,
      bp_filter_alg_, &bp_filter_byte));
  workspace_count_ = std::max(std::max(fp_byte, bp_data_byte), bp_filter_byte) /
                         sizeof(float) +
                     1;
  workspace_ = Tensor(Shape{workspace_count_}, dev, dtype);
  has_init_cudnn_ = true;
}

const Tensor CudnnConvolution::Forward(int flag, const Tensor &input) {
  CHECK_EQ(input.device()->lang(), kCuda);
  CHECK_EQ(input.shape().size(), 4);
  buf_.push(input);
  batchsize_ = input.shape()[0];
  DataType dtype = input.data_type();
  Device *dev = input.device();

  if (!has_init_cudnn_) InitCudnn(dtype, dev, dev->context(0));

  Shape shape{batchsize_, num_filters_, conv_height_, conv_width_};
  Tensor output(shape, dev, dtype);
  float alpha = 1.f, beta = 0.f;
  output.device()->Exec(
      [input, output, alpha, beta, this](Context *ctx) {
        Blob *inblob = input.blob(), *outblob = output.blob(),
             *wblob = this->weight_.blob();
        cudnnConvolutionForward(ctx->cudnn_handle, &alpha, this->x_desc_,
                                inblob->data(), this->filter_desc_,
                                wblob->data(), this->conv_desc_, this->fp_alg_,
                                this->workspace_.blob()->mutable_data(),
                                this->workspace_count_ * sizeof(float), &beta,
                                this->y_desc_, outblob->mutable_data());
      },
      {input.blob(), weight_.blob()}, {output.blob()}, workspace_.blob());

  if (bias_term_) {
    beta = 1.f;
    output.device()->Exec(
        [output, alpha, beta, this](Context *ctx) {
          Blob *outblob = output.blob(), *bblob = this->bias_.blob();
          cudnnAddTensor(ctx->cudnn_handle, &alpha, this->bias_desc_,
                         bblob->data(), &beta, this->y_desc_,
                         outblob->mutable_data());
        },
        {output.blob(), bias_.blob()}, {output.blob()});
  }
  return output;
}

const std::pair<Tensor, vector<Tensor>> CudnnConvolution::Backward(
    int flag, const Tensor &grad) {
  CHECK_EQ(grad.device()->lang(), kCuda);
  CHECK_EQ(grad.shape().size(), 4);
  Tensor src_data = buf_.top();
  buf_.pop();
  float alpha = 1.f, beta = 0.f;
  vector<Tensor> param_grad;
  Tensor dx;
  dx.ResetLike(src_data);
  Tensor db, dw;
  db.ResetLike(bias_);
  dw.ResetLike(weight_);

  // LOG(ERROR) << "backward bias";
  if (bias_term_) {
    dx.device()->Exec(
        [grad, db, alpha, beta, this](Context *ctx) {
          Blob *dyblob = grad.blob(), *dbblob = db.blob();
          cudnnConvolutionBackwardBias(ctx->cudnn_handle, &alpha, this->y_desc_,
                                       dyblob->data(), &beta, this->bias_desc_,
                                       dbblob->mutable_data());
        },
        {grad.blob()}, {db.blob()});
  }
  // LOG(ERROR) << "backward w";
  dx.device()->Exec(
      [grad, dw, src_data, alpha, beta, this](Context *ctx) {
        Blob *inblob = src_data.blob(), *dyblob = grad.blob(),
             *dwblob = dw.blob();
        cudnnConvolutionBackwardFilter(
            ctx->cudnn_handle, &alpha, this->x_desc_, inblob->data(),
            this->y_desc_, dyblob->data(), this->conv_desc_,
            this->bp_filter_alg_, this->workspace_.blob()->mutable_data(),
            this->workspace_count_ * sizeof(float), &beta, this->filter_desc_,
            dwblob->mutable_data());
      },
      {grad.blob(), src_data.blob()}, {dw.blob(), workspace_.blob()});

  // LOG(ERROR) << "backward src";
  dx.device()->Exec(
      [dx, grad, alpha, beta, this](Context *ctx) {
        Blob *wblob = this->weight_.blob(), *dyblob = grad.blob(),
             *dxblob = dx.blob();
        cudnnConvolutionBackwardData(
            ctx->cudnn_handle, &alpha, this->filter_desc_, wblob->data(),
            this->y_desc_, dyblob->data(), this->conv_desc_, this->bp_data_alg_,
            this->workspace_.blob()->mutable_data(),
            this->workspace_count_ * sizeof(float), &beta, this->x_desc_,
            dxblob->mutable_data());
      },
      {grad.blob(), weight_.blob()}, {dx.blob(), workspace_.blob()});
  param_grad.push_back(dw);
  param_grad.push_back(db);
  return std::make_pair(dx, param_grad);
}

}  // namespace singa
#endif  // USE_CUDNN
