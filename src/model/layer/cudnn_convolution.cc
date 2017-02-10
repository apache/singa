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
RegisterLayerClass(cudnn_convolution, CudnnConvolution);
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

void CudnnConvolution::Setup(const Shape& in_sample, const LayerConf &conf) {
  Convolution::Setup(in_sample, conf);
  ConvolutionConf conv_conf = conf.convolution_conf();
  // convert MB to bytes
  workspace_byte_limit_ = conv_conf.workspace_byte_limit() << 20;
  prefer_ = ToLowerCase(conv_conf.prefer());
  CHECK(prefer_ == "fastest" || prefer_ == "limited_workspace" ||
        prefer_ == "no_workspace" || prefer_ == "autotune")
      << "CudnnConvolution only supports four algorithm preferences: fastest, "
         "limited_workspace, no_workspace and autotune";
}

void CudnnConvolution::ToDevice(std::shared_ptr<Device> device) {
  Convolution::ToDevice(device);
  workspace_.ToDevice(device);
}

void CudnnConvolution::InitCudnn(const Tensor &input) {
  CHECK(!has_init_cudnn_);
  DataType dtype = input.data_type();
  auto dev = input.device();
  Context *ctx = dev->context(0);
  size_t batchsize = input.shape(0);
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  if (bias_term_)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize,
                                         channels_, height_, width_));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      y_desc_, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize,
      num_filters_, conv_height_, conv_width_));
  if (bias_term_)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW,
                                           GetCudnnDataType(dtype), 1,
                                           num_filters_, 1, 1));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_, pad_h_, pad_w_,
                                              stride_h_, stride_w_, 1, 1,
                                              CUDNN_CROSS_CORRELATION));
#if CUDNN_MAJOR == 5
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, GetCudnnDataType(dtype),
                                         CUDNN_TENSOR_NCHW, num_filters_,
                                         channels_, kernel_h_, kernel_w_));
#elif CUDNN_MAJOR == 4
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(
      filter_desc_, GetCudnnDataType(dtype), CUDNN_TENSOR_NCHW, num_filters_,
      channels_, kernel_h_, kernel_w_));
#else
  LOG(FATAL) << "Not supported CUDNN version = " << CUDNN_MAJOR;
#endif

  if (prefer_ == "fastest" || prefer_ == "limited_workspace" ||
      prefer_ == "no_workspace") {
    cudnnConvolutionFwdPreference_t fwd_pref;
    cudnnConvolutionBwdFilterPreference_t bwd_filt_pref;
    cudnnConvolutionBwdDataPreference_t bwd_data_pref;
    if (prefer_ == "fastest") {
      fwd_pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
      bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
      bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
    } else if (prefer_ == "limited_workspace") {
      fwd_pref = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
      bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
      bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
    } else {
      fwd_pref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
      bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
      bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
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
  } else if (prefer_ == "autotune") {
    const int topk = 1;
    int num_fp_alg, num_bp_filt_alg, num_bp_data_alg;
    cudnnConvolutionFwdAlgoPerf_t fp_alg_perf[topk];
    cudnnConvolutionBwdFilterAlgoPerf_t bp_filt_perf[topk];
    cudnnConvolutionBwdDataAlgoPerf_t bp_data_perf[topk];
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        ctx->cudnn_handle, x_desc_, filter_desc_, conv_desc_, y_desc_, topk,
        &num_fp_alg, fp_alg_perf));
    fp_alg_ = fp_alg_perf[0].algo;
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
        ctx->cudnn_handle, x_desc_, y_desc_, conv_desc_, filter_desc_, topk,
        &num_bp_filt_alg, bp_filt_perf));
    bp_filter_alg_ = bp_filt_perf[0].algo;
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
        ctx->cudnn_handle, filter_desc_, y_desc_, conv_desc_, x_desc_, topk,
        &num_bp_data_alg, bp_data_perf));
    bp_data_alg_ = bp_data_perf[0].algo;
  } else {
    LOG(FATAL) << "Preferred algorithm is not available!";
  }

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
  if (workspace_count_ > workspace_byte_limit_)
    LOG(WARNING) << "The required memory for workspace ("
      << workspace_count_ * sizeof(float)
      << ") is larger than the expected Bytes ("
      << workspace_byte_limit_ << ")";
  workspace_ = Tensor(Shape{workspace_count_}, dev, dtype);
  has_init_cudnn_ = true;
}

const Tensor CudnnConvolution::Forward(int flag, const Tensor &input) {
  CHECK(buf_.empty());
  CHECK_EQ(input.device()->lang(), kCuda);
  CHECK_EQ(input.nDim(), 4u);
  if (flag & kTrain) buf_.push(input);  // buffer the input for backward
  size_t batchsize = input.shape()[0];
  DataType dtype = input.data_type();
  auto dev = input.device();

  if (!has_init_cudnn_) InitCudnn(input);

  Shape shape{batchsize, num_filters_, conv_height_, conv_width_};
  Tensor output(shape, dev, dtype);
  output.device()->Exec([input, output, this](Context *ctx) {
    Block *inblock = input.block(), *outblock = output.block(),
          *wblock = this->weight_.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionForward(ctx->cudnn_handle, &alpha, this->x_desc_,
                            inblock->data(), this->filter_desc_, wblock->data(),
                            this->conv_desc_, this->fp_alg_,
                            this->workspace_.block()->mutable_data(),
                            this->workspace_count_ * sizeof(float), &beta,
                            this->y_desc_, outblock->mutable_data());
  }, {input.block(), weight_.block()}, {output.block()}, workspace_.block());

  if (bias_term_) {
    output.device()->Exec([output, this](Context *ctx) {
      float beta = 1.f, alpha = 1.0f;
      Block *outblock = output.block(), *bblock = this->bias_.block();
      cudnnAddTensor(ctx->cudnn_handle, &alpha, this->bias_desc_,
                     bblock->data(), &beta, this->y_desc_,
                     outblock->mutable_data());
    }, {output.block(), bias_.block()}, {output.block()});
  }
  return output;
}

const std::pair<Tensor, vector<Tensor>> CudnnConvolution::Backward(
    int flag, const Tensor &grad) {
  CHECK(has_init_cudnn_);
  CHECK_EQ(grad.device()->lang(), kCuda);
  CHECK_EQ(grad.nDim(), 4u);
  CHECK(!buf_.empty());
  Tensor src_data = buf_.top();
  buf_.pop();
  vector<Tensor> param_grad;
  Tensor dx;
  dx.ResetLike(src_data);
  Tensor db, dw;
  dw.ResetLike(weight_);

  // LOG(ERROR) << "backward bias";
  if (bias_term_) {
    db.ResetLike(bias_);
    dx.device()->Exec([grad, db, this](Context *ctx) {
      Block *dyblock = grad.block(), *dbblock = db.block();
      float alpha = 1.f, beta = 0.f;
      cudnnConvolutionBackwardBias(ctx->cudnn_handle, &alpha, this->y_desc_,
                                   dyblock->data(), &beta, this->bias_desc_,
                                   dbblock->mutable_data());
    }, {grad.block()}, {db.block()});
  }
  // LOG(ERROR) << "backward w";
  dx.device()->Exec([grad, dw, src_data, this](Context *ctx) {
    Block *inblock = src_data.block(), *dyblock = grad.block(),
          *dwblock = dw.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardFilter(
        ctx->cudnn_handle, &alpha, this->x_desc_, inblock->data(),
        this->y_desc_, dyblock->data(), this->conv_desc_, this->bp_filter_alg_,
        this->workspace_.block()->mutable_data(),
        this->workspace_count_ * sizeof(float), &beta, this->filter_desc_,
        dwblock->mutable_data());
  }, {grad.block(), src_data.block()}, {dw.block(), workspace_.block()});

  // LOG(ERROR) << "backward src";
  dx.device()->Exec([dx, grad, this](Context *ctx) {
    Block *wblock = this->weight_.block(), *dyblock = grad.block(),
          *dxblock = dx.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardData(ctx->cudnn_handle, &alpha, this->filter_desc_,
                                 wblock->data(), this->y_desc_, dyblock->data(),
                                 this->conv_desc_, this->bp_data_alg_,
                                 this->workspace_.block()->mutable_data(),
                                 this->workspace_count_ * sizeof(float), &beta,
                                 this->x_desc_, dxblock->mutable_data());
  }, {grad.block(), weight_.block()}, {dx.block(), workspace_.block()});
  param_grad.push_back(dw);
  if (bias_term_)
    param_grad.push_back(db);
  return std::make_pair(dx, param_grad);
}

}  // namespace singa
#endif  // USE_CUDNN
