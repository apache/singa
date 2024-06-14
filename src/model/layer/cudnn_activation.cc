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
#include "singa/singa_config.h"
#ifdef USE_CUDNN
#include <cudnn.h>

#include "./cudnn_activation.h"
#include "./cudnn_utils.h"
#include "singa/core/common.h"
#include "singa/utils/logging.h"

namespace singa {
RegisterLayerClass(cudnn_relu, CudnnActivation);
RegisterLayerClass(cudnn_sigmoid, CudnnActivation);
RegisterLayerClass(cudnn_tanh, CudnnActivation);
CudnnActivation::~CudnnActivation() {
  if (acti_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(acti_desc_));
  if (desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
}

void CudnnActivation::InitCudnn(size_t size, DataType dtype) {
  if (!has_init_cudnn_) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&acti_desc_));

    if (mode_ == "sigmoid")
      cudnn_mode_ = CUDNN_ACTIVATION_SIGMOID;
    else if (mode_ == "tanh")
      cudnn_mode_ = CUDNN_ACTIVATION_TANH;
    else if (mode_ == "relu")
      cudnn_mode_ = CUDNN_ACTIVATION_RELU;
    else
      LOG(FATAL) << "Unkown activation: " << mode_;

    CUDNN_CHECK(cudnnSetActivationDescriptor(acti_desc_, cudnn_mode_,
                                             CUDNN_PROPAGATE_NAN, 0.0f));
  }

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      desc_, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), 1, 1, 1, size));

  has_init_cudnn_ = true;
}

const Tensor CudnnActivation::Forward(int flag, const Tensor& input) {
  CHECK(buf_.empty());
  auto size = input.Size();
  DataType dtype = input.data_type();
  if (!has_init_cudnn_) {
    InitCudnn(size, dtype);
  } else {
    int n, c, h, w, s;
    cudnnDataType_t type;
    CUDNN_CHECK(cudnnGetTensor4dDescriptor(desc_, &type, &n, &c, &h, &w, &s, &s,
                                           &s, &s));
    if (size != static_cast<size_t>(w)) InitCudnn(size, dtype);
  }

  Tensor output;
  output.ResetLike(input);
  output.device()->Exec(
      [input, output, this](Context* ctx) {
        Block *inblock = input.block(), *outblock = output.block();
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnActivationForward(
            ctx->cudnn_handle, this->acti_desc_, &alpha, this->desc_,
            inblock->data(), &beta, this->desc_, outblock->mutable_data()));
      },
      {input.block()}, {output.block()}, "cudnnActivationForward");
  if (flag & kTrain) {
    if (cudnn_mode_ == CUDNN_ACTIVATION_SIGMOID ||
        cudnn_mode_ == CUDNN_ACTIVATION_TANH) {
      buf_.push(output);
    } else if (cudnn_mode_ == CUDNN_ACTIVATION_RELU) {
      buf_.push(input);
    }
  }
  return output;
}

const std::pair<Tensor, vector<Tensor>> CudnnActivation::Backward(
    int flag, const Tensor& grad) {
  vector<Tensor> param_grad;
  Tensor dx;
  CHECK(!buf_.empty());
  // inout means either used as input or output, only one is valid for one type
  // of activation
  Tensor inout = buf_.top();
  buf_.pop();
  dx.ResetLike(grad);
  dx.device()->Exec(
      [dx, grad, inout, this](Context* ctx) {
        Block *dyblock = grad.block(), *dxblock = dx.block(),
              *yblock = inout.block(), *xblock = inout.block();
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnActivationBackward(
            ctx->cudnn_handle, this->acti_desc_, &alpha, this->desc_,
            yblock->data(), this->desc_, dyblock->data(), this->desc_,
            xblock->data(), &beta, this->desc_, dxblock->mutable_data()));
      },
      {grad.block(), inout.block()}, {dx.block()}, "cudnnActivationBackward");
  return std::make_pair(dx, param_grad);
}
}  // namespace singa
#endif  // USE_CUDNN
