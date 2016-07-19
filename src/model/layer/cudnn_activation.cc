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
#include "./cudnn_activation.h"
#include <cudnn.h>

#include "./cudnn_utils.h"
#include "singa/core/common.h"
#include "singa/utils/logging.h"

namespace singa {
RegisterLayerClass(CudnnActivation);
CudnnActivation::~CudnnActivation() {
  if (acti_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(acti_desc_));
  if (desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
}

void CudnnActivation::InitCudnn(size_t size, DataType dtype) {
  CHECK(!has_init_cudnn_);
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&acti_desc_));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      desc_, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), 1, 1, 1, size));

  if (mode_ == "SIGMOID")
    cudnn_mode_ = CUDNN_ACTIVATION_SIGMOID;
  else if (mode_ == "TANH")
    cudnn_mode_ = CUDNN_ACTIVATION_TANH;
  else if (mode_ == "RELU")
    cudnn_mode_ = CUDNN_ACTIVATION_RELU;
  else
    LOG(FATAL) << "Unkown activation: " << mode_;

  CUDNN_CHECK(cudnnSetActivationDescriptor(
        acti_desc_, cudnn_mode_, CUDNN_PROPAGATE_NAN, 0.0f));
  has_init_cudnn_ = true;
}

const Tensor CudnnActivation::Forward(int flag, const Tensor& input) {
  CHECK(buf_.empty());
  auto size = input.Size();
  DataType dtype = input.data_type();
  if (!has_init_cudnn_) {
    InitCudnn(size, dtype);
  }
  Tensor output;
  output.ResetLike(input);
  output.device()->Exec([input, output, this](Context* ctx) {
    Block* inblock = input.block(), * outblock = output.block();
    float alpha = 1.0f, beta = 0.0f;
#if CUDNN_VERSION_MAJOR == 5
    CUDNN_CHECK(cudnnActivationForward(
        ctx->cudnn_handle, this->acti_desc_, &alpha, this->desc_,
        inblock->data(), &beta, this->desc_, outblock->mutable_data()));
#elif CUDNN_VERSION_MAJOR == 4
    CUDNN_CHECK(cudnnActivationForward_v4(
        ctx->cudnn_handle, this->acti_desc_, &alpha, this->desc_,
        inblock->data(), &beta, this->desc_, outblock->mutable_data()));
#endif
  }, {input.block()}, {output.block()});
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
  dx.device()->Exec([dx, grad, inout, this](Context* ctx) {
    Block* dyblock = grad.block(), * dxblock = dx.block(),
           * yblock = inout.block(), * xblock = inout.block();
    float alpha = 1.0f, beta = 0.0f;
#if CUDNN_VERSION_MAJOR == 5
    CUDNN_CHECK(cudnnActivationBackward(
        ctx->cudnn_handle, this->acti_desc_, &alpha, this->desc_,
        yblock->data(), this->desc_, dyblock->data(), this->desc_,
        xblock->data(), &beta, this->desc_, dxblock->mutable_data()));
#elif CUDNN_VERSION_MAJOR == 4
    CUDNN_CHECK(cudnnActivationBackward_v4(
        ctx->cudnn_handle, this->acti_desc_, &alpha, this->desc_, yblock->data(),
        this->desc_, dyblock->data(), this->desc_, xblock->data(), &beta,
        this->desc_, dxblock->mutable_data()));
#endif
  }, {grad.block(), inout.block()}, {dx.block()});
  return std::make_pair(dx, param_grad);
}
}  // namespace singa
#endif  // USE_CUDNN
