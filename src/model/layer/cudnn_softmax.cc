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
#include "singa_config.h"
#include "./cudnn_softmax.h"
#ifdef USE_CUDNN
#include <cudnn.h>
#include "./cudnn_utils.h"
#include "singa/utils/logging.h"
namespace singa {
CudnnSoftmax::~CudnnSoftmax() {
  if (desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
}

void CudnnSoftmax::InitCudnn(size_t size, DataType dtype) {
  CHECK(!has_init_cudnn_);
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      desc_, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), 1, 1, 1, size));

  algorithm_ = CUDNN_SOFTMAX_ACCURATE;
  mode_ = CUDNN_SOFTMAX_MODE_INSTANCE;
  has_init_cudnn_ = true;
}

const Tensor CudnnSoftmax::Forward(int flag, const Tensor& input) {
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
    cudnnSoftmaxForward(ctx->cudnn_handle, this->algorithm_, this->mode_,
                        &alpha, this->desc_, inblock->data(), &beta,
                        this->desc_, outblock->mutable_data());
  }, {input.block()}, {output.block()});
  if (flag & kTrain) buf_.push(output);
  return output;
}

const std::pair<Tensor, vector<Tensor>> CudnnSoftmax::Backward(
    int flag, const Tensor& grad) {
  vector<Tensor> param_grad;
  CHECK(!buf_.empty());
  Tensor dx, output = buf_.top();
  buf_.pop();
  dx.ResetLike(grad);
  dx.device()->Exec([dx, grad, output, this](Context* ctx) {
    Block* dyblock = grad.block(), * dxblock = dx.block(),
           * yblock = output.block();
    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxBackward(ctx->cudnn_handle, this->algorithm_, this->mode_,
                         &alpha, this->desc_, yblock->data(), this->desc_,
                         dyblock->data(), &beta, this->desc_,
                         dxblock->mutable_data());
  }, {grad.block(), output.block()}, {dx.block()});
  return std::make_pair(dx, param_grad);
}
}  // namespace singa
#endif  // USE_CUDNN
