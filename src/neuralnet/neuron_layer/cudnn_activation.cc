/************************************************************
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
*************************************************************/

#include "singa/neuralnet/neuron_layer.h"

namespace singa {

void CudnnActivationLayer::InitCudnn() {
  CudnnBase::InitCudnn();

  // TODO(wangwei) make the mode case insensitive
  if (layer_conf_.activation_conf().type() == SIGMOID)
    mode_ = CUDNN_ACTIVATION_SIGMOID;
  else if (layer_conf_.activation_conf().type() == TANH)
    mode_ = CUDNN_ACTIVATION_TANH;
  else if (layer_conf_.activation_conf().type() == RELU)
    mode_ = CUDNN_ACTIVATION_RELU;
  else
    LOG(FATAL) << "Unkown activation: " << layer_conf_.activation_conf().type();

  const auto& shape = data_.shape();
  CHECK_GT(shape.size(), 0);
  // TODO(wangwei) cudnnSetTensorNdDescriptor reports error if nbdim is < 4.
  const int nbdim = 4;
  // size of each dimension
  int* sdim = new int[nbdim];
  int* stride = new int[nbdim];
  int i = shape.size() - 1;
  sdim[i] = shape[i];
  stride[i] = 1;
  // LOG(ERROR) << "layer " << name();
  // LOG(ERROR) << sdim[i] << " " << stride[i];
  for (--i; i >= 0; i--) {
    sdim[i] = shape[i];
    stride[i] = shape[i + 1] * stride[i + 1];
    // LOG(ERROR) << sdim[i] << " " << stride[i];
  }
  // padding sdim and stride to 4 dimensions
  for (i = shape.size(); i < nbdim; i++) {
    sdim[i] = 1;
    stride[i] = 1;
  }
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(src_desc_,
        CUDNN_DATA_FLOAT,
        nbdim,
        sdim,
        stride));
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(my_desc_,
        CUDNN_DATA_FLOAT,
        nbdim,
        sdim,
        stride));
  delete[] sdim;
  delete[] stride;
}

void CudnnActivationLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  if (!has_init_cudnn_)
    InitCudnn();
  float alpha = 1.0f, beta = 0.0f;
  // currently only consider single src layer
  CHECK_EQ(srclayers.size(), 1);
  CHECK_CUDNN(cudnnActivationForward(handle_,
        mode_,
        &alpha,
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        &beta,
        my_desc_,
        data_.mutable_gpu_data()));
}

void CudnnActivationLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnActivationBackward(handle_,
        mode_,
        &alpha,
        my_desc_,
        data_.gpu_data(),
        my_desc_,
        grad_.gpu_data(),
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        &beta,
        src_desc_,
        srclayers[0]->mutable_grad(this)->mutable_gpu_data()));
}
}   // namespace singa
