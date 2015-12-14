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

void CudnnSoftmaxLayer::InitCudnn() {
  CudnnBase::InitCudnn();
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(src_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchsize_,
        dim_,
        1,
        1));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(my_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchsize_,
        dim_,
        1,
        1));
}

void CudnnSoftmaxLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  if (!has_init_cudnn_)
    InitCudnn();
  const float alpha = 1.0f, beta = 0.0f;
  CHECK_EQ(srclayers.at(0)->data(this).shape().size(), 2);
  CHECK_CUDNN(cudnnSoftmaxForward(handle_,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        src_desc_,
        srclayers.at(0)->data(this).gpu_data(),
        &beta,
        my_desc_,
        data_.mutable_gpu_data()));
}

void CudnnSoftmaxLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  const float alpha = 1.f, beta = 0.f;
  CHECK_CUDNN(cudnnSoftmaxBackward(handle_,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        my_desc_,
        data_.gpu_data(),
        my_desc_,
        grad_.gpu_data(),
        &beta,
        src_desc_,
        srclayers.at(0)->mutable_grad(this)->mutable_gpu_data()));
}
}  // namespace singa
