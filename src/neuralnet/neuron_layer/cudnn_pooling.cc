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

CudnnPoolLayer::~CudnnPoolLayer() {
  if (has_init_cudnn_) {
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pool_desc_));
  }
}

void CudnnPoolLayer::InitCudnn() {
  CudnnBase::InitCudnn();
  CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool_desc_));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(src_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchsize_,
        channels_,
        height_,
        width_));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(my_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchsize_,
        channels_,
        pooled_height_,
        pooled_width_));
  auto pool_method = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  if (pool_ == PoolingProto_PoolMethod_MAX)
    pool_method = CUDNN_POOLING_MAX;
  CHECK_CUDNN(cudnnSetPooling2dDescriptor(pool_desc_,
        pool_method,
        kernel_y_,
        kernel_x_,
        pad_y_,
        pad_x_,
        stride_y_,
        stride_x_));
}

void CudnnPoolLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if (!has_init_cudnn_)
    InitCudnn();
  float alpha = 1.0f, beta = 0.0f;
  // currently only consider single src layer
  CHECK_EQ(srclayers.size(), 1);
  CHECK_CUDNN(cudnnPoolingForward(handle_,
        pool_desc_,
        &alpha,
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        &beta,
        my_desc_,
        data_.mutable_gpu_data()));
}

void
CudnnPoolLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnPoolingBackward(handle_,
        pool_desc_,
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
}  // namespace singa

