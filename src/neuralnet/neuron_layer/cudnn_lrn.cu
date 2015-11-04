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
CudnnLRNLayer::~CudnnLRNLayer() {
  if (!init_cudnn_) {
    cudnnDestroyLRNDescriptor(norm_desc_);
  }
}

void CudnnLRNLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  LRNLayer::Setup(proto, srclayers);
  mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
}

void CudnnLRNLayer::InitCudnn() {
  CudnnLayer::InitCudnn(srclayers);
  CHECK_EQ(cudnnCreateLRNDescriptor(&norm_desc_), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnSetLRNDescriptor(norm_desc_,
        lsize_,
        alpha_,
        beta_,
        knorm_), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateTensorDescriptor(&src_desc_), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnSetTensor4dDescriptor(src_desc_,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batchsize_,
      channels_,
      height_,
      width_), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateTensorDescriptor(&my_desc_), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnSetTensor4dDescriptor(my_desc_,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batchsize_,
      channels_,
      height_,
      width_), CUDNN_STATUS_SUCCESS);
}
void ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if (init_cudnn_) {
    InitCudnn();
    init_cudnn_ = false;
  }
  CHECK_EQ(cudnnLRNCrossChannelForward(handle_,
      norm_desc_,
      mode_,
      &alpha,
      src_desc_,
      srclayers[0]->data(this).gpu_data(),
      &beta,
      my_desc_,
      data_.mutable_gpu_data()), CUDNN_STATUS_SUCCESS);
}
void ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  CHECK_EQ(cudnnLRNCrossChannelBackward(handle_,
        norm_desc_,
        mode_,
        &alpha,
        my_desc_, // ???
        data_.gpu_data(),
        my_desc_,
        grad_.gpu_data()
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        &beta,
        src_desc_,
        srclayers[0]->mutable_grad(this)->mutable_gpu_data()),
      CUDNN_STATUS_SUCCESS);
}


} /* singa */
