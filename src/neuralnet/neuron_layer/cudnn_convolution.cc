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

CudnnConvLayer::~CudnnConvLayer() {
  if (has_init_cudnn_) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc_));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }
}

void CudnnConvLayer::InitCudnn() {
  CudnnBase::InitCudnn();
  // convert MB to bytes
  workspace_byte_limit_
    = layer_conf_.convolution_conf().workspace_byte_limit() << 20;

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc_));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));

  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
        pad_y_,
        pad_x_,
        stride_y_,
        stride_x_,
        1,
        1,
        CUDNN_CROSS_CORRELATION));
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
        CUDNN_DATA_FLOAT,
        num_filters_,
        channels_,
        kernel_y_,
        kernel_x_));
  if (bias_) {
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          1,
          num_filters_,
          1,
          1));
  }
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
        num_filters_,
        conv_height_,
        conv_width_));

  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(handle_,
        src_desc_,
        filter_desc_,
        conv_desc_,
        my_desc_,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        workspace_byte_limit_,
        &fp_alg_));

  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(handle_,
        src_desc_,
        my_desc_,
        conv_desc_,
        filter_desc_,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        workspace_byte_limit_,
        &bp_filter_alg_));
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(handle_,
        filter_desc_,
        my_desc_,
        conv_desc_,
        src_desc_,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        workspace_byte_limit_,
        &bp_data_alg_));

  size_t fp_byte, bp_data_byte, bp_filter_byte;
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle_,
        src_desc_,
        filter_desc_,
        conv_desc_,
        my_desc_,
        fp_alg_,
        &fp_byte));
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_,
        filter_desc_,
        my_desc_,
        conv_desc_,
        src_desc_,
        bp_data_alg_,
        &bp_data_byte));
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_,
        src_desc_,
        my_desc_,
        conv_desc_,
        filter_desc_,
        bp_filter_alg_,
        &bp_filter_byte));
  workspace_count_ = std::max(std::max(fp_byte, bp_data_byte), bp_filter_byte)
    / sizeof(float) + 1;
}

void CudnnConvLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if (!has_init_cudnn_)
    InitCudnn();
  float alpha = 1.f, beta = 0.f;
  Blob<float> workspace(vector<int>{static_cast<int>(workspace_count_)});
  CHECK_CUDNN(cudnnConvolutionForward(handle_,
        &alpha,
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        filter_desc_,
        weight_->data().gpu_data(),
        conv_desc_,
        fp_alg_,
        workspace.mutable_gpu_data(),
        workspace_count_ * sizeof(float),
        &beta,
        my_desc_,
        data_.mutable_gpu_data()));
  if (bias_) {
    beta = 1.f;
    CHECK_CUDNN(cudnnAddTensor(handle_,
          CUDNN_ADD_SAME_C,
          &alpha,
          bias_desc_,
          bias_->data().gpu_data(),
          &beta,
          my_desc_,
          data_.mutable_gpu_data()));
  }
}

void
CudnnConvLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  float alpha = 1.f, beta = 0.f;
  Blob<float> workspace(vector<int>{static_cast<int>(workspace_count_)});
  // LOG(ERROR) << "backward bias";
  if (bias_) {
    CHECK_CUDNN(cudnnConvolutionBackwardBias(handle_,
          &alpha,
          my_desc_,
          grad_.gpu_data(),
          &beta,
          bias_desc_,
          bias_->mutable_grad()->mutable_gpu_data()));
  }
  // LOG(ERROR) << "backward w";
  CHECK_CUDNN(cudnnConvolutionBackwardFilter_v3(handle_,
        &alpha,
        src_desc_,
        srclayers[0]->data(this).gpu_data(),
        my_desc_,
        grad_.gpu_data(),
        conv_desc_,
        bp_filter_alg_,
        workspace.mutable_gpu_data(),
        workspace_count_ * sizeof(float),
        &beta,
        filter_desc_,
        weight_->mutable_grad()->mutable_gpu_data()));
  // LOG(ERROR) << "backward src";
  if (srclayers[0]->mutable_grad(this) != nullptr) {
    CHECK_CUDNN(cudnnConvolutionBackwardData_v3(handle_,
          &alpha,
          filter_desc_,
          weight_->data().gpu_data(),
          my_desc_,
          grad_.gpu_data(),
          conv_desc_,
          bp_data_alg_,
          workspace.mutable_gpu_data(),
          workspace_count_ * sizeof(float),
          &beta,
          src_desc_,
          srclayers[0]->mutable_grad(this)->mutable_gpu_data()));
  }
}
}  // namespace singa
