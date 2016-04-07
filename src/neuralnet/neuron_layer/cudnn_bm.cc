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

#if CUDNN_MAJOR == 4
namespace singa {

CudnnBMLayer::~CudnnBMLayer() {
  if (has_init_cudnn_) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc_));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bnScaleBiasDiff_desc_));
  }
}

void CudnnBMLayer::InitCudnn() {
  CudnnBase::InitCudnn();

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc_));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasDiff_desc_));

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
        height_,
        width_));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(bnScaleBiasMeanVar_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,
        channels_,
        1,
        1));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(bnScaleBiasDiff_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,
        channels_,
        1,
        1));

  vector<int> shape{1, channels_, 1, 1};

  resultSaveMean_.Reshape(shape);
  resultSaveInvVariance_.Reshape(shape);

  mode_ = CUDNN_BATCHNORM_SPATIAL;
}

void CudnnBMLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  if (!has_init_cudnn_)
    InitCudnn();

  const float alpha = 1.0f, beta = 0.0f;
  double exponentialAverageFactor = 1.0;
  double epsilon = CUDNN_BN_MIN_EPSILON;

  // check training
  if ((flag & kTrain) != kTrain) {
    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(handle_,
          mode_,
          &alpha,
          &beta,
          src_desc_,
          srclayers.at(0)->data(this).gpu_data(),
          my_desc_,
          data_.mutable_gpu_data(),
          bnScaleBiasMeanVar_desc_,
          bnScale_->data().gpu_data(),
          bnBias_->data().gpu_data(),
          resultRunningMean_->data().gpu_data(),
          resultRunningInvVariance_->data().gpu_data(),
          epsilon));
  } else {
    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(handle_,
          mode_,
          &alpha,
          &beta,
          src_desc_,
          srclayers.at(0)->data(this).gpu_data(),
          my_desc_,
          data_.mutable_gpu_data(),
          bnScaleBiasMeanVar_desc_,
          bnScale_->data().gpu_data(),
          bnBias_->data().gpu_data(),
          exponentialAverageFactor,
          resultRunningMean_->mutable_data()->mutable_gpu_data(),
          resultRunningInvVariance_->mutable_data()->mutable_gpu_data(),
          epsilon,
          resultSaveMean_.mutable_gpu_data(),
          resultSaveInvVariance_.mutable_gpu_data()));
  }
}

void CudnnBMLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {

  const float alpha = 1.0f, beta = 0.0f, alphaDiff = 1.0f, betaDiff = 0.0f;
  double epsilon = CUDNN_BN_MIN_EPSILON;

  CHECK_CUDNN(cudnnBatchNormalizationBackward(handle_,
      mode_,
      &alpha,
      &beta,
      &alphaDiff,
      &betaDiff,
      src_desc_,
      srclayers.at(0)->data(this).gpu_data(),
      my_desc_,
      grad_.gpu_data(),
      src_desc_,
      srclayers.at(0)->mutable_grad(this)->mutable_gpu_data(),
      bnScaleBiasDiff_desc_,
      bnScale_->data().gpu_data(),
      bnScale_->mutable_grad()->mutable_gpu_data(),
      bnBias_->mutable_grad()->mutable_gpu_data(),
      epsilon,
      resultSaveMean_.gpu_data(),
      resultSaveInvVariance_.gpu_data()));
}
}  // namespace singa
#endif
