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

#include "singa/neuralnet/loss_layer.h"
#include "singa/utils/blob.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/math_kernel.h"

namespace singa {
void CudnnSoftmaxLossLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  LossLayer::Setup(conf, srclayers);
  softmax_.Setup(conf, vector<Layer*> {srclayers.at(0)});
  data_.Reshape(softmax_.data(this).shape());
  data_.ShareData(softmax_.mutable_data(this), false);
  batchsize_ = data_.shape(0);
  dim_ = data_.count() / batchsize_;
}
void CudnnSoftmaxLossLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  softmax_.ComputeFeature(flag, srclayers);
  Blob<int> label(batchsize_);
  int *labelptr = label.mutable_cpu_data();
  // aux_data: vector<int>, convert vector to int array.
  for (int i = 0; i < batchsize_; ++i) {
    labelptr[i] = srclayers[1]->aux_data(this)[i];
  }

  Blob<float> loss(batchsize_);
  singa_gpu_softmaxloss_forward(batchsize_, dim_, data_.gpu_data(),
      label.gpu_data(), loss.mutable_gpu_data());
  loss_ += Asum(loss);
  counter_++;
}

void CudnnSoftmaxLossLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  Blob<float>* gsrcblob = srclayers[0]->mutable_grad(this);
  Copy(data_, gsrcblob);
  // gsrcblob->CopyFrom(data_);
  float* gsrcptr = gsrcblob->mutable_gpu_data();

  Blob<int> label(batchsize_);
  int *labelptr = label.mutable_cpu_data();

  // aux_data: vector<int>, convert vector to int array.
  for (int i = 0; i < batchsize_; ++i) {
    labelptr[i] = srclayers[1]->aux_data(this)[i];
  }

  singa_gpu_softmaxloss_backward(batchsize_, dim_, 1.0f, label.gpu_data(),
      gsrcptr);
  Scale(1.0f / batchsize_, gsrcblob);
}

const std::string CudnnSoftmaxLossLayer::ToString(bool debug, int flag) {
  if (debug)
    return Layer::ToString(debug, flag);

  string disp = "Loss = " + std::to_string(loss_ / counter_);
  counter_ = 0;
  loss_ = 0;
  return disp;
}
}  // namespace singa
