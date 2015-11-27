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

namespace singa {
void CudnnSoftmaxLossLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CudnnSoftmaxLayer::Setup(conf, srclayers);
  topk_ = conf.softmaxloss_conf().topk();
  loss_ = accuracy_ = 0.0f;
  counter_ = 0;
}
void CudnnSoftmaxLossLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  CudnnSoftmaxLayer::ComputeFeature(flag, srclayers);
  // compute loss
  counter_++;
  // add loss and accuracy
}

void CudnnSoftmaxLossLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
 // compute gradient
}

const std::string CudnnSoftmaxLossLayer::ToString(bool debug, int flag) {
  string disp = "Loss = " + std::to_string(loss_ / counter_)
    + ", accuracy = " + std::to_string(accuracy_ / counter_);
  counter_ = 0;
  loss_ = accuracy_ = 0;
  return disp;
}
}  // namespace singa
