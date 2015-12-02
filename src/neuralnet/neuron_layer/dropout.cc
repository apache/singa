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

#include "singa/neuralnet/neuron_layer/dropout.h"

#include <glog/logging.h>
#include "singa/utils/singleton.h"

namespace singa {
using std::vector;

void DropoutLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  data_.resize(1);
  data_.at(0).ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(*srclayers[0]->mutable_grad(this));
  mask_.Reshape(srclayers[0]->data(this).shape());
  pdrop_ = conf.dropout_conf().dropout_ratio();
}

void DropoutLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  // check training
  if ((flag & kTrain) != kTrain) {
    data_.at(0).CopyFrom(srclayers[0]->data(this));
    return;
  }
  float pkeep = 1 - pdrop_;
  auto mask = Tensor1(&mask_);
  mask = expr::F<op::threshold>(TSingleton<Random<cpu>>::Instance() \
                      ->uniform(mask.shape), pkeep) * (1.0f/pkeep);
  auto data = Tensor1(&data_.at(0));
  auto src = Tensor1(srclayers[0]->mutable_data(this));
  data = src * mask;
}

void DropoutLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers)  {
  auto mask = Tensor1(&mask_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers[0]->mutable_grad(this));
  gsrc = grad * mask;
}

}  // namespace singa

