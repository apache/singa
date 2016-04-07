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

#include <glog/logging.h>
#include "singa/neuralnet/neuron_layer.h"
#include "singa/utils/singleton.h"

namespace singa {

using std::vector;

void BMLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(srclayers[0]->grad(this));

  const vector<int>& srcshape = srclayers[0]->data(this).shape();

  batchsize_ = srcshape[0];
  channels_ = srcshape[1];
  height_ = srcshape[2];
  width_ = srcshape[3];

  bnScale_ = Param::Create(conf.param(0));
  bnScale_->Setup(vector<int>{1, channels_, 1, 1});

  bnBias_ = Param::Create(conf.param(1));
  bnBias_->Setup(vector<int>{1, channels_, 1, 1});

  resultRunningMean_ = Param::Create(conf.param(2));
  resultRunningMean_->Setup(vector<int>{1, channels_, 1, 1});

  resultRunningInvVariance_ = Param::Create(conf.param(3));
  resultRunningInvVariance_->Setup(vector<int>{1, channels_, 1, 1});
}

void BMLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  // Todo
}

void BMLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  // Todo
}

}  //  namespace singa
