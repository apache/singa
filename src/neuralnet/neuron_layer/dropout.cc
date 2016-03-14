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
#include "singa/utils/singa_op.h"
#include "singa/utils/math_blob.h"

namespace singa {
using std::vector;

void DropoutLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(*srclayers[0]->mutable_grad(this));
  mask_.Reshape(srclayers[0]->data(this).shape());
  pdrop_ = conf.dropout_conf().dropout_ratio();
}

void DropoutLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  // check training
  if ((flag & kTrain) != kTrain) {
    data_.CopyFrom(srclayers[0]->data(this));
    return;
  }

  float pkeep = 1 - pdrop_;
  Blob<float> rand(data_.count());
  SampleUniform(0.0f, 1.0f, &rand);
  Map<op::Threshold<float>, float>(pkeep, rand, &mask_);
  // scale the mask to avoid scaling in ComputeGradient
  Scale(1.0f / pkeep, &mask_);
  Mult(srclayers[0]->data(this), mask_, &data_);
}

void DropoutLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers)  {
  Mult(grad_, mask_, srclayers[0]->mutable_grad(this));
  // no need to mult scale as mask is scaled already.
}

}  // namespace singa

