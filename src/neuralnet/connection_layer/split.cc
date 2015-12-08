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

#include "singa/neuralnet/connection_layer/split.h"

namespace singa {

using std::vector;

void SplitLayer::Setup(const LayerProto& conf,
                       const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  split_num_ = conf.split_conf().split_num();
  data_.Reshape(srclayers[0]->data(this).shape());
  data_.ShareData(srclayers[0]->data(this));
  grads_.resize(split_num_);
  for (int i = 0; i < split_num_; ++i)
    grads_[i].Reshape(srclayers[0]->data(this).shape());
}

void SplitLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  // data is shared from its source,
  // nothing to do in compute feature phase
}

void SplitLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  // aggregate all gradients to grad_[0]
  for (int i = 1; i < split_num_; ++i)
    for (int j = 0; j < grads_[0].count(); ++j)
      grads_[0].mutable_cpu_data()[j] += grads_[i].cpu_data()[j];
  // copy grad_[0] to srclayer's grad
  srclayers[0]->mutable_grad(this)->CopyFrom(grads_[0]);
}

const Blob<float>& SplitLayer::grad(const Layer* from) const {
  CHECK(from);
  CHECK_LT(from->partition_id(), grads_.size());
  return grads_[from->partition_id()];
}

Blob<float>* SplitLayer::mutable_grad(const Layer* from) {
  CHECK(from);
  CHECK_LT(from->partition_id(), grads_.size());
  return &grads_[from->partition_id()];
}

}  // namespace singa
