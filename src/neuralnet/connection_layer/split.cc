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

SplitLayer::~SplitLayer() {
  for (size_t i = 1; i < gradvec_.size(); ++i) {
    if (gradvec_[i] != nullptr) delete gradvec_[i];
  }
}

void SplitLayer::Setup(const LayerProto& conf,
                       const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  data_.Reshape(srclayers[0]->data(this).shape());
  data_.ShareData(srclayers[0]->data(this));
  CHECK_GT(num_partitions(), 0);
  // add num_partitions()-1 more grad blobs
  for (int i = 1; i < num_partitions(); ++i) {
    gradvec_.push_back(new Blob<float>());
  }
  for (int i = 0; i < num_partitions(); ++i)
    gradvec_[i]->Reshape(srclayers[0]->data(this).shape());
}

void SplitLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  // data is shared from its source,
  // nothing to do in compute feature phase
}

void SplitLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  // aggregate all gradients to grad_[0]
  for (int i = 1; i < num_partitions(); ++i)
    for (int j = 0; j < gradvec_[0]->count(); ++j)
      gradvec_[0]->mutable_cpu_data()[j] += gradvec_[i]->cpu_data()[j];
  // copy grad_[0] to srclayer's grad
  srclayers[0]->mutable_grad(this)->CopyFrom(*gradvec_[0]);
}

const Blob<float>& SplitLayer::grad(const Layer* from) const {
  CHECK(from);
  CHECK_LT(from->partition_id(), num_partitions());
  return *gradvec_[from->partition_id()];
}

Blob<float>* SplitLayer::mutable_grad(const Layer* from) {
  CHECK(from);
  CHECK_LT(from->partition_id(), num_partitions());
  return gradvec_[from->partition_id()];
}

}  // namespace singa
