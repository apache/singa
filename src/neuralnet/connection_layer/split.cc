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

#include "singa/neuralnet/connection_layer.h"
#include "singa/utils/math_blob.h"

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
  data_.ShareData(srclayers[0]->mutable_data(this), false);
  num_splits_ = conf.split_conf().num_splits();
  CHECK_GT(num_splits_, 0);
  // add num_splits-1 more grad blobs
  for (int i = 1; i < num_splits_; ++i) {
    gradvec_.push_back(new Blob<float>());
  }
  for (int i = 0; i < num_splits_; ++i)
    gradvec_[i]->Reshape(srclayers[0]->data(this).shape());
}

void SplitLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  // data is shared from its source,
  // nothing to do in compute feature phase
}

void SplitLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  // aggregate all gradients to grad_[0]
  for (int i = 1; i < num_splits_; ++i)
    AXPY<float>(1.0, *gradvec_[i], gradvec_[0]);
  // copy grad_[0] to srclayer's grad
  Copy(*gradvec_[0], srclayers[0]->mutable_grad(this));
}

const Blob<float>& SplitLayer::grad(const Layer* from) {
  CHECK(from);
  int idx = layer_idx_.Get(from);
  CHECK_LT(idx, num_splits_);
  return *gradvec_[idx];
}

Blob<float>* SplitLayer::mutable_grad(const Layer* from) {
  CHECK(from);
  int idx = layer_idx_.Get(from);
  CHECK_LT(idx, num_splits_);
  return gradvec_[idx];
}
const std::string SplitLayer::ToString(bool debug, int flag) {
  if (!debug)
    return "";
  string ret = "";
  if ((flag & kForward) == kForward && data_.count() !=0) {
    ret += StringPrintf("data:%13.9f ", Asum(data_));
  }
  if ((flag & kBackward) == kBackward && grad_.count() != 0) {
    for (unsigned k = 0; k < gradvec_.size(); k++)
    ret += StringPrintf("grad-%u:%13.9f ", k, Asum(*gradvec_.at(k)));
  }
  return ret;
}
}  // namespace singa
