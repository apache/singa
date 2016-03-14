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

void RNNDummyLayer::Setup(const LayerProto& conf,
                       const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  dynamic_src_ = AddPrefixSuffix(unroll_index(), partition_id(),
      conf.rnn_dummy_conf().dynamic_srclayer());
  LOG(ERROR) << dynamic_src_;
  vector<int> shape;
  for (int s : conf.rnn_dummy_conf().shape())
    shape.push_back(s);
  integer_ = conf.rnn_dummy_conf().integer();
  low_ = conf.rnn_dummy_conf().low();
  high_ = conf.rnn_dummy_conf().high();
  // if no src layer, then it will genereate data by itself based on shape
  // and random range
  if (srclayers.size() == 0) {
    CHECK(shape.size());
    CHECK_NE(low_, high_);
    data_.Reshape(shape);
    srclayer_ = nullptr;
  } else {
    srclayer_ = srclayers.at(0);
    data_.ReshapeLike(srclayer_->data(this));
    data_.ShareData(srclayer_->mutable_data(this), false);
  }
}

void RNNDummyLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if (srclayers.size() == 0) {
    SampleUniform(low_, high_, &data_);
    if (integer_) {
      for (int i = 0; i < data_.count(); i ++) {
        data_.mutable_cpu_data()[i] = floor(data_.cpu_data()[i]);
      }
    }
  } else if (srclayer_ != srclayers.at(0)) {
    srclayer_ = srclayers.at(0);
    data_.ShareData(srclayer_->mutable_data(this), false);
  }
}
}  // namespace singa

