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
  Layer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  data_.resize(1);
  data_.at(0).Reshape(srclayers[0]->data(this).shape());
  grad_.Reshape(srclayers[0]->data(this).shape());
}

void SplitLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented";
}

void SplitLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented";
}

}  // namespace singa
