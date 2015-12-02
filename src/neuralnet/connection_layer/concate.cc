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

#include "singa/neuralnet/connection_layer/concate.h"

namespace singa {

using std::vector;

void ConcateLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  size_t concate_dim = conf.concate_conf().concate_dim();
  CHECK_GE(concate_dim, 0);
  CHECK_GT(srclayers.size(), 1);
  vector<int> shape = srclayers[0]->data(this).shape();
  for (size_t i = 1; i < srclayers.size(); i++) {
    const vector<int>& srcshape = srclayers[i]->data(this).shape();
    for (size_t j = 0; j < shape.size(); j++)
      if (j == concate_dim)
        shape[j] += srcshape[j];
      else
        CHECK_EQ(shape[j], srcshape[j]);
  }
  data_.resize(1);
  data_.at(0).Reshape(shape);
  grad_.Reshape(shape);
}

void ConcateLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented for Concate Layer";
}

void ConcateLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented for Concate Layer";
}

}  // namespace singa
