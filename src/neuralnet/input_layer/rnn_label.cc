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

#include "singa/neuralnet/input_layer.h"
namespace singa {
void RNNLabelLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(proto, srclayers);
  aux_data_.resize(srclayers[0]->data(unroll_index() + 1).shape(0));
}
void RNNLabelLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  const float* input = srclayers[0]->data(unroll_index() + 1).cpu_data();
  for (unsigned i = 0; i < aux_data_.size(); i++) {
    aux_data_[i] = static_cast<int>(input[i]);
  }
}
}  // namespace singa
