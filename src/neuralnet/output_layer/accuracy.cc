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

#include <algorithm>
#include "singa/neuralnet/output_layer.h"

namespace singa {

void AccuracyLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 2);
  ArgSortLayer::Setup(proto, vector<Layer*>{srclayers.at(0)});
}

void AccuracyLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  ArgSortLayer::ComputeFeature(flag, vector<Layer*>{srclayers.at(0)});
  const auto& label = srclayers[1]->aux_data(this);
  int ncorrect = 0;
  for (int n = 0; n < batchsize_; n++) {
    const float* pos = data_.cpu_data() + topk_ * n;
    // check if true label is in top k predictions
    for (int k = 0; k < topk_; k++) {
      if (pos[k] == label[n]) {
        ncorrect++;
        break;
      }
    }
  }
  accuracy_ += ncorrect * 1.0f / batchsize_;
  counter_++;
}

const std::string AccuracyLayer::ToString(bool debug, int flag) {
  if (debug)
    return Layer::ToString(debug, flag);

  string disp = "accuracy = " + std::to_string(accuracy_ / counter_);
  counter_ = 0;
  accuracy_ = 0;
  return disp;
}
}  // namespace singa
