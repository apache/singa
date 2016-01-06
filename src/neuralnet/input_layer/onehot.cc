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
void OneHotLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(conf, srclayers);
  batchsize_ = srclayers.at(0)->data(unroll_index()).shape(0);
  dim_ = conf.onehot_conf().vocab_size();
  data_.Reshape(batchsize_, dim_);
}

void OneHotLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  float* ptr = data_.mutable_cpu_data();
  memset(ptr, 0, sizeof(float) * data_.count());
  const float* idx = srclayers[0]->data(unroll_index()).cpu_data();
  for (int i = 0; i < batchsize_; i++) {
    ptr[i * dim_ + static_cast<int>(idx[i])] = 1;
  }
}
}  // namespace singa
