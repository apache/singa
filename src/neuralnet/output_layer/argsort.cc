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

void ArgSortLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  OutputLayer::Setup(proto, srclayers);
  batchsize_ = srclayers[0]->data(this).shape()[0];
  dim_ = srclayers[0]->data(this).count() / batchsize_;
  topk_ = proto.argsort_conf().topk();
  data_.Reshape(vector<int>{batchsize_, topk_});
}

void ArgSortLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  // TODO(wangwei) check flag to ensure it is not called in training phase
  const float* srcptr = srclayers.at(0)->data(this).cpu_data();
  float* ptr = data_.mutable_cpu_data();
  for (int n = 0; n < batchsize_; n++) {
    vector<std::pair<float, int> > vec;
    for (int j = 0; j < dim_; ++j)
      vec.push_back(std::make_pair(srcptr[j], j));
    std::partial_sort(vec.begin(), vec.begin() + topk_, vec.end(),
                      std::greater<std::pair<float, int> >());

    for (int j = 0; j < topk_; ++j)
      ptr[j] = static_cast<float> (vec.at(j).second);
    ptr += topk_;
    srcptr += dim_;
  }
}

}  // namespace singa
