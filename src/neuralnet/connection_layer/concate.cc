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
  CHECK_GT(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  vector<int> shape = srclayers[0]->data(this).shape();
  CHECK_GE(partition_dim(), 0);
  CHECK_LT(partition_dim(), shape.size());
  CHECK_EQ(num_partitions(), srclayers.size());
  for (size_t i = 1; i < srclayers.size(); i++) {
    const vector<int>& src_shape = srclayers[i]->data(this).shape();
    for (size_t j = 0; j < shape.size(); j++)
      if (static_cast<int>(j) == partition_dim())
        shape[j] += src_shape[j];
      else
        CHECK_EQ(shape[j], src_shape[j]);
  }
  data_.Reshape(shape);
  grad_.Reshape(shape);
}

void ConcateLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  CHECK_GT(srclayers.size(), 1);
  CHECK_EQ(num_partitions(), srclayers.size());
  // calculate step for each memcpy
  int step = srclayers[0]->data(this).shape()[partition_dim()];
  for (unsigned i = partition_dim() + 1; i < data_.shape().size(); ++i)
    step *= data_.shape()[i];
  int srclayer_offset = 0;
  int concate_offset = 0;
  while (concate_offset < data_.count()) {
    for (size_t i = 0; i < srclayers.size(); ++i) {
      const float* src = srclayers[i]->data(this).cpu_data() + srclayer_offset;
      float* dst = data_.mutable_cpu_data() + concate_offset;
      memcpy(dst, src, step * sizeof(float));
      concate_offset += step;
    }
    srclayer_offset += step;
  }
}

void ConcateLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  CHECK_GT(srclayers.size(), 1);
  CHECK_EQ(num_partitions(), srclayers.size());
  // calculate step for each memcpy
  int step = srclayers[0]->grad(this).shape()[partition_dim()];
  for (unsigned i = partition_dim() + 1; i < grad_.shape().size(); ++i)
    step *= grad_.shape()[i];
  int srclayer_offset = 0;
  int concate_offset = 0;
  while (concate_offset < grad_.count()) {
    for (size_t i = 0; i < srclayers.size(); ++i) {
      const float* src = grad_.cpu_data() + concate_offset;
      float* dst = srclayers[i]->mutable_grad(this)->mutable_cpu_data()
                   + srclayer_offset;
      memcpy(dst, src, step * sizeof(float));
      concate_offset += step;
    }
    srclayer_offset += step;
  }
}

}  // namespace singa
