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

#include "singa/neuralnet/connection_layer/slice.h"

namespace singa {

using std::vector;

void SliceLayer::Setup(const LayerProto& conf,
                       const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  slice_dim_ = conf.slice_conf().slice_dim();
  slice_num_ = conf.slice_conf().slice_num();
  vector<int> shape = srclayers[0]->data(this).shape();
  CHECK_GE(slice_dim_, 0);
  CHECK_LT(slice_dim_, shape.size());
  CHECK_GT(slice_num_, 0);
  datavec_.resize(slice_num_);
  gradvec_.resize(slice_num_);
  // TODO(wangsh): remove equal-size restrict later
  CHECK_EQ(shape[slice_dim_] % slice_num_, 0);
  shape[slice_dim_] /= slice_num_;
  for (int i = 0; i < slice_num_; ++i) {
    // if (i == slice_num - 1) shape[slice_dim_] += remain;
    datavec_[i].Reshape(shape);
    gradvec_[i].Reshape(shape);
  }
}

void SliceLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  const Blob<float>& blob = srclayers[0]->data(this);
  // calculate step for each memcpy
  int step = datavec_[0].shape()[slice_dim_];
  for (unsigned i = slice_dim_ + 1; i < datavec_[0].shape().size(); ++i)
    step *= datavec_[0].shape()[i];
  int srclayer_offset = 0;
  int slice_offset = 0;
  while (srclayer_offset < blob.count()) {
    for (int i = 0; i < slice_num_; ++i) {
      const float* src = blob.cpu_data() + srclayer_offset;
      float* dst = datavec_[i].mutable_cpu_data() + slice_offset;
      memcpy(dst, src, step * sizeof(float));
      srclayer_offset += step;
    }
    slice_offset += step;
  }
}

void SliceLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  Blob<float>* blob = srclayers[0]->mutable_grad(this);
  // calculate step for each memcpy
  int step = gradvec_[0].shape()[slice_dim_];
  for (size_t i = slice_dim_ + 1; i < gradvec_[0].shape().size(); ++i)
    step *= gradvec_[0].shape()[i];
  int srclayer_offset = 0;
  int slice_offset = 0;
  while (srclayer_offset < blob->count()) {
    for (int i = 0; i < slice_num_; ++i) {
      const float* src = gradvec_[i].cpu_data() + slice_offset;
      float* dst = blob->mutable_cpu_data() + srclayer_offset;
      memcpy(dst, src, step * sizeof(float));
      srclayer_offset += step;
    }
    slice_offset += step;
  }
}

const Blob<float>& SliceLayer::data(const Layer* from) const {
  CHECK(from);
  CHECK_LT(from->partition_id(), datavec_.size());
  return datavec_[from->partition_id()];
}

const Blob<float>& SliceLayer::grad(const Layer* from) const {
  CHECK(from);
  CHECK_LT(from->partition_id(), gradvec_.size());
  return gradvec_[from->partition_id()];
}

Blob<float>* SliceLayer::mutable_data(const Layer* from) {
  CHECK(from);
  CHECK_LT(from->partition_id(), datavec_.size());
  return &datavec_[from->partition_id()];
}

Blob<float>* SliceLayer::mutable_grad(const Layer* from) {
  CHECK(from);
  CHECK_LT(from->partition_id(), gradvec_.size());
  return &gradvec_[from->partition_id()];
}

}  // namespace singa
