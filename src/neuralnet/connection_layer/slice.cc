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

SliceLayer::~SliceLayer() {
  for (size_t i = 1; i < datavec_.size(); ++i) {
    if (datavec_[i] != nullptr) delete datavec_[i];
    if (gradvec_[i] != nullptr) delete gradvec_[i];
  }
}

void SliceLayer::Setup(const LayerProto& conf,
                       const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  vector<int> shape = srclayers[0]->data(this).shape();
  slice_dim = conf.slice_conf().slice_dim();
  num_slices = conf.slice_conf().num_slices();
  CHECK_GE(slice_dim, 0);
  CHECK_LT(slice_dim, shape.size());
  CHECK_GT(num_slices, 0);
  // add num_slices-1 more blobs
  for (int i = 1; i < num_slices; ++i) {
    datavec_.push_back(new Blob<float>());
    gradvec_.push_back(new Blob<float>());
  }
  // TODO(wangsh): remove equal-size restrict later
  CHECK_EQ(shape[slice_dim] % num_slices, 0);
  shape[slice_dim] /= num_slices;
  for (int i = 0; i < num_slices; ++i) {
    // if (i == slice_num - 1) shape[slice_dim] += remain;
    datavec_[i]->Reshape(shape);
    gradvec_[i]->Reshape(shape);
  }
}

void SliceLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  const Blob<float>& blob = srclayers[0]->data(this);
  // calculate step for each memcpy
  int step = datavec_[0]->shape()[slice_dim];
  for (unsigned i = slice_dim + 1; i < datavec_[0]->shape().size(); ++i)
    step *= datavec_[0]->shape()[i];
  int srclayer_offset = 0;
  int slice_offset = 0;
  while (srclayer_offset < blob.count()) {
    for (int i = 0; i < num_slices; ++i) {
      const float* src = blob.cpu_data() + srclayer_offset;
      float* dst = datavec_[i]->mutable_cpu_data() + slice_offset;
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
  int step = gradvec_[0]->shape()[slice_dim];
  for (size_t i = slice_dim + 1; i < gradvec_[0]->shape().size(); ++i)
    step *= gradvec_[0]->shape()[i];
  int srclayer_offset = 0;
  int slice_offset = 0;
  while (srclayer_offset < blob->count()) {
    for (int i = 0; i < num_slices; ++i) {
      const float* src = gradvec_[i]->cpu_data() + slice_offset;
      float* dst = blob->mutable_cpu_data() + srclayer_offset;
      memcpy(dst, src, step * sizeof(float));
      srclayer_offset += step;
    }
    slice_offset += step;
  }
}

const Blob<float>& SliceLayer::data(const Layer* from) const {
  int id = from ? from->partition_id() : 0;
  CHECK_LT(id, num_slices);
  return *datavec_[id];
}

const Blob<float>& SliceLayer::grad(const Layer* from) const {
  int id = from ? from->partition_id() : 0;
  CHECK_LT(id, num_slices);
  return *gradvec_[id];
}

Blob<float>* SliceLayer::mutable_data(const Layer* from) {
  int id = from ? from->partition_id() : 0;
  CHECK(from);
  CHECK_LT(id, num_slices);
  return datavec_[id];
}

Blob<float>* SliceLayer::mutable_grad(const Layer* from) {
  int id = from ? from->partition_id() : 0;
  CHECK_LT(id, num_slices);
  return gradvec_[id];
}
const std::string SliceLayer::ToString(bool debug, int flag) {
  if (!debug)
    return "";
  string ret = "";
  if ((flag & kForward) == kForward && data_.count() !=0) {
    for (unsigned k = 0; k < datavec_.size(); k++)
      ret += StringPrintf("data-%u :%13.9f ", k, Asum(*datavec_.at(k)));
  }
  if ((flag & kBackward) == kBackward && grad_.count() != 0) {
    for (unsigned k = 0; k < gradvec_.size(); k++)
    ret += StringPrintf("grad-%u:%13.9f ", k, Asum(*gradvec_.at(k)));
  }
  return ret;
}
}  // namespace singa
