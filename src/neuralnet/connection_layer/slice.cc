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
  /*
  Layer::Setup(conf, npartitions);
  slice_dim_ = conf.slice_conf().slice_dim();
  slice_num_ = npartitions;
  CHECK_GE(slice_dim_, 0);
  CHECK_EQ(slice_num_, dstlayers_.size());
  data_.Reshape(srclayers[0]->data(this).shape());
  grad_.ReshapeLike(data_);
  datavec_.resize(slice_num_);
  gradvec_.resize(slice_num_);
  CHECK_EQ(data_.count() % slice_num_, 0);  // restrict equal slicing
  // LOG(ERROR)<<"slice dim "<<slice_dim<<" slice num "<<slice_num;
  for (int i = 0; i < slice_num_; i++) {
    vector<int> newshape(data_.shape());
    newshape[slice_dim_] = newshape[slice_dim_] / slice_num_ +
      ((i == slice_num_ - 1) ? newshape[slice_dim_] % slice_num_ : 0);
    datavec_[i].Reshape(newshape);
    gradvec_[i].Reshape(newshape);
    // LOG(ERROR)<<"slice "<<IntVecToString(newshape);
  }
  */
  data_.resize(1);
  LOG(FATAL) << "Not implemented";
}

void SliceLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  /*
  CHECK_EQ(srclayers.size(), 1);
  if (slice_dim_ == 0) {
    const auto& blob = srclayers.at(0)->data(this);
    int size = blob.count() / slice_num_;
    for (int i = 0; i < slice_num_; i++) {
      float* dst = datavec_[i].mutable_cpu_data();
      const float* src = blob.cpu_data() + i * size;
      memcpy(dst, src, size*sizeof(float));
    }
  }
  */
  LOG(FATAL) << "Not implemented";
}

void SliceLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented";
}

/*
int SliceLayer::SliceID(const Layer* layer) const {
  CHECK(layer != nullptr);
  for (size_t i = 0; i < datavec_.size(); i++) {
    // LOG(ERROR)<<"get slice "<<IntVecToString(shapes_[i]);
    if (dstlayers_[i] == layer)
      return i;
  }
  CHECK(false);
  return -1;
}*/

}  // namespace singa
