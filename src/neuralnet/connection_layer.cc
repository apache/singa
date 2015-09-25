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

#include "neuralnet/connection_layer.h"

namespace singa {

using std::vector;
/********* Implementation for BridgeDstLayer **************/
void BridgeDstLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  Layer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  data_.Reshape(srclayers[0]->data(this).shape());
  grad_.ReshapeLike(data_);
}
/************* Implementation for ConcateLayer ***********/
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
  data_.Reshape(shape);
  grad_.Reshape(shape);
}

void ConcateLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented for Concate Layer";
}

void ConcateLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented for Concate Layer";
}

/************* Implementation for SliceLayer****************/
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

/************* Implementation for SplitLayer****************/
void SplitLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  data_.Reshape(srclayers[0]->data(this).shape());
  grad_.Reshape(srclayers[0]->data(this).shape());
}

void SplitLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented";
}

void SplitLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented";
}
}  // namespace singa
