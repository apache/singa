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

#include <glog/logging.h>
#include "singa/neuralnet/neuron_layer.h"
#include "singa/utils/singleton.h"


namespace singa {

using std::vector;

void LRNLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  lsize_ = conf.lrn_conf().local_size();
  CHECK_EQ(lsize_ % 2, 1) << "LRN only supports odd values for Localvol";
  knorm_ = conf.lrn_conf().knorm();
  alpha_ = conf.lrn_conf().alpha();
  beta_ = conf.lrn_conf().beta();
  const vector<int>& s = srclayers[0]->data(this).shape();
  data_.Reshape(s);
  grad_.Reshape(s);
  norm_.Reshape(s);
  batchsize_ = s[0];
  channels_ = s[1];
  height_ = s[2];
  width_ = s[3];
}

void LRNLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers[0]->mutable_data(this));
  auto data = Tensor4(&data_);
  auto norm = Tensor4(&norm_);
  // stores normalizer without power
  norm = expr::chpool<red::sum>(expr::F<op::square>(src), lsize_) * salpha
    + knorm_;
  data = src * expr::F<op::power>(norm, -beta_);
}

void LRNLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers[0]->mutable_data(this));
  auto norm = Tensor4(&norm_);
  auto grad = Tensor4(&grad_);
  auto gsrc = Tensor4(srclayers[0]->mutable_grad(this));

  gsrc = grad * expr::F<op::power>(norm, -beta_);
  Tensor<cpu, 4> tmp(gsrc.shape);
  AllocSpace(tmp);
  tmp = gsrc * src / norm;
  gsrc += (- 2.0f * beta_ * salpha) * expr::chpool<red::sum>(tmp, lsize_) * src;
  FreeSpace(tmp);
}

}  // namespace singa
