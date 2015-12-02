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

#include "singa/neuralnet/neuron_layer/inner_product.h"

#include <glog/logging.h>
#include "singa/utils/singleton.h"

namespace singa {

using std::vector;

InnerProductLayer::~InnerProductLayer() {
  delete weight_;
  delete bias_;
}

void InnerProductLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  const auto& src = srclayers[0]->data(this);
  batchsize_ = src.shape()[0];
  vdim_ = src.count() / batchsize_;
  hdim_ = layer_conf_.innerproduct_conf().num_output();
  transpose_ = conf.innerproduct_conf().transpose();
  if (partition_dim() > 0)
    hdim_ /= srclayers.at(0)->num_partitions();
  data_.resize(1);
  data_.at(0).Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_.at(0));
  weight_ = Param::Create(conf.param(0));
  bias_ = Param::Create(conf.param(1));
  if (transpose_)
    weight_->Setup(vector<int>{vdim_, hdim_});
  else
    weight_->Setup(vector<int>{hdim_, vdim_});
  bias_->Setup(vector<int>{hdim_});
}

void InnerProductLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  auto data = Tensor2(&data_.at(0));
  auto src = Tensor2(srclayers[0]->mutable_data(this));
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  if (transpose_)
    data = dot(src, weight);
  else
    data = dot(src, weight.T());
  // repmat: repeat bias vector into batchsize rows
  data += expr::repmat(bias, batchsize_);
}

void InnerProductLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  auto src = Tensor2(srclayers[0]->mutable_data(this));
  auto grad = Tensor2(&grad_);
  auto weight = Tensor2(weight_->mutable_data());
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  gbias = expr::sum_rows(grad);
  if (transpose_)
    gweight = dot(src.T(), grad);
  else
    gweight = dot(grad.T(), src);
  if (srclayers[0]->mutable_grad(this) != nullptr) {
    auto gsrc = Tensor2(srclayers[0]->mutable_grad(this));
    if (transpose_)
      gsrc = dot(grad, weight.T());
    else
      gsrc = dot(grad, weight);
  }
}

}  // namespace singa
