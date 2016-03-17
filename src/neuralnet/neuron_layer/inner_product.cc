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
#include "singa/utils/math_blob.h"

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
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
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
  if (transpose_)
    MMDot(srclayers[0]->data(this), weight_->data(), &data_);
  else
    MMDot(srclayers[0]->data(this), weight_->data().T(), &data_);
  MVAddRow(bias_->data(), &data_);
}

void InnerProductLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  float beta = 0.0f;
  if (flag & kAggGrad)
    beta = 1.0f;
  MVSumRow(1.0f, beta, grad_, bias_->mutable_grad());
  if (transpose_)
    GEMM(1.0f, beta, srclayers[0]->data(this).T(), grad_,
        weight_->mutable_grad());
  else
    GEMM(1.0f, beta, grad_.T(), srclayers[0]->data(this),
        weight_->mutable_grad());

  if (srclayers[0]->mutable_grad(this) != nullptr) {
    if (transpose_)
      MMDot(grad_, weight_->data().T(), srclayers[0]->mutable_grad(this));
    else
      MMDot(grad_, weight_->data(), srclayers[0]->mutable_grad(this));
  }
  //clee auto w = weight_->mutable_cpu_data();
  //LOG(ERROR) << srclayers[0]->name() << " " << w[0];
}
}  // namespace singa
