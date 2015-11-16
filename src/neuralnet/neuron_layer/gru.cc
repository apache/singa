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

#include "singa/neuralnet/neuron_layer/gru.h"

#include <glog/logging.h>
#include "singa/utils/singleton.h"

namespace singa {

using std::vector;

GRULayer::~GRULayer() {
  delete weight_z_hx_;
  delete weight_z_hh_;
  delete bias_z_; 

  delete weight_r_hx_;
  delete weight_r_hh_;
  delete bias_r_;

  delete weight_c_hx_; 
  delete weight_c_hh_;
  delete bias_c_; 
}

void GRULayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_LE(srclayers.size(), 2);
  const auto& src = srclayers[0]->data(this);
  
  batchsize_ = src.shape()[0]; // size of batch
  vdim_ = src.count() / (batchsize_); // dimension of input

  hdim_ = layer_conf_.gru_conf().dim_hidden(); // dimension of hidden state

  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);

  // Initialize the parameters
  weight_z_hx_ = Param::Create(conf.param(0));
  weight_z_hh_ = Param::Create(conf.param(1));
  bias_z_ = Param::Create(conf.param(2));

  weight_r_hx_ = Param::Create(conf.param(3));
  weight_r_hh_ = Param::Create(conf.param(4));
  bias_r_ = Param::Create(conf.param(5)); 

  weight_c_hx_ = Param::Create(conf.param(6));
  weight_c_hh_ = Param::Create(conf.param(7));
  bias_c_ = Param::Create(conf.param(8));
  
  
  weight_z_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_z_hh_->Setup(vector<int>{hdim_, hdim_});
  bias_z_->Setup(vector<int>{hdim_});

  weight_r_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_r_hh_->Setup(vector<int>{hdim_, hdim_});
  bias_r_->Setup(vector<int>{hdim_});

  weight_c_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_c_hh_->Setup(vector<int>{hdim_, hdim_});
  bias_c_->Setup(vector<int>{hdim_});

}

void GRULayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
/*  auto data = Tensor3(&data_);
  auto src = Tensor3(srclayers[0]->mutable_data(this));

  auto weight_z_hx = Tensor2(weight_z_hx_->mutable_data());
  auto weight_z_hh = Tensor2(weight_z_hh_->mutable_data());
  auto bias_z = Tensor1(bias_z_->mutable_data());

  auto weight_r_hx = Tensor2(weight_r_hx_->mutable_data());
  auto weight_r_hh = Tensor2(weight_r_hh_->mutable_data());
  auto bias_r = Tensor1(bias_r_->mutable_data());

  auto weight_c_hx = Tensor2(weight_c_hx_->mutable_data());
  auto weight_c_hh = Tensor2(weight_c_hh_->mutable_data());
  auto bias_c = Tensor1(bias_c_->mutable_data());

  auto init_hidden;

  for (int t = 0; t < tdim_; t++) {
    auto prev_hidden = init_hidden; 
    if (t > 0) prev_hidden = 
    auto z_t = expr::F<op::sigmoid>(dot(src[t], weight_z_hx.T())
                 + dot(prev_hidden, weight_z_hh.T())
                 + bias_z);
    auto r_t = expr::F<op::sigmoid>(dot(src[t], weight_r_hx.T())
                 + dot(prev_hidden, weight_r_hh.T())
                 + bias_r);
    auto c_t = expr::F<op::stanh>(dot(src[t], weight_c_hx.T()
                 + dot(prev_hidden, weight_c_hh)); 
  }
  
  if (transpose_)
    data = dot(src, weight);
  else
    data = dot(src, weight.T());
  // repmat: repeat bias vector into batchsize rows
  data += expr::repmat(bias, batchsize_);
  */
}

void GRULayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
/*  auto src = Tensor2(srclayers[0]->mutable_data(this));
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
  */
}

}  // namespace singa
