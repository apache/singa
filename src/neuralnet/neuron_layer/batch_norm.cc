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

#include "singa/neuralnet/neuron_layer/batch_norm.h"

#include <glog/logging.h>
#include <cmath>
#include "singa/utils/singleton.h"
#include "singa/utils/math_blob.h"

namespace singa {
    
using std::vector;

void BatchNormLayer::Setup(const LayerProto& conf,
    const vector<Layer *>& srclayers) {
  Layer::Setup(conf, srclayers);

  auto src = srclayers[0]->data(this);
  batchsize_ = src.shape()[0];
  dim_ = src.count()/src.shape()[0];
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(srclayers[0]->grad(this));
  var_.Reshape(vector<int> {dim_});
  mean_.ReshapeLike(var_);
  tmpx_.ReshapeLike(data_);
  tmptx_.ReshapeLike(data_);
  tmp_.ReshapeLike(var_);
  xnorm_.ReshapeLike(data_);
  gvar_.ReshapeLike(var_);
  gmean_.ReshapeLike(var_);
  gxnorm_.ReshapeLike(xnorm_);
  gamma_ = Param::Create(conf.param(0));
  beta_ = Param::Create(conf.param(1));
  gamma_->Setup(vector<int>{dim_});
  beta_->Setup(vector<int>{dim_});
}

void BatchNormLayer::ComputeFeature(int flag, 
    const vector<Layer*>& srclayers) {
  const Blob<float>& src_ = srclayers[0]->data(this);
  //compute mean
  MVSumRow<float>(cpu, 1.0/batchsize_, .0, src_, &mean_);
  //compute var
  Copy<float>(cpu, src_, &xnorm_);
  MVAddRow<float>(cpu, -1.0, 1.0, mean_, &xnorm_);
  Map<op::Square<float>, float>(cpu, xnorm_, &tmpx_);
  MVSumRow<float>(cpu, 1.0/batchsize_, .0, tmpx_, &var_);
  //compute xnorm 
  tmp_.SetValue(1e-5);
  AXPY<float>(cpu, 1.0, var_, &tmp_);
  Map<op::Sqrt<float>, float>(cpu, tmp_, &tmp_);
  RepmatRow<float>(cpu, tmp_, &tmpx_);
  Div<float>(cpu, xnorm_, tmpx_, &xnorm_);
  //compute data
  RepmatRow<float>(cpu, gamma_->data(), &tmpx_);
  Mult<float>(cpu, xnorm_, tmpx_, &data_);
  MVAddRow<float>(cpu, 1.0, 1.0, beta_->data(), &data_);
}

void BatchNormLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  const Blob<float>& src_ = srclayers[0]->data(this);
  
  //compute gxnorm
  RepmatRow<float>(cpu, gamma_->data(), &tmpx_);
  Mult<float>(cpu, grad_, tmpx_, &gxnorm_);
  //compute gvar
  Copy<float>(cpu, src_, &tmptx_);
  MVAddRow<float>(cpu, -1.0, 1.0, mean_, &tmptx_);
  Mult<float>(cpu, gxnorm_, tmptx_, &tmpx_);
  tmp_.SetValue(1e-5);
  AXPY<float>(cpu, 1.0, var_, &tmp_);
  Map<op::Pow<float>, float>(cpu, -1.5, tmp_, &tmp_);
  Map<op::Mult<float>, float>(cpu, -0.5, tmp_, &tmp_);
  RepmatRow<float>(cpu, tmp_, &tmptx_);
  Mult<float>(cpu, tmpx_, tmptx_, &tmpx_);
  MVSumRow<float>(cpu, 1.0, .0, tmpx_, &gvar_);
  //compute gmean
  tmp_.SetValue(1e-5);
  AXPY<float>(cpu, 1.0, var_, &tmp_);
  Map<op::Pow<float>, float>(cpu, -0.5, tmp_, &tmp_);
  Map<op::Mult<float>, float>(cpu, -1.0, tmp_, &tmp_);
  RepmatRow<float>(cpu, tmp_, &tmpx_);
  Mult<float>(cpu, tmpx_, gxnorm_, &tmpx_);
  MVSumRow<float>(cpu, 1.0, .0, tmpx_, &gmean_);
  //compute gsrc
  if (srclayers[0]->mutable_grad(this) != nullptr) {
    auto gsrc_ = srclayers[0]->mutable_grad(this);
    tmp_.SetValue(1e-5);
    AXPY<float>(cpu, 1.0, var_, &tmp_);
    Map<op::Pow<float>, float>(cpu, -0.5, tmp_, &tmp_);
    RepmatRow<float>(cpu, tmp_, &tmpx_);
    Mult<float>(cpu, tmpx_, gxnorm_, gsrc_);
    Copy<float>(cpu, src_, &tmptx_);
    MVAddRow<float>(cpu, -1.0, 1.0, mean_, &tmptx_);
    RepmatRow<float>(cpu, gvar_, &tmpx_);
    Mult<float>(cpu, tmpx_, tmptx_, &tmpx_);
    AXPY<float>(cpu, 2.0/batchsize_, tmpx_, gsrc_);
    RepmatRow<float>(cpu, gmean_, &tmpx_);
    AXPY<float>(cpu, 1.0/batchsize_, tmpx_, gsrc_);
  }
  //compute ggamma
  Mult<float>(cpu, grad_, xnorm_, &tmpx_);
  MVSumRow<float>(cpu, 1.0, .0, tmpx_, gamma_->mutable_grad());
  //compute gbeta
  MVSumRow<float>(cpu, 1.0, .0, grad_, beta_->mutable_grad());
}
}  //  namespace singa
