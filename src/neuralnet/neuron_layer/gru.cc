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
#include "singa/utils/singa_op.h"

using namespace std;

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

  delete update_gate_;
  delete reset_gate_;
  delete new_memory_;
  // delete reset_context_;
}

void GRULayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_LE(srclayers.size(), 2);
  const auto& src = srclayers[0]->data(this);

  batchsize_ = src.shape()[0];  // size of batch
  vdim_ = src.count() / (batchsize_);  // dimension of input

  hdim_ = layer_conf_.gru_conf().dim_hidden();  // dimension of hidden state

  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  // one for grad from dst GRU, one for grad from upper layer
  gradvec_.push_back(new Blob<float>(grad_.shape()));

  // Initialize the parameters
  weight_z_hx_ = Param::Create(conf.param(0));
  weight_r_hx_ = Param::Create(conf.param(1));
  weight_c_hx_ = Param::Create(conf.param(2));

  weight_z_hh_ = Param::Create(conf.param(3));
  weight_r_hh_ = Param::Create(conf.param(4));
  weight_c_hh_ = Param::Create(conf.param(5));

  if (conf.param_size() > 6) {
    bias_z_ = Param::Create(conf.param(6));
    bias_r_ = Param::Create(conf.param(7));
    bias_c_ = Param::Create(conf.param(8));
  }

  weight_z_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_r_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_c_hx_->Setup(vector<int>{hdim_, vdim_});

  weight_z_hh_->Setup(vector<int>{hdim_, hdim_});
  weight_r_hh_->Setup(vector<int>{hdim_, hdim_});
  weight_c_hh_->Setup(vector<int>{hdim_, hdim_});

  if (conf.param_size() > 6) {
    bias_z_->Setup(vector<int>{hdim_});
    bias_r_->Setup(vector<int>{hdim_});
    bias_c_->Setup(vector<int>{hdim_});
  }

  update_gate_ = new Blob<float>(batchsize_, hdim_);
  reset_gate_ = new Blob<float>(batchsize_, hdim_);
  new_memory_ = new Blob<float>(batchsize_, hdim_);
}

void GRULayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  CHECK_LE(srclayers.size(), 2);

  // Do transpose
  Blob<float> *w_z_hx_t = Transpose(weight_z_hx_->data());
  Blob<float> *w_z_hh_t = Transpose(weight_z_hh_->data());
  Blob<float> *w_r_hx_t = Transpose(weight_r_hx_->data());
  Blob<float> *w_r_hh_t = Transpose(weight_r_hh_->data());
  Blob<float> *w_c_hx_t = Transpose(weight_c_hx_->data());
  Blob<float> *w_c_hh_t = Transpose(weight_c_hh_->data());

  // Prepare the data input and the context
  const auto& src = srclayers[0]->data(this);
  const Blob<float> *context;
  if (srclayers.size() == 1) {  // only have data input
    context = new Blob<float>(batchsize_, hdim_);
  } else {  // have data input & context
    context = &srclayers[1]->data(this);
  }

  // Compute the update gate
  GEMM(1.0f, 0.0f, src, *w_z_hx_t, update_gate_);
  if (bias_z_ != nullptr)
    MVAddRow(1.0f, 1.0f, bias_z_->data(), update_gate_);
  GEMM(1.0f, 1.0f, *context, *w_z_hh_t, update_gate_);
  Map<op::Sigmoid<float>, float>(*update_gate_, update_gate_);
  // LOG(ERROR) << "Update Gate: " << update_gate_->cpu_data()[0];
  // Compute the reset gate
  GEMM(1.0f, 0.0f, src, *w_r_hx_t, reset_gate_);
  if (bias_r_ != nullptr)
    MVAddRow(1.0f, 1.0f, bias_r_->data(), reset_gate_);
  GEMM(1.0f, 1.0f, *context, *w_r_hh_t, reset_gate_);
  Map<op::Sigmoid<float>, float>(*reset_gate_, reset_gate_);
  // LOG(ERROR) << "Reset Gate: " << reset_gate_->cpu_data()[0];
  // Compute the new memory
  GEMM(1.0f, 0.0f, *context, *w_c_hh_t, new_memory_);
  Mult<float>(*reset_gate_, *new_memory_, new_memory_);
  GEMM(1.0f, 1.0f, src, *w_c_hx_t, new_memory_);
  if (bias_c_ != nullptr)
    MVAddRow(1.0f, 1.0f, bias_c_->data(), new_memory_);
  Map<op::Tanh<float>, float>(*new_memory_, new_memory_);

  Sub(*context, *new_memory_, &data_);
  Mult(data_, *update_gate_, &data_);
  Add(data_, *new_memory_, &data_);

  // delete the pointers
  if (srclayers.size() == 1)
    delete context;

  delete w_z_hx_t;
  delete w_z_hh_t;
  delete w_r_hx_t;
  delete w_r_hh_t;
  delete w_c_hx_t;
  delete w_c_hh_t;
}

void GRULayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  CHECK_LE(srclayers.size(), 2);
  // agg grad from two dst layers, gradvec_[0] is grad_
  AXPY(1.0f, *gradvec_[1], &grad_);
  float beta = 1.0f;  // agg param gradients

  Layer* ilayer = srclayers[0];  // input layer
  Layer* clayer = nullptr;  // context layer
  // Prepare the data input and the context
  const Blob<float>& src = ilayer->data(this);
  const Blob<float> *context;
  if (srclayers.size() == 1) {  // only have data input
    context = new Blob<float>(batchsize_, hdim_);
  } else {  // have data input & context
    clayer = srclayers[1];
    context = &(clayer->data(this));
  }

  // Compute intermediate gradients which are used for other computations
  Blob<float> dugatedz(batchsize_, hdim_);
  Map<singa::op::SigmoidGrad<float>, float>(*update_gate_, &dugatedz);
  Blob<float> drgatedr(batchsize_, hdim_);
  Map<singa::op::SigmoidGrad<float>, float>(*reset_gate_, &drgatedr);
  Blob<float> dnewmdc(batchsize_, hdim_);
  Map<singa::op::TanhGrad<float>, float>(*new_memory_, &dnewmdc);

  Blob<float> dLdz(batchsize_, hdim_);
  Sub<float>(*context, *new_memory_, &dLdz);
  Mult<float>(dLdz, grad_, &dLdz);
  Mult<float>(dLdz, dugatedz, &dLdz);

  Blob<float> dLdc(batchsize_, hdim_);
  Blob<float> z1(batchsize_, hdim_);
  z1.SetValue(1.0f);
  AXPY<float>(-1.0f, *update_gate_, &z1);
  Mult(grad_, z1, &dLdc);
  Mult(dLdc, dnewmdc, &dLdc);

  Blob<float> reset_dLdc(batchsize_, hdim_);
  Mult(dLdc, *reset_gate_, &reset_dLdc);

  Blob<float> dLdr(batchsize_, hdim_);
  Blob<float> cprev(batchsize_, hdim_);
  GEMM(1.0f, 0.0f, *context, weight_c_hh_->data().T(), &cprev);
  Mult(dLdc, cprev, &dLdr);
  Mult(dLdr, drgatedr, &dLdr);

  // Compute gradients for parameters of update gate
  Blob<float> *dLdz_t = Transpose(dLdz);
  GEMM(1.0f, beta, *dLdz_t, src, weight_z_hx_->mutable_grad());
  GEMM(1.0f, beta, *dLdz_t, *context, weight_z_hh_->mutable_grad());
  if (bias_z_ != nullptr)
    MVSumRow<float>(1.0f, beta, dLdz, bias_z_->mutable_grad());
  delete dLdz_t;

  // Compute gradients for parameters of reset gate
  Blob<float> *dLdr_t = Transpose(dLdr);
  GEMM(1.0f, beta, *dLdr_t, src, weight_r_hx_->mutable_grad());
  GEMM(1.0f, beta, *dLdr_t, *context, weight_r_hh_->mutable_grad());
  if (bias_r_ != nullptr)
    MVSumRow(1.0f, beta, dLdr, bias_r_->mutable_grad());
  delete dLdr_t;

  // Compute gradients for parameters of new memory
  Blob<float> *dLdc_t = Transpose(dLdc);
  GEMM(1.0f, beta, *dLdc_t, src, weight_c_hx_->mutable_grad());
  if (bias_c_ != nullptr)
    MVSumRow(1.0f, beta, dLdc, bias_c_->mutable_grad());
  delete dLdc_t;

  Blob<float> *reset_dLdc_t = Transpose(reset_dLdc);
  GEMM(1.0f, beta, *reset_dLdc_t, *context, weight_c_hh_->mutable_grad());
  delete reset_dLdc_t;

  // Compute gradients for data input layer
  if (srclayers[0]->mutable_grad(this) != nullptr) {
    GEMM(1.0f, 0.0f, dLdc, weight_c_hx_->data(), ilayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdz, weight_z_hx_->data(), ilayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdr, weight_r_hx_->data(), ilayer->mutable_grad(this));
  }

  if (clayer != nullptr && clayer->mutable_grad(this) != nullptr) {
    // Compute gradients for context layer
    GEMM(1.0f, 0.0f, reset_dLdc, weight_c_hh_->data(),
        clayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdr, weight_r_hh_->data(), clayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdz, weight_z_hh_->data(), clayer->mutable_grad(this));
    Add(clayer->grad(this), *update_gate_, clayer->mutable_grad(this));
    // LOG(ERROR) << "grad to prev gru " << Asum(clayer->grad(this));
  }

  if (srclayers.size() == 1)
    delete context;
}

}  // namespace singa
