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
#include "singa/utils/math_blob.h"
#include "singa/utils/singa_op.h"

#include <iostream>
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

  delete update_gate;
  delete reset_gate;
  delete new_memory;
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
  weight_r_hx_ = Param::Create(conf.param(1));
  weight_c_hx_ = Param::Create(conf.param(2));

  weight_z_hh_ = Param::Create(conf.param(3));
  weight_r_hh_ = Param::Create(conf.param(4));
  weight_c_hh_ = Param::Create(conf.param(5));

  if (conf.gru_conf().bias_term()) {
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

  if (conf.gru_conf().bias_term()) {
	  bias_z_->Setup(vector<int>{hdim_});
	  bias_r_->Setup(vector<int>{hdim_});
	  bias_c_->Setup(vector<int>{hdim_});
  }

  update_gate = new Blob<float>(batchsize_, hdim_);
  reset_gate = new Blob<float>(batchsize_, hdim_);
  new_memory = new Blob<float>(batchsize_, hdim_);

}

void GRULayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
	CHECK_LE(srclayers.size(), 2);

	// Do transpose
	Blob<float> *w_z_hx_t = Transpose (weight_z_hx_->data());
	Blob<float> *w_z_hh_t = Transpose (weight_z_hh_->data());
	Blob<float> *w_r_hx_t = Transpose (weight_r_hx_->data());
	Blob<float> *w_r_hh_t = Transpose (weight_r_hh_->data());
	Blob<float> *w_c_hx_t = Transpose (weight_c_hx_->data());
	Blob<float> *w_c_hh_t = Transpose (weight_c_hh_->data());

	// Prepare the data input and the context
	const auto& src = srclayers[0]->data(this);
	const Blob<float> *context;
	if (srclayers.size() == 1) { // only have data input
		context = new Blob<float>(batchsize_, hdim_);
	} else { // have data input & context
		context = &srclayers[1]->data(this);
	}

	// Compute the update gate
	GEMM(cpu, 1.0f, 0.0f, src,*w_z_hx_t,update_gate);
	if (bias_z_ != nullptr)
		MVAddRow(cpu,1.0f,1.0f,bias_z_->data(),update_gate);
	Blob<float> zprev (batchsize_,hdim_);
	GEMM(cpu, 1.0f, 0.0f, *context,*w_z_hh_t, &zprev);
	Add<float>(cpu, *update_gate, zprev, update_gate);
	Map<op::Sigmoid<float>,float>(cpu, *update_gate, update_gate);

	// Compute the reset gate
	GEMM(cpu, 1.0f, 0.0f, src,*w_r_hx_t,reset_gate);
	if (bias_r_ != nullptr)
		MVAddRow(cpu,1.0f,1.0f,bias_r_->data(),reset_gate);
	Blob<float> rprev (batchsize_, hdim_);
	GEMM(cpu, 1.0f, 0.0f, *context, *w_r_hh_t, &rprev);
	Add<float>(cpu, *reset_gate, rprev, reset_gate);
	Map<op::Sigmoid<float>,float>(cpu, *reset_gate, reset_gate);

	// Compute the new memory
	GEMM(cpu, 1.0f, 0.0f, src, *w_c_hx_t, new_memory);
	if (bias_c_ != nullptr)
		MVAddRow(cpu, 1.0f,1.0f,bias_c_->data(), new_memory);
	Blob<float> cprev (batchsize_, hdim_);
	GEMM(cpu, 1.0f, 0.0f, *context, *w_c_hh_t, &cprev);
	Mult<float>(cpu, *reset_gate, cprev, &cprev);
	Add<float>(cpu, *new_memory, cprev, new_memory);
	Map<op::Tanh<float>,float>(cpu, *new_memory, new_memory);

	// Compute data - new memory part
	Blob<float> z1 (batchsize_,hdim_);
	for (int i = 0; i < z1.count(); i ++) {
		z1.mutable_cpu_data()[i] = 1.0f; // generate a matrix with ones
	}
	AXPY<float>(cpu, -1.0f, *update_gate, &z1);
	Mult<float>(cpu, z1, *new_memory, &data_);

	// Compute data - context part
	Blob<float> data_prev (batchsize_, hdim_);
	Mult<float>(cpu,*update_gate,*context,&data_prev);
	Add<float>(cpu, data_, data_prev, &data_);

	// delete the pointers
	if (srclayers.size() == 1) delete context;
	else context = NULL;

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

	// Prepare the data input and the context
	const Blob<float>& src = srclayers[0]->data(this);
	const Blob<float> *context;
	if (srclayers.size() == 1) { // only have data input
		context = new Blob<float>(batchsize_, hdim_);
	} else { // have data input & context
		context = &srclayers[1]->data(this);
	}

	// Prepare gradient of output neurons
	Blob<float> *grad_t = Transpose (grad_);

	// Compute intermediate gradients which are used for other computations
	Blob<float> dugatedz (batchsize_, hdim_);
	Map<singa::op::SigmoidGrad<float>, float>(cpu, *update_gate, &dugatedz);
	Blob<float> drgatedr (batchsize_, hdim_);
	Map<singa::op::SigmoidGrad<float>, float>(cpu, *reset_gate, &drgatedr);
	Blob<float> dnewmdc (batchsize_, hdim_);
	Map<singa::op::TanhGrad<float>, float>(cpu, *new_memory,&dnewmdc);

	Blob<float> dLdz (batchsize_, hdim_);
	Sub<float>(cpu, *context, *new_memory, &dLdz);
	Mult<float>(cpu,dLdz, grad_, &dLdz);
	Mult<float>(cpu,dLdz, dugatedz, &dLdz);

	Blob<float> dLdc (batchsize_,hdim_);
	Blob<float> z1 (batchsize_,hdim_);
	for (int i = 0; i < z1.count(); i ++) {
		z1.mutable_cpu_data()[i] = 1.0f; // generate a matrix with ones
	}
	AXPY<float>(cpu, -1.0f, *update_gate, &z1);
	Mult(cpu,grad_,z1,&dLdc);
	Mult(cpu,dLdc,dnewmdc,&dLdc);

	Blob<float> reset_dLdc (batchsize_,hdim_);
	Mult(cpu,dLdc, *reset_gate, &reset_dLdc);

	Blob<float> dLdr (batchsize_, hdim_);
	Blob<float> cprev (batchsize_, hdim_);
	Blob<float> *w_c_hh_t = Transpose(weight_c_hh_->data());
	GEMM(cpu,1.0f,0.0f,*context,*w_c_hh_t, &cprev);
	delete w_c_hh_t;
	Mult(cpu,dLdc,cprev,&dLdr);
	Mult(cpu,dLdr,drgatedr,&dLdr);


	// Compute gradients for parameters of update gate
	Blob<float> *dLdz_t = Transpose(dLdz);
	GEMM(cpu,1.0f,0.0f,*dLdz_t,src,weight_z_hx_->mutable_grad());
	GEMM(cpu,1.0f,0.0f,*dLdz_t,*context,weight_z_hh_->mutable_grad());
	if (bias_z_ != nullptr)
		MVSumRow<float>(cpu,1.0f,0.0f,dLdz,bias_z_->mutable_grad());
	delete dLdz_t;

	// Compute gradients for parameters of reset gate
	Blob<float> *dLdr_t = Transpose(dLdr);
	GEMM(cpu,1.0f,0.0f,*dLdr_t,src,weight_r_hx_->mutable_grad());
	GEMM(cpu,1.0f,0.0f,*dLdr_t,*context,weight_r_hh_->mutable_grad());
	if (bias_r_ != nullptr)
		MVSumRow(cpu,1.0f,0.0f,dLdr,bias_r_->mutable_grad());
	delete dLdr_t;

	// Compute gradients for parameters of new memory
	Blob<float> *dLdc_t = Transpose(dLdc);
	GEMM(cpu,1.0f,0.0f,*dLdc_t,src,weight_c_hx_->mutable_grad());
	if (bias_c_ != nullptr)
		MVSumRow(cpu,1.0f,0.0f,dLdc,bias_c_->mutable_grad());
	delete dLdc_t;

	Blob<float> *reset_dLdc_t = Transpose(reset_dLdc);
	GEMM(cpu,1.0f,0.0f,*reset_dLdc_t,*context,weight_c_hh_->mutable_grad());
	delete reset_dLdc_t;

	// Compute gradients for data input layer
	if (srclayers[0]->mutable_grad(this) != nullptr) {
		GEMM(cpu,1.0f,0.0f,dLdc,weight_c_hx_->data(),srclayers[0]->mutable_grad(this));
		GEMM(cpu,1.0f,1.0f,dLdz,weight_z_hx_->data(),srclayers[0]->mutable_grad(this));
		GEMM(cpu,1.0f,1.0f,dLdr,weight_r_hx_->data(), srclayers[0]->mutable_grad(this));
	}

	if (srclayers.size() > 1 && srclayers[1]->mutable_grad(this) != nullptr) {
		// Compute gradients for context layer
		GEMM(cpu,1.0f,0.0f,reset_dLdc,weight_c_hh_->data(), srclayers[1]->mutable_grad(this));
		GEMM(cpu,1.0f,1.0f,dLdr, weight_r_hh_->data(), srclayers[1]->mutable_grad(this));
		GEMM(cpu,1.0f,1.0f,dLdz,weight_z_hh_->data(), srclayers[1]->mutable_grad(this));
		Add(cpu,srclayers[1]->grad(this), *update_gate, srclayers[1]->mutable_grad(this));
	}

	if (srclayers.size() == 1) delete context;
	else context = NULL;
	delete grad_t;
}

}  // namespace singa
