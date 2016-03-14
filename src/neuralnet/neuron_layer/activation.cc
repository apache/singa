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
#include "singa/neuralnet/neuron_layer.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/singa_op.h"
#include "singa/proto/job.pb.h"
namespace singa {

void ActivationLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  NeuronLayer::Setup(conf, srclayers);
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(data_);
  if (conf.share_src_blobs()) {
    data_.ShareData(srclayers[0]->mutable_data(this), false);
    grad_.ShareData(srclayers[0]->mutable_grad(this), false);
  }
}
void
ActivationLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  switch (layer_conf_.activation_conf().type()) {
    case RELU:
      Map<op::Relu<float>, float>(srclayers[0]->data(this), &data_);
      break;
    case SIGMOID:
      Map<op::Sigmoid<float>, float>(srclayers[0]->data(this), &data_);
      break;
    case TANH:
      Map<op::Tanh<float>, float>(srclayers[0]->data(this), &data_);
      break;
      /*
    case ActivationType_STANH:
      Map<op::STanh<float>, float>(srclayers[0]->data(this), &data_);
      break;
      */
    default:
      LOG(ERROR) << "Unknow activation type " <<
        layer_conf_.activation_conf().type();
  }
}
void
ActivationLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  Blob<float> * gsrc = srclayers[0]->mutable_grad(this);
  switch (layer_conf_.activation_conf().type()) {
    case RELU:
      Map<op::ReluGrad<float>, float>(data_, gsrc);
      Mult(*gsrc, grad_, gsrc);
      break;
    case SIGMOID:
      Map<op::SigmoidGrad<float>, float>(data_, gsrc);
      Mult(*gsrc, grad_, gsrc);
      break;
    case TANH:
      Map<op::TanhGrad<float>, float>(data_, gsrc);
      Mult(*gsrc, grad_, gsrc);
      break;
      /*
    case ActivationType_STANH:
      Map<op::STanhGrad<float>, float>(data_, gsrc);
      Mult(*gsrc, grad_, gsrc);
      break;
      */
    default:
      LOG(ERROR) << "Unknow activation type " <<
        layer_conf_.activation_conf().type();
  }
}

}  // namespace singa
