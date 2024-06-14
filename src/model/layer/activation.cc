/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "./activation.h"

#include "singa/model/layer.h"
#include "singa/utils/string.h"
namespace singa {

RegisterLayerClass(singa_relu, Activation);
RegisterLayerClass(singa_sigmoid, Activation);
RegisterLayerClass(singa_tanh, Activation);

RegisterLayerClass(singacpp_relu, Activation);
RegisterLayerClass(singacuda_relu, Activation);
RegisterLayerClass(singacl_relu, Activation);
RegisterLayerClass(singacpp_sigmoid, Activation);
RegisterLayerClass(singacuda_sigmoid, Activation);
RegisterLayerClass(singacl_sigmoid, Activation);
RegisterLayerClass(singacpp_tanh, Activation);
RegisterLayerClass(singacuda_tanh, Activation);
RegisterLayerClass(singacl_tanh, Activation);

void Activation::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  auto pos = conf.type().find_first_of('_');
  CHECK_NE(pos, string::npos)
      << "There should be a '_' in the laye type " << conf.type();
  mode_ = ToLowerCase(conf.type().substr(pos + 1));
  if (mode_ != "relu" && mode_ != "sigmoid" && mode_ != "tanh")
    LOG(FATAL) << "Unkown activation type: " << conf.type() << " " << mode_
               << ". Please use singa_relu, singa_sigmoid, or singa_tanh";
  if (mode_ == "relu") {
    neg_slope_ = conf.relu_conf().negative_slope();
  }
  out_sample_shape_ = in_sample;
}

const Tensor Activation::Forward(int flag, const Tensor& input) {
  Tensor output;
  if (mode_ == "sigmoid") {
    output = Sigmoid(input);
    if (flag & kTrain) buf_.push(output);
  } else if (mode_ == "tanh") {
    output = Tanh(input);
    if (flag & kTrain) buf_.push(output);
  } else if (mode_ == "relu") {
    output = ReLU(input);
    if (flag & kTrain) buf_.push(input);
  } else
    LOG(FATAL) << "Unknown activation: " << mode_;
  return output;
}

const std::pair<Tensor, vector<Tensor>> Activation::Backward(
    int flag, const Tensor& grad) {
  vector<Tensor> param_grad;
  CHECK(!buf_.empty());
  // inout means either input or output, but only one is valid for an
  // activation.
  Tensor input_grad, inout = buf_.top();
  buf_.pop();
  if (mode_ == "sigmoid")
    input_grad = grad * inout * (inout * (-1.f) + 1.f);
  else if (mode_ == "tanh")
    input_grad = grad * (inout * inout * (-1.f) + 1.f);
  else if (mode_ == "relu")
    input_grad = grad * (inout > 0.f) + (inout <= 0.f) * neg_slope_;
  else
    LOG(FATAL) << "Unkown activation: " << mode_;
  return std::make_pair(input_grad, param_grad);
}

}  // namespace singa
