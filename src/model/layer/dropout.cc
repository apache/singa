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

#include "singa/model/layer.h"
#include "./dropout.h"
namespace singa {

RegisterLayerClass(Dropout);
void Dropout::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  dropout_ratio_ = conf.dropout_conf().dropout_ratio();
  out_sample_shape_= in_sample;
}

const Tensor Dropout::Forward(int flag, const Tensor& input) {
  Tensor out;
  if (flag & kTrain) {
    mask_.ResetLike(input);
    // set mask_[i] = 1 with prob 1-dropout_rato_
    Bernoulli(1.0f - dropout_ratio_, &mask_);
    mask_ *= 1.0f / (1.0f - dropout_ratio_);
    out = input * mask_;
  } else {
    out = input;
  }
  return out;
}

const std::pair<Tensor, vector<Tensor>> Dropout::Backward(int flag,
                                                          const Tensor& grad) {
  vector<Tensor> param_grad;
  Tensor input_grad;
  if (flag & kTrain) {
    // note mask is already scaled by 1/(1-dropout_ratio_)
    input_grad = grad * mask_;
  } else {
    LOG(ERROR) << "Do not call backward for evaluation phase";
  }
  return std::make_pair(input_grad, param_grad);
}

void Dropout::ToDevice(std::shared_ptr<Device> device) {
  Layer::ToDevice(device);
  mask_.ToDevice(device);
}

}  // namespace singa
