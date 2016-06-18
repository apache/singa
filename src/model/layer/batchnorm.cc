/*********************************************************
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
************************************************************/
#include "batchnorm.h"

namespace singa {
RegisterLayerClass(BatchNorm);
void BatchNorm::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  out_sample_shape_ = in_sample;
  factor_ = conf.batchnorm_conf().factor();
  channels_ = in_sample.at(0);
  height_ = in_sample.at(1);
  width_ = in_sample.at(2);

  bnScale_.Reshape(Shape{channels_ * height_ * width_});
  bnBias_.ResetLike(bnScale_);
  runningMean_.ResetLike(bnScale_);
  runningVariance_.ResetLike(bnScale_);

  dbnScale_.ResetLike(bnScale_);
  dbnBias_.ResetLike(bnBias_);
  // Push back params into param_values_
  // Assume the order of param is: bnScale, bnBias, runningMean, runningVariance
  for (const auto &spec : conf.param()) param_specs_.push_back(spec);
  param_values_.push_back(&bnScale_);
  param_values_.push_back(&bnBias_);
  param_values_.push_back(&runningMean_);
  param_values_.push_back(&runningVariance_);
}

void BatchNorm::ToDevice(Device* device) {
  bnScale_.ToDevice(device);
  bnBias_.ToDevice(device);
  dbnScale_.ToDevice(device);
  dbnBias_.ToDevice(device);
  runningMean_.ToDevice(device);
  runningVariance_.ToDevice(device);
}

const Tensor BatchNorm::Forward(int flag, const Tensor& input) {
  LOG(FATAL) << "Not implemented";
  Tensor output;
  return output;
}

const std::pair<Tensor, vector<Tensor>> BatchNorm::Backward(
    int flag, const Tensor& grad) {
  LOG(FATAL) << "Not implemented";
  Tensor dx;
  vector<Tensor> param_grad;
  return std::make_pair(dx, param_grad);
}

}  // namespace
