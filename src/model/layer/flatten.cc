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
#include "./flatten.h"
namespace singa {

RegisterLayerClass(Flatten);
void Flatten::Setup(const Shape& in_sample, const LayerConf &conf) {
  Layer::Setup(in_sample, conf);
  axis_ = conf.flatten_conf().axis();
  size_t len = 1;
  if (axis_ > 0)
    for (size_t i = axis_ - 1; i < in_sample.size(); i++)
      len *= in_sample.at(i);
  out_sample_shape_.push_back(len);
}

const Tensor Flatten::Forward(int flag, const Tensor &input) {
  Tensor output;
  input_shape_ = input.shape();
  if (axis_ == 0)
    output = Reshape(input, vector<size_t>{input.Size()});
  else
    output =
        Reshape(input, vector<size_t>{input.Size() / out_sample_shape_.at(0),
                                      out_sample_shape_.at(0)});
  return output;
}

const std::pair<Tensor, vector<Tensor> > Flatten::Backward(int flag,
                                                           const Tensor &grad) {
  vector<Tensor> param_grad;
  Tensor input_grad = grad;
  input_grad.Reshape(input_shape_);
  return std::make_pair(input_grad, param_grad);
}

} // namespace singa
