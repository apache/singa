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

#include "./split.h"

#include "singa/model/layer.h"
namespace singa {

RegisterLayerClass(singa_split, Split);
RegisterLayerClass(singacpp_split, Split);
RegisterLayerClass(singacuda_split, Split);
RegisterLayerClass(singacl_split, Split);

void Split::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  SplitConf split_conf = conf.split_conf();
  output_size_ = split_conf.output_size();
  out_sample_shape_ = in_sample;
}

const vector<Tensor> Split::Forward(int flag, const vector<Tensor>& inputs) {
  vector<Tensor> outputs;
  CHECK_EQ(inputs.size(), 1u) << "Split layer only have one input tensor.";
  for (size_t i = 0; i < output_size_; i++) outputs.push_back(inputs.at(0));
  return outputs;
}

const std::pair<vector<Tensor>, vector<Tensor>> Split::Backward(
    int flag, const vector<Tensor>& grads) {
  vector<Tensor> input_grad, param_grad;
  CHECK_EQ(grads.size(), output_size_);

  /// Input_grad is the sum of all the output gradients.
  Tensor temp = grads.at(0);
  for (size_t i = 1; i < output_size_; i++) temp += grads.at(i);
  input_grad.push_back(temp);
  return std::make_pair(input_grad, param_grad);
}

}  // namespace singa
