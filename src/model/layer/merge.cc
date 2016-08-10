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
#include "./merge.h"
namespace singa {

RegisterLayerClass(singa_merge, Merge);

void Merge::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  MergeConf merge_conf = conf.merge_conf();
  input_size_ = merge_conf.input_size();
  out_sample_shape_ = in_sample;
}

const vector<Tensor> Merge::Forward(int flag, const vector<Tensor>& inputs) {
  vector<Tensor> outputs;
  //input_size_ = inputs.size();
  if (input_size_ == 1u) {
    outputs = inputs;
  } else {
    Tensor sum = inputs.at(0);
    for (size_t i = 1; i < inputs.size(); i++) {
      Tensor temp = inputs.at(i);
      CHECK_EQ(sum.nDim(), temp.nDim());
      for (size_t j = 0; j < temp.nDim(); j++)
        CHECK_EQ(sum.shape(j), temp.shape(j));
      sum += temp;
    }
    outputs.push_back(sum);
  }
  return outputs;
}

const std::pair<vector<Tensor>, vector<Tensor>> Merge::Backward(
    int flag, const vector<Tensor>& grads) {
  vector<Tensor> input_grad, param_grad;
  if (grads.size() != 1u) {
    LOG(INFO) << "Merge layer only have one output tensor.";
  }
  for (size_t i = 0; i < input_size_; i++)
    input_grad.push_back(grads.at(0));
  return std::make_pair(input_grad, param_grad);
}

}  // namespace singa
