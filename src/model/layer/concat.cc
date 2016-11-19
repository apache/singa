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
#include "./concat.h"
namespace singa {

RegisterLayerClass(singa_concat, Concat);
RegisterLayerClass(singacpp_concat, Concat);
RegisterLayerClass(singacuda_concat, Concat);
RegisterLayerClass(singacl_concat, Concat);

void Concat::Setup(const vector<Shape>& in_shapes, const LayerConf& conf) {
  Layer::Setup(in_shapes, conf);
  dim_size_.clear();
  axis_ = conf.concat_conf().axis();
  out_sample_shape_ = {0, 0};
  out_sample_shape_[1 - axis_] = in_shapes[0][1 - axis_];
  for (auto& s: in_shapes) {
    out_sample_shape_[axis_] += s[axis_];
    dim_size_.push_back(s[axis_]);
    // LOG(ERROR) << s[axis_];
  }
}

const vector<Tensor> Concat::Forward(int flag, const vector<Tensor>& inputs) {
  vector<Tensor> outputs;
  if (inputs.size() == 1u) {
    outputs = inputs;
  } else {
    if(axis_ == 0)
      outputs.push_back(ConcatRows(inputs));
    else
      outputs.push_back(ConcatColumns(inputs));
  }
  return outputs;
}

const std::pair<vector<Tensor>, vector<Tensor>> Concat::Backward(
    int flag, const vector<Tensor>& grads) {
  vector<Tensor> input_grad, param_grad;
  CHECK_EQ(grads.size(), 1u) << "Concat layer only have one output tensor.";
  for (size_t i = 0, offset = 0; i < dim_size_.size(); i++) {
    if (axis_ == 0)
      input_grad.push_back(SliceRows(grads.at(0), offset,
            offset + dim_size_[i]));
    else
      input_grad.push_back(SliceColumns(grads.at(0), offset,
            offset + dim_size_[i]));
    offset += dim_size_[i];
  }
  return std::make_pair(input_grad, param_grad);
}

}  // namespace singa
