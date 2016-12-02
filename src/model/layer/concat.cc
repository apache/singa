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
  out_sample_shape_.clear();
  slice_point_.clear();
  axis_ = conf.concat_conf().axis();
  if (axis_ == 0)
    out_sample_shape_.push_back(in_shapes[0][0]);
  else {
    size_t l = 0;
    for (auto& s: in_shapes) {
       l += s[0];
    }
    out_sample_shape_.push_back(l);
  }
}

const vector<Tensor> Concat::Forward(int flag, const vector<Tensor>& inputs) {
  vector<Tensor> outputs;
  slice_point_.clear();
  size_t offset = 0;
  for (auto& x : inputs) {
    offset += x.shape(axis_);
    slice_point_.push_back(offset);
  }
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
  size_t last_offset = 0u;
  for (auto p : slice_point_) {
    if (axis_ == 0)
      input_grad.push_back(SliceRows(grads.at(0), last_offset, p));
    else
      input_grad.push_back(SliceColumns(grads.at(0), last_offset, p));
    last_offset = p;
  }
  return std::make_pair(input_grad, param_grad);
}

}  // namespace singa
