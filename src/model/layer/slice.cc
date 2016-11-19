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
#include "./slice.h"
namespace singa {

RegisterLayerClass(singa_slice, Slice);
RegisterLayerClass(singacpp_slice, Slice);
RegisterLayerClass(singacuda_slice, Slice);
RegisterLayerClass(singacl_slice, Slice);

void Slice::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  out_sample_shapes_.clear();
  axis_ = conf.slice_conf().axis();
  int offset = 0;
  // #slice point = # out tensors - 1
  for (size_t p : conf.slice_conf().slice_point()) {
    Shape s{0, 0};
    s[1 - axis_] = in_sample[1 - axis_];
    s[axis_] = p - offset;
    offset = p;
    out_sample_shapes_.push_back(s);
  }
  Shape s{0, 0};
  s[1 - axis_] = in_sample[1 - axis_];
  s[axis_] = in_sample[axis_] - offset;
  out_sample_shapes_.push_back(s);
}

const vector<Tensor> Slice::Forward(int flag, const vector<Tensor>& inputs) {
  vector<Tensor> outputs;
  CHECK_EQ(inputs.size(), 1u) << "Split layer only have one input tensor.";
  size_t offset = 0;
  for (auto& s : out_sample_shapes_) {
    if (axis_ == 0)
      outputs.push_back(SliceRows(inputs.at(0), offset, offset + s[axis_]));
    else
      outputs.push_back(SliceColumns(inputs.at(0), offset, offset + s[axis_]));
    offset += s[axis_];
  }
  return outputs;
}

const std::pair<vector<Tensor>, vector<Tensor>> Slice::Backward(
    int flag, const vector<Tensor>& grads) {
  vector<Tensor> input_grad, param_grad;
  CHECK_EQ(grads.size(), out_sample_shapes_.size());
  if (axis_ == 0)
    input_grad.push_back(ConcatRows(grads));
  else
    input_grad.push_back(ConcatColumns(grads));
  return std::make_pair(input_grad, param_grad);
}

}  // namespace singa
