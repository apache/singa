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
  slice_point_.clear();
  axis_ = conf.slice_conf().axis();
  int offset = 0;
  // #slice point = # out tensors - 1
  for (size_t p : conf.slice_conf().slice_point()) {
    slice_point_.push_back(p);
    if (axis_ == 1) {
      out_sample_shapes_.push_back({p - offset});
      offset = p;
    } else {
      out_sample_shapes_.push_back(in_sample);
    }
  }
  slice_point_.push_back(in_sample[0]);
  if (axis_ == 1) {
    out_sample_shapes_.push_back({in_sample[0] - offset});
  } else {
    out_sample_shapes_.push_back(in_sample);
  }
}

const vector<Tensor> Slice::Forward(int flag, const vector<Tensor>& inputs) {
  vector<Tensor> outputs;
  CHECK_EQ(inputs.size(), 1u) << "Split layer only have one input tensor.";
  size_t offset = 0;
  for (auto& s : slice_point_) {
    if (axis_ == 0)
      outputs.push_back(SliceRows(inputs.at(0), offset, s));
    else
      outputs.push_back(SliceColumns(inputs.at(0), offset, s));
    offset = s;
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
