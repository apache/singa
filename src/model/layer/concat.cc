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

#include "./concat.h"

#include "singa/model/layer.h"
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
  CHECK_GE(axis_, 0);

  if (axis_ == 0) {
    out_sample_shape_ = in_shapes[0];
    size_t fea_size = Product(in_shapes[0]);
    for (auto& s : in_shapes) {
      CHECK_EQ(Product(s), fea_size)
          << "Feature length of all source samples " << "must be the same";
    }
  } else {
    out_sample_shape_ = in_shapes[0];
    size_t fea_size = Product(in_shapes[0]) / in_shapes[0][axis_ - 1];
    size_t l = 0;
    for (auto& s : in_shapes) {
      CHECK_GE(s.size(), axis_);
      l += s[axis_ - 1];
      CHECK_EQ(fea_size, Product(s) / s[axis_ - 1])
          << "Feature length for all axis except axis_ must be the same";
    }
    out_sample_shape_[axis_ - 1] = l;
  }
}

const vector<Tensor> Concat::Forward(int flag, const vector<Tensor>& inputs) {
  // TODO(wangwei) check the inputs shape to be the same for all iterations
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
    outputs.push_back(ConcatOn(inputs, axis_));
  }
  return outputs;
}

const std::pair<vector<Tensor>, vector<Tensor>> Concat::Backward(
    int flag, const vector<Tensor>& grads) {
  vector<Tensor> input_grad, param_grad;
  CHECK_EQ(grads.size(), 1u) << "Concat layer only have one output tensor.";
  size_t last_offset = 0u;
  for (auto p : slice_point_) {
    input_grad.push_back(SliceOn(grads.at(0), last_offset, p, axis_));
    last_offset = p;
  }
  return std::make_pair(input_grad, param_grad);
}

}  // namespace singa
