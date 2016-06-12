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

void Flatten::Setup(const LayerConf &conf) {
  Layer::Setup(conf);
  axis_ = conf.flatten_conf().axis();
}

const Tensor Flatten::Forward(int flag, const Tensor &input) {
  Tensor output = input;
  input_shape_ = input.shape();
  if (!Axis()) {
    // reshape to 1D
    size_t dim = output.Size();
    output.Reshape(Shape{dim});
    output_shape_ = Shape{dim};
  } else {
    // reshape to 2D
    size_t dim1 = 1, dim2;
    for (int i = 0; i < Axis(); i++) dim1 *= output.shape(i);
    dim2 = output.Size() / dim1;
    output.Reshape(Shape{dim1, dim2});
    output_shape_ = Shape{dim1, dim2};
  }
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
