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

#include "./softmax.h"
namespace singa {

RegisterLayerClass(Softmax);
void Softmax::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  CHECK_EQ(in_sample.size(), 1u);
  out_sample_shape_ = in_sample;
}

const Tensor Softmax::Forward(int flag, const Tensor& input) {
  CHECK_LE(input.nDim(), 2u);
  Tensor output =  SoftMax(input);
  if (flag & kTrain)
    buf_.push(output);
  return output;
}

const std::pair<Tensor, vector<Tensor>> Softmax::Backward(int flag,
                                                          const Tensor& grad) {
  CHECK_LE(grad.nDim(), 2u);
  Tensor input_grad = grad.Clone();
  CHECK(!buf_.empty());
  Tensor y = buf_.top();
  buf_.pop();
  CHECK(y.shape() == input_grad.shape());
  Tensor sigma = input_grad * y;

  size_t nrow = 1, ncol = grad.Size();
  if (grad.nDim() > 1) {
    nrow = grad.shape(0);
    ncol = grad.shape(1);
  } else {
    input_grad.Reshape({nrow, ncol});
    sigma.Reshape({nrow, ncol});
  }
  Tensor sum(Shape{nrow}, grad.device(), grad.data_type());
  SumColumns(sigma, &sum);
  // dL / dy_i = grad_i
  // dy_i / dx_i = y_i - y_i^2, if i == j
  // dy_i / dx_j = - y_i * y_j, if i != j
  // dL / dx_i = sum_j((dL / dy_j) * (dy_j / dx_i))
  // dL / dx_i = y_i * (grad_i - sum), where sum = sum_i(grad_i * y_i);
  SubColumn(sum, &input_grad);
  input_grad = input_grad * y;
  if (grad.nDim() == 1)
    input_grad.Reshape(Shape{ncol});
  // Mult(input_grad, y, &input_grad);
  vector<Tensor> param_grad;
  return std::make_pair(input_grad, param_grad);
}

}  // namespace singa
