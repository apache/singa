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

#include "./rnn.h"
#include <vector>
#include "singa/model/layer.h"
#include "singa/utils/string.h"

namespace singa {

void RNN::Setup(const Shape& in_sample, const LayerConf &conf) {
  Layer::Setup(in_sample, conf);

  RNNConf rnn_conf = conf.rnn_conf();
  hidden_dim_ = rnn_conf.hidden_dim();
  CHECK_GT(hidden_dim_, 0u);
  num_stacks_ = rnn_conf.num_stacks();
  CHECK_GT(num_stacks_, 0u);
  input_dim_ = Product(in_sample);
  CHECK_GT(input_dim_, 0u);
  dropout_ = rnn_conf.dropout();
  CHECK_GE(dropout_, 0);

  input_mode_ = ToLowerCase(rnn_conf.input_mode());
  CHECK(input_mode_ == "linear" || input_mode_ == "skip")
      << "Input mode of " << input_mode_ << " is not supported; Please use "
      << "'linear' and 'skip'";

  direction_ = ToLowerCase(rnn_conf.direction());
  if (direction_ == "unidirectional")
    num_directions_ = 1;
  else if (direction_ == "bidirectional")
    num_directions_ = 2;
  else
    LOG(FATAL) << "Direction of " << direction_
      << " is not supported; Please use unidirectional or bidirectional";

  rnn_mode_ = ToLowerCase(rnn_conf.rnn_mode());
  if (rnn_mode_ == "lstm") {
    has_cell_ = true;
  } else if (rnn_mode_ !="relu" && rnn_mode_ != "tanh" && rnn_mode_ != "gru") {
    LOG(FATAL) << "RNN memory unit (mode) of " << rnn_mode_
      << " is not supported Please use 'relu', 'tanh', 'lstm' and 'gru'";
  }
  // the first constant (4) is the size of float
  // the second constant (2, 8, 6) is the number of sets of params
  int mult = 1;
  if (rnn_mode_ == "relu" || rnn_mode_ == "tanh")
    mult *= 1;
  else if (rnn_mode_ == "lstm")
    mult *= 4;
  else if (rnn_mode_ == "gru")
    mult *= 3;
  if (direction_ == "bidirectional")
    mult *= 2;

  size_t weight_size = 0;
  for (size_t i = 0; i < num_stacks_; i++) {
    size_t dim = hidden_dim_ * (in_sample[0] +  hidden_dim_ + 2);
    if (i > 0)
      dim = hidden_dim_ * (hidden_dim_ +  hidden_dim_ + 2);
    weight_size += mult * dim;
  }
  weight_.Reshape(Shape{weight_size});
}

const vector<Tensor> RNN::Forward(int flag, const vector<Tensor>& inputs) {
  vector<Tensor> data_output;
  return data_output;
}

const std::pair<vector<Tensor>, vector<Tensor>> RNN::Backward(int flag,
    const vector<Tensor>& grads) {
  vector<Tensor> param_grad;
  vector<Tensor> data_grad;
  return std::make_pair(data_grad, param_grad);
}

void RNN::ToDevice(std::shared_ptr<Device> device) {
  Layer::ToDevice(device);
  weight_.ToDevice(device);
}
}  /* singa */
