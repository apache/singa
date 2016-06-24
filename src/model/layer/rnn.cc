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

namespace singa {

void RNN::Setup(const Shape& in_sample, const LayerConf &conf) {
  Layer::Setup(in_sample, conf);
  RNNConf rnn_conf = conf.rnn_conf();
  hiddenSize_ = rnn_conf.hiddensize();
  CHECK_GT(hiddenSize_, 0u);

  numLayers_ = rnn_conf.numlayers();
  CHECK_GT(numLayers_, 0u);

  dropout_ = rnn_conf.dropout();
  CHECK_GE(dropout_, 0u);
}

const vector<Tensor> RNN::Forward(int flag, const vector<Tensor>& inputs) {
  vector<Tensor> data_output;
  return data_output;
}

const std::pair<vector<Tensor>, vector<Tensor>> RNN::Backward(int flag, const vector<Tensor>& grads) {
  vector<Tensor> param_grad;
  vector<Tensor> data_grad;
  return std::make_pair(data_grad, param_grad);
}

void RNN::ToDevice(std::shared_ptr<Device> device) {
  Layer::ToDevice(device);
  weight_.ToDevice(device);
}
}  /* singa */
