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
#ifndef SRC_MODEL_LAYER_RNN_H_
#define SRC_MODEL_LAYER_RNN_H_

#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "singa/model/layer.h"

namespace singa {
/// To enable use the same layer multiple times in one iteration in RNN,
/// the Forward() function pushes the 'input' or 'output' that are
/// necessary for Backward() in a stack (states_). If neither 'input' or
/// 'output' is used by Backward(), then do not store them. The Backward()
/// pops data from the states_ stack to compute gradients. Users are
/// responsible for accumulating the gradients for the same parameters.
class RNN : public Layer {
 public:
  /// \copydoc Layer::layer_type()
  // const std::string layer_type() const override { return "RNN"; }

  /// Setup the RNN layer.
  /// in_shape is the shape of a single training instance from one timestep,
  void Setup(const Shape& in_shape, const LayerConf& conf) override;

  /// The inputs vector includes <x1, ... xn, hx, cx> where xi is the input
  /// tensor at the i-th time step. hx is used to initialize the hidden tensor,
  /// which could be a dummy tensor (like Tensor hx;). cx is used to initialize
  /// the cell tensor, which could be a dummy tensor( like Tensor cx;). For
  /// dummy tensors, 0's would be used during computation.
  /// cx is missing for gru/relu/tanh RNNs, and is valid for lstm.
  /// The dim order of xi is <batch, feature>, and the batchsize of xi must be
  /// >= that of x(i+1).
  /// The output vector includes <y1, ... yn, hy, cy> where yi is the output
  /// tensor at the i-th time step. hy is the final hidden tensor, cy is the
  /// final cell tensor. cy is missing for gru/relu/tanh RNNs and is valid for
  /// lstm.
  const vector<Tensor> Forward(int flag, const vector<Tensor>& inputs) override;

  /// The grads vector includes <dy1, dy2, ... dyn, dhy, dcy>, the symbols are
  /// similar to those for Forward. dcy is missing for gru/relu/tanh RNNs and is
  /// valid for lstm.
  /// The first vector of the output includes <dx1, dx2, ... dxn, dhx, dcx>.
  /// The second vector of the output includes the gradients of all parameters.
  const std::pair<vector<Tensor>, vector<Tensor>> Backward(
      int flag, const vector<Tensor>& grads) override;

  const vector<Tensor> param_values() override {
    return vector<Tensor>{weight_};
  }

  void ToDevice(std::shared_ptr<Device> device) override;
  /// Return the internal state stack, which should be empty at the beginning
  /// of one iteration.
  // std::stack<Tensor> states() const { return states_; }

  string input_mode() const { return input_mode_; }
  string direction() const { return direction_; }
  string rnn_mode() const { return rnn_mode_; }

 protected:
  /// Storing input or output from Forward(), which are used in Backward().
  /// Rules:
  /// 1. push the 'input' or 'output' into states_ if the flag of Forward() is
  ///    for kTrain and 'input' or 'output' is necessary for Backward().
  /// 2. pop data out in Backward().
  std::stack<Tensor> buf_;
  bool has_cell_ = false;
  size_t num_directions_ = 1;
  size_t input_size_ = 0, hidden_size_ = 0, num_stacks_ = 0, seq_length_ = 0,
         max_length_ = 0;
  size_t batch_size_ = 0;
  size_t seed_ = 0x1234567;
  float dropout_ = 0.0f;
  string input_mode_, direction_, rnn_mode_;
  Tensor weight_;
};
}  // namespace singa
#endif  // SRC_MODEL_LAYER_RNN_H_
