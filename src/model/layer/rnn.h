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

#include <utility>
#include <string>
#include <vector>
#include <stack>

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
  const std::string layer_type() const override { return "RNN"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const vector<size_t>& in_shape, const LayerConf& conf) override;

  /// \copydoc Layer::Forward(int flag, const vector<Tensor>&)
  const vector<Tensor> Forward(int flag, const vector<Tensor>& inputs) override;

  /// \copydoc Layer::Backward(int, const vector<Tensor>&);
  const std::pair<vector<Tensor>, vector<Tensor>> Backward(
      int flag, const vector<Tensor>& grads) override;


  size_t hiddenSize() const { return hiddenSize_; }
  size_t numLayers() const { return numLayers_; }
  size_t weightSize() const { return weightSize_; }
  float dropout() const { return dropout_; }
  
  void set_weight(Tensor w) {
    weight_.ResetLike(w);
    weight_.CopyData(w);
  }


  void ToDevice(std::shared_ptr<Device> device) override;
  /// Return the internal state stack, which should be empty at the beginning
  /// of
  /// one iteration.
  // std::stack<Tensor> states() const { return states_; }

 protected:
  /// Storing input or output from Forward(), which are used in Backward().
  /// Rules:
  /// 1. push the 'input' or 'output' into states_ if the flag of Forward() is
  ///    for kTrain and 'input' or 'output' is necessary for Backward().
  /// 2. pop data out in Backward().
  // std::stack<Tensor*> states_;
  std::stack<Tensor> buf_;
  size_t hiddenSize_;
  size_t numLayers_;
  size_t numLinearLayer_;
  size_t seqLength_;
  size_t weightSize_; /*all the weights and biases*/
  float dropout_;
  Tensor weight_;
};
}  // namespace singa
#endif  // SRC_MODEL_LAYER_RNN_H_
