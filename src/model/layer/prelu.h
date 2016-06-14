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
#ifndef SINGA_MODEL_LAYER_PRELU_H_
#define SINGA_MODEL_LAYER_PRELU_H_
#include <utility>
#include <string>
#include <vector>
#include "singa/model/layer.h"
#include "singa_config.h"

namespace singa {
class PReLU : public Layer {
 public:
  /// \copydoc Layer::layer_type()
  const std::string layer_type() const override { return "PReLU"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const LayerConf &conf) override;

  /// \copydoc Layer::Forward(int flag, const Tensor&)
  const Tensor Forward(int flag, const Tensor &input) override;

  /// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
  const std::pair<Tensor, vector<Tensor> > Backward(
      int flag, const Tensor &grad) override;

  void ToDevice(Device *device);

  const bool Channel_shared() const { return channel_shared_; }
  const Tensor A() const { return a_; }
  const std::string Format() const { return format_; }

  void Set_a(Tensor a) {
    a_.ResetLike(a);
    a_.CopyData(a);
  }

 protected:
  bool channel_shared_;
  std::string format_;  // format_ has two valid value, i.e. NCHW, NHWC
  Tensor a_;            // shape of a_ is 2D, i.e. (channels, 1)
  std::stack<Tensor> buf_;
};
}  // namespace singa
#endif  // SINGA_MODEL_LAYER_PRELU_H_
