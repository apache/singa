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
#ifndef SRC_MODEL_LAYER_DENSE_H_
#define SRC_MODEL_LAYER_DENSE_H_
#include <string>
#include <utility>
#include <vector>
#include <stack>
#include "singa/model/layer.h"

namespace singa {
class Dense : public Layer {
 public:
  ~Dense();
  /// \copydoc Layer::layer_type()
  const std::string layer_type() const override { return "Dense"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const LayerConf& conf) override;

  void SetupParam(const Tensor& input);
  /// \copydoc Layer::Forward(int flag, const Tensor&)
  const Tensor Forward(int flag, const Tensor& input) override;

  /// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
  const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                   const Tensor& grad) override;

  void ToDevice(Device* device) override;
  
  size_t num_output() const { return hdim_; }
  size_t num_input() const { return vdim_; }
  bool transpose() const { return transpose_; }
  const Tensor &weight() const { return weight_; }
  const Tensor &bias() const { return bias_; }

  void set_weight(Tensor w) {
    weight_.ResetLike(w);
    weight_.CopyData(w);
  }
  void set_bias(Tensor b) {
    bias_.ResetLike(b);
    bias_.CopyData(b);
  }

protected:
  size_t batchsize_, vdim_, hdim_;
  bool transpose_;
  Tensor weight_, bias_;
  // Tensor data_, grad_;
  std::stack<Tensor> buf_;
};
}  // namespace singa
#endif  // SRC_MODEL_LAYER_DENSE_H_
