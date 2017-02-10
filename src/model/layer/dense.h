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
  // const std::string layer_type() const override { return "Dense"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const Shape& in_sample, const LayerConf& conf) override;
  const Shape GetOutputSampleShape() const override {
    CHECK(hdim_) << "You may haven't call Setup()";
    return vector<size_t>{hdim_};
  }

  /// \copydoc Layer::Forward(int flag, const Tensor&)
  const Tensor Forward(int flag, const Tensor& input) override;

  /// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
  const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                   const Tensor& grad) override;

  void ToDevice(std::shared_ptr<Device> device) override;
  const std::vector<Tensor> param_values() override {
    if (bias_term_)
      return std::vector<Tensor>{weight_, bias_};
    else
      return std::vector<Tensor>{weight_};
  }
  size_t num_output() const { return hdim_; }
  size_t num_input() const { return vdim_; }
  bool transpose() const { return transpose_; }
  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }

  void set_weight(Tensor w) {
    weight_.ResetLike(w);
    weight_.CopyData(w);
  }
  void set_bias(Tensor b) {
    bias_.ResetLike(b);
    bias_.CopyData(b);
  }

 protected:
  /// Used in auto-encoder, where the decoder would share its weight matrix from
  /// the encoder's transposed weight matrix.
  bool transpose_ = false;
  /// use bias or not;
  bool bias_term_ = true;
  size_t vdim_, hdim_;
  Tensor weight_, bias_;
  // Tensor data_, grad_;
  std::stack<Tensor> buf_;
};
}  // namespace singa
#endif  // SRC_MODEL_LAYER_DENSE_H_
