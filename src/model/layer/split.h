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
#ifndef SRC_MODEL_LAYER_SPLIT_H_
#define SRC_MODEL_LAYER_SPLIT_H_
#include <string>
#include <utility>
#include <vector>

#include "singa/model/layer.h"

namespace singa {
/// Duplicate the input into multiple outputs
/// need to configure the number of outputs
class Split : public Layer {
 public:
  /// \copydoc Layer::layer_type()
  // const std::string layer_type() const override { return "Split"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const Shape& in_sample, const LayerConf& conf) override;
  const Shape GetOutputSampleShape() const override {
    CHECK(out_sample_shape_.size()) << "You may haven't call Setup()";
    return out_sample_shape_;
  }
  /// The inputs should have only one Tensor
  /// The outputs is a set of replicated Tensor
  const vector<Tensor> Forward(int flag, const vector<Tensor>& inputs) override;

  /// \copydoc Layer::Backward(int, const vector<Tensor>&);
  const std::pair<vector<Tensor>, vector<Tensor> > Backward(
      int flag, const vector<Tensor>& grads) override;

  size_t output_size() const { return output_size_; }

 protected:
  // To store the input and output(of forward) tensors
  Shape out_sample_shape_;
  size_t output_size_;
};
}  // namespace singa
#endif  // SRC_MODEL_LAYER_SPLIT_H_
