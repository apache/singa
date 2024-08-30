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
#ifndef SINGA_MODEL_LAYER_CONCAT_H_
#define SINGA_MODEL_LAYER_CONCAT_H_
#include <string>
#include <utility>
#include <vector>

#include "singa/model/layer.h"

namespace singa {
/**
 * Cocnatenate tensors from the source layers along the give axis.
 */
class Concat : public Layer {
 public:
  /// \copydoc Layer::layer_type()
  // const std::string layer_type() const override { return "Concat"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const vector<Shape>& in_shapes, const LayerConf& conf);
  const Shape GetOutputSampleShape() const override {
    CHECK(out_sample_shape_.size()) << "You may haven't call Setup()";
    return out_sample_shape_;
  }

  /// \copydoc Layer::Forward(int flag, const Tensor&)
  const vector<Tensor> Forward(int flag, const vector<Tensor>& input) override;

  /// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
  const std::pair<vector<Tensor>, vector<Tensor>> Backward(
      int flag, const vector<Tensor>& grad) override;

 protected:
  /// >= 0
  int axis_ = 0;
  /// slice_point_[i] the end offset of the i-th source tensor on the concat
  /// axis of the result tensor
  vector<int> slice_point_;
  Shape out_sample_shape_;
};

}  // namespace singa
#endif  // SINGA_MODEL_LAYER_CONCAT_H_
