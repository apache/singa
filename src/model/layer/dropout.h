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
#ifndef SRC_MODEL_LAYER_DROPOUT_H_
#define SRC_MODEL_LAYER_DROPOUT_H_
#include <string>
#include <utility>
#include <vector>

#include "singa/model/layer.h"

namespace singa {
class Dropout : public Layer {
 public:
  /// \copydoc Layer::layer_type()
  // const std::string layer_type() const override { return "Dropout"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const Shape& in_sample, const LayerConf& conf) override;
  const Shape GetOutputSampleShape() const override {
    CHECK(out_sample_shape_.size()) << "You may haven't call Setup()";
    return out_sample_shape_;
  }

  /// \copydoc Layer::Forward(int flag, const Tensor&)
  /// if flag is kTrain, then do dropout with given dropout_ratio;
  /// otherwise if it is kEval, copy input directly to the output
  /// TODO(wangwei) There are diff implementations, Caffe vs
  /// <a
  /// href="https://github.com/nitishsrivastava/deepnet/blob/master/deepnet/fastdropoutnet.py">
  const Tensor Forward(int flag, const Tensor& input) override;

  /// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
  const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                   const Tensor& grad) override;

  void ToDevice(std::shared_ptr<Device> device) override;

  float dropout_ratio() const { return dropout_ratio_; }

  const Tensor& mask() const { return mask_; }

 protected:
  /// the proability to set each element to 0.
  float dropout_ratio_;
  Tensor mask_;
  vector<size_t> out_sample_shape_;
};
}  // namespace singa
#endif  // SRC_MODEL_LAYER_DROPOUT_H_
