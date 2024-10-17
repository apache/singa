/*********************************************************
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 ************************************************************/
#ifndef SINGA_MODEL_LAYER_BATCHNORM_H
#define SINGA_MODEL_LAYER_BATCHNORM_H
#include <stack>

#include "singa/core/common.h"
#include "singa/model/layer.h"
#include "singa/proto/core.pb.h"

namespace singa {
class BatchNorm : public Layer {
 public:
  /// \copydoc Layer::layer_type()
  // const std::string layer_type() const override { return "BatchNorm"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const Shape& in_sample, const LayerConf& conf) override;
  const Shape GetOutputSampleShape() const override {
    CHECK(out_sample_shape_.size()) << "You may haven't call Setup()";
    return out_sample_shape_;
  }

  const Tensor Forward(int flag, const Tensor& input) override;

  /// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
  const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                   const Tensor& grad) override;
  virtual const std::vector<Tensor> param_values() override {
    return std::vector<Tensor>{bnScale_, bnBias_, runningMean_,
                               runningVariance_};
  }
  float factor() const { return factor_; }
  const Tensor& bnScale() const { return bnScale_; }
  const Tensor& bnBias() const { return bnBias_; }
  const Tensor& runningMean() const { return runningMean_; }
  const Tensor& runningVariance() const { return runningVariance_; }
  size_t channels() const { return channels_; }
  size_t height() const { return height_; }
  size_t width() const { return width_; }
  void set_bnScale(const Tensor& x) {
    bnScale_.ResetLike(x);
    bnScale_.CopyData(x);
  }
  void set_bnBias(const Tensor& x) {
    bnBias_.ResetLike(x);
    bnBias_.CopyData(x);
  }
  void set_runningMean(const Tensor& x) {
    runningMean_.ResetLike(x);
    runningMean_.CopyData(x);
  }
  void set_runningVariance(const Tensor& x) {
    runningVariance_.ResetLike(x);
    runningVariance_.CopyData(x);
  }
  virtual void ToDevice(std::shared_ptr<Device> device) override;

 protected:
  float factor_;
  size_t channels_, height_, width_;
  bool is_2d_ = false;
  Tensor bnScale_, bnBias_;
  Tensor dbnScale_, dbnBias_;
  Tensor runningMean_, runningVariance_;
  // Store intermediate data, i.e., input tensor
  std::stack<Tensor> buf_;
  Shape out_sample_shape_;
};  // class batchnorm
}  // namespace singa

#endif  // SINGA_MODEL_LAYER_BATCHNORM_H
