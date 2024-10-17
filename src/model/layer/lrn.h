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
#ifndef SINGA_MODEL_LAYER_LRN_H_
#define SINGA_MODEL_LAYER_LRN_H_
#include <stack>

#include "singa/model/layer.h"

namespace singa {
class LRN : public Layer {
 public:
  /// \copydoc Layer::layer_type()
  // const std::string layer_type() const override { return "LRN"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const Shape& in_sample, const LayerConf& conf) override;
  const Shape GetOutputSampleShape() const override {
    CHECK(out_sample_shape_.size()) << "You may haven't call Setup()";
    return out_sample_shape_;
  }

  /**
   * Local Response Normalization edge
   *
   * @f$ b_i=a_i/x_i^beta @f$
   * @f$x_i=k+alpha*\sum_{j=max(0,i-n/2)}^{min(N,i+n/2)}(a_j)^2 @f$
   * n is size of local response area.
   * @f$a_i@f$, the activation (after ReLU) of a neuron convolved with the i-th
   * kernel.
   * @f$b_i@f$, the neuron after normalization, N is the total num of kernels
   */
  const Tensor Forward(int flag, const Tensor& input) override;

  /// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
  const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                   const Tensor& grad) override;

  int local_size() const { return local_size_; }
  float alpha() const { return alpha_; }
  float beta() const { return beta_; }
  float k() const { return k_; }

 protected:
  //!< hyper-parameter: size local response (neighbor) area
  int local_size_;
  //!< other hyper-parameters
  float alpha_, beta_, k_;
  // store intermediate data, i.e., input tensor
  std::stack<Tensor> buf_;
  Shape out_sample_shape_;

};  // class LRN
}  // namespace singa

#endif  // SINGA_MODEL_LAYER_LRN_H_
