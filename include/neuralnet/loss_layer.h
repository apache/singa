/************************************************************
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
*************************************************************/

#ifndef SINGA_NEURALNET_LOSS_LAYER_H_
#define SINGA_NEURALNET_LOSS_LAYER_H_

#include <vector>
#include "neuralnet/layer.h"

/**
 * @file this file includes the declarations of layers that inherit the base
 * LossLayer for measuring the objective training loss.
 */
namespace singa {
using std::vector;
/**
 * Squared Euclidean loss as @f$0.5 ||p - t||^2@f$, where p is for prediction
 * t is for ground truth.
 */
class EuclideanLossLayer : public LossLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
};

/**
 * Cross-entropy loss applied to the probabilities computed from Softmax.
 * @f$ L_i = -log P_{t_i}, t_i\in [0, C] @f$ is the label for the i-th object,
 * C is the total number of classes.
 */
class SoftmaxLossLayer : public LossLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

  /**
   * softmax is not recommendeded for partition because it requires the whole
   * src layer for normalization.
   */
  ConnectionType src_neuron_connection(int k) const override {
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }

 private:
  int batchsize_;
  int dim_;
  float scale_;
  int topk_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_LOSS_LAYER_H_
