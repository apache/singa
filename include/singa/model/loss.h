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

#ifndef SINGA_MODEL_LOSS_H_
#define SINGA_MODEL_LOSS_H_
#include "singa/proto/model.pb.h"
#include "singa/core/tensor.h"
namespace singa {

/// The base loss class, which declares the APIs for computing the objective
/// score (loss) for a pair of prediction (from the model) and the target (i.e.
/// the ground truth). It also computes the gradients of the objective w.r.t.
/// the prediction. It has similar APIs as Layer.
template <typename T = Tensor>
class Loss {
 public:
  Loss() = default;
  void Setup(const string& conf) {
    LossConf loss;
    loss.ParseFromString(conf);
    Setup(loss);
  }
	virtual ~Loss(){};
  /// Set meta fields from user configurations.
  virtual void Setup(const LossConf& conf) {}

  /// Compute the loss values for each sample/instance given the prediction
  /// and the target.
  virtual Tensor Forward(const Tensor& prediction, const T& target) = 0;

  /// Average loss values for all samples in the mini-batch
  /// It calls Forward() internally. The calling pattern should be
  /// [Evaluate|Forward] Backward.
  float Evaluate(const Tensor& prediction, const T& target) {
    const Tensor& loss = Forward(prediction, target);
    return Sum<float>(loss) / (1.0f * loss.Size());
  }

  /// Compute the gradients of the loss values w.r.t. the prediction.
  virtual Tensor Backward() = 0;
};
}  // namespace singa

#endif  // SINGA_MODEL_LOSS_H_


