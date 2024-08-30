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
#include <stack>

#include "singa/core/tensor.h"
#include "singa/proto/model.pb.h"
namespace singa {

/// The base loss class, which declares the APIs for computing the objective
/// score (loss) for a pair of prediction (from the model) and the target (i.e.
/// the ground truth). It also computes the gradients of the objective w.r.t.
/// the prediction. It has similar APIs as Layer.
// template <typename T = Tensor>
class Loss {
 public:
  Loss() = default;
  void Setup(const string &conf) {
    LossConf loss;
    loss.ParseFromString(conf);
    Setup(loss);
  }
  virtual ~Loss(){};
  virtual void ToDevice(std::shared_ptr<Device> device) {}
  /// Set meta fields from user configurations.
  virtual void Setup(const LossConf &conf) {}

  /// Compute the loss values for each sample/instance given the prediction
  /// and the target.
  virtual Tensor Forward(int flag, const Tensor &prediction,
                         const Tensor &target) = 0;

  /// Average loss values for all samples in the mini-batch
  /// It calls Forward() internally. The calling pattern should be
  /// [Evaluate|Forward] Backward.
  float Evaluate(int flag, const Tensor &prediction, const Tensor &target) {
    Tensor loss = Forward(flag, prediction, target);
    return Sum<float>(loss) / (1.0f * loss.Size());
  }

  /// Compute the gradients of the loss values w.r.t. the prediction.
  virtual Tensor Backward() = 0;
};

// ============= Mean Squared Error ===========================================
/// MSE is for mean squared error or squared euclidean distance.
class MSE : public Loss {
 public:
  /// Compute the loss values for each sample/instance given the prediction
  /// and the target, which is 0.5/||prediction-target||^2
  /// Users can call Average(const Tensor&) to get the average
  /// loss value over all samples in the batch.
  Tensor Forward(int flag, const Tensor &prediction,
                 const Tensor &target) override;

  /// Compute the gradients of the loss values w.r.t. the prediction,
  /// which is (prediction-target)/batchsize
  Tensor Backward() override;

 private:
  // to buffer intermediate data, i.e., prediction-target
  std::stack<Tensor> buf_;
};

// ===============Softamx Cross Entropy =======================================
/// Softmax + cross entropy for multi-category classification
class SoftmaxCrossEntropy : public Loss {
 public:
  /// Compute the loss values for each sample/instance given the prediction
  /// and the target.
  ///
  /// If the target consists one integer per instance, i.e. the label index
  /// (dentoed as idx_truth), the loss is -log(p[idx_truth]), p[] is the
  /// probability for each category, computed from Softmax(prediction).
  /// If the target consists one array per instance (e.g., for multiple
  /// labels), the loss is -\sum_i (t[i] * log(p[i]) / \sum_j t[j], t[i]
  /// is the weight of the i-th label (e.g., 1: the instance has this label, 0:
  /// the instance does not have this label).
  ///
  /// Users can call Average(const Tensor&) to get the average
  /// loss value over all samples in the batch.
  Tensor Forward(int flag, const Tensor &prediction,
                 const Tensor &target) override;

  /// Compute the gradients of the loss values w.r.t. the prediction,
  /// which is: p[i] - t[i]/\sum_j t[j]
  Tensor Backward() override;

 private:
  // to buffer intermediate data, i.e., probability for each category and
  // the target (ground truth)
  std::stack<Tensor> buf_;
};

}  // namespace singa

#endif  // SINGA_MODEL_LOSS_H_
