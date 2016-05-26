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

#ifndef SINGA_MODEL_METRIC_H_
#define SINGA_MODEL_METRIC_H_
#include "singa/core/tensor.h"
#include "singa/proto/model.pb.h"
namespace singa {

/// The base metric class, which declares the APIs for computing the performance
/// evaluation metrics given the prediction of the model and the ground truth,
/// i.e., the target.
/// The target type is a template argument.  For data samples with a single
/// label, T could be 1-d tenor (or vector<int>); If each data sample has
/// multiple labels, T could be vector<vector<int>>, one vector per sample.
template <typename T = Tensor>
class Metric {
 public:
  // TODO(wangwei) call Setup using a default MetricConf.
  Metric() = default;
  void Setup(const string& conf) {
    MetricConf metric;
    metric.ParseFromString(conf);
    Setup(metric);
  }

  /// Set meta fields from user configurations.
  virtual void Setup(const MetricConf& conf) {}

  /// Compute the metric for each data sample
  virtual Tensor Forward(const Tensor& prediction, const T& target) = 0;

  /// Comptue the metric value averaged over all samples (in a batch)
  float Evaluate(const Tensor& prediction, const T& target) {
    const Tensor& metric = Forward(prediction, target);
    return Sum<float>(metric) / (1.0f * metric.Size());
  }
};

}  // namespace singa

#endif  // SINGA_MODEL_METRIC_H_
