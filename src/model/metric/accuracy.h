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

#ifndef SINGA_MODEL_METRIC_ACCURACY_H_
#define SINGA_MODEL_METRIC_ACCURACY_H_
#include "singa/model/metric.h"
#include <algorithm>
namespace singa {

/// Compute the accuray of the prediction, which is matched against the
/// ground truth labels.
/// TODO(wangwei) consider multi-label cases.
class Accuracy : public Metric<Tensor> {
 public:
  /// Set meta fields from user configurations.
  void Setup(const MetricConf& conf) override { top_k_ = conf.top_k(); }

  /// Check the prediction against the target (ground truth) for each data
  /// sample. The returned Tensor has a float value for each sample, 0 for wrong
  /// and 1 for correct. Users can call Sum(const Tensor&) / Tensor::Size() to
  /// get the accuracy.
  Tensor Forward(const Tensor& prediction, const Tensor& target);

 private:
  /// \copydoc Match(const Tensor&, const Tensor&);
  Tensor Match(const Tensor& prediction, const vector<int>& target);
  /// If the ground truth label is in the top k predicted labels, then the
  /// prediction is correct.
  size_t top_k_ = 1;
};

Tensor Accuracy::Match(const Tensor& prediction, const vector<int>& target) {
  size_t batchsize = target.size();
  size_t nb_classes = prediction.Size() / batchsize;
  // each row of prediction is the prob distribution for one sample
  CHECK_EQ(prediction.shape().at(0), batchsize);
  // TODO(wangwei) CloneToDevice(host);
  const float* prob = prediction.data<float>();
  float* score = new float[batchsize];
  for (size_t b = 0; b < batchsize; b++) {
    vector<std::pair<float, int>> prob_class;
    for (size_t c = 0; c < nb_classes; c++) {
      prob_class.push_back(std::make_pair(prob[b * nb_classes + c], c));
    }
    std::partial_sort(prob_class.begin(), prob_class.begin() + top_k_,
                      prob_class.end(), std::greater<std::pair<float, int>>());

    for (size_t k = 0; k < top_k_; k++)
      if (prob_class.at(k).second == target.at(b)) score[b] = 1;
  }
  Tensor ret(Shape{batchsize});
  ret.CopyDataFromHostPtr(score, batchsize);
  return ret;
}

// TODO(wangwei) consider multi-label cases, where target is of shape
// nb_samples * nb_classes
Tensor Accuracy::Forward(const Tensor& prediction, const Tensor& target) {
  vector<int> target_vec;
  // TODO(wangwei) copy target to host.
  const int* target_value = target.data<int>();
  for (size_t i = 0; i < target.Size(); i++)
    target_vec.push_back(target_value[i]);
  return Match(prediction, target_vec);
}

}  // namespace singa

#endif  // SINGA_MODEL_METRIC_ACCURACY_H_
