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

#include <algorithm>

#include "singa/model/metric.h"
namespace singa {

Tensor Accuracy::Match(const Tensor& predict, const vector<int>& target) {
  Tensor prediction(predict.shape());
  prediction.CopyData(predict);
  size_t batchsize = target.size();
  size_t nb_classes = prediction.Size() / batchsize;
  // each row of prediction is the prob distribution for one sample
  CHECK_EQ(prediction.shape().at(0), batchsize);
  // TODO(wangwei) CloneToDevice(host);
  const float* prob = prediction.data<float>();
  float* score = new float[batchsize];
  memset(score, 0, batchsize * sizeof(float));
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
  delete[] score;
  return ret;
}

// TODO(wangwei) consider multi-label cases, where target is of shape
// nb_samples * nb_classes
Tensor Accuracy::Forward(const Tensor& prediction, const Tensor& t) {
  Tensor target(t.shape(), t.data_type());
  target.CopyData(t);
  vector<int> target_vec;
  // TODO(wangwei) copy target to host.
  const int* target_value = target.data<int>();
  for (size_t i = 0; i < target.Size(); i++)
    target_vec.push_back(target_value[i]);
  return Match(prediction, target_vec);
}

}  // namespace singa
