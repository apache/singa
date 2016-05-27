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

#ifndef SRC_MODEL_LOSS_CROSS_ENTROPY_H_
#define SRC_MODEL_LOSS_CROSS_ENTROPY_H_
#include <stack>
#include "singa/model/loss.h"

namespace singa {

/// Cross entropy is for cross entropy loss.
class CrossEntropy : public Loss<Tensor> {
 public:
  /// Compute the loss values for each sample/instance given the prediction
  /// and the target, which is sum {-log(prob_of_truth)}
  /// Users can call Average(const Tensor&) to get the average
  /// loss value over all samples in the batch.
  Tensor Forward(const Tensor& prediction, const Tensor& target) override;

  /// Compute the gradients of the loss values w.r.t. the prediction,
  /// which is: if the entry x corresponds to ground truth,
  /// then softmax(x) - 1; else, softmax(x)
  Tensor Backward() override;

 private:
  // to buffer intermediate data, i.e., softmax(prediction), target
  std::stack<Tensor> buf_;
};

Tensor CrossEntropy::Forward(const Tensor& prediction, const Tensor& target) {
  CHECK(buf_.empty()) << "Do not call Forward successively for more than twice."
                      << " The calling pattern is [Forward|Evaluate] Backward";

  size_t batchsize = 1;
  if (prediction.nDim() > 1) batchsize = prediction.shape().at(0);
  size_t dim = prediction.Size() / batchsize;
  // a temporal Softmax layer for forward computation
//  LayerConf conf; // TODO(kaiping): this is currently commented
//  Softmax softmax_tmp;
//  softmax_tmp.Setup(conf);
//  Tensor softmax = softmax_tmp.Forward(0, prediction);

  Tensor softmax(Shape{batchsize, dim});  // TODO(kaiping): Delete
//  softmax.SetValue<float>(0.5f); // TODO(kaiping): Delete

  softmax.Reshape(Shape{batchsize, dim});
  // buffer intermediate data
  buf_.push(softmax);
  buf_.push(target);

  // Compute loss for each sample
  Tensor loss(Shape{batchsize, 1});
  float * pre_ptr = reinterpret_cast<float*>(softmax.blob()->mutable_data());
  float * truth_ptr = reinterpret_cast<float*>(target.blob()->mutable_data());
  float * loss_ptr = reinterpret_cast<float*>(loss.blob()->mutable_data());
  for (size_t i = 0; i < batchsize; i++) {
    int ilabel = static_cast<int>(truth_ptr[i]);
    CHECK_GE(ilabel, 0);
    float prob_of_truth = pre_ptr[ilabel];
    loss_ptr[i] = -log(prob_of_truth);
    pre_ptr += dim;  // change to the next sample
  }
  return loss;
}

Tensor CrossEntropy::Backward() {
  const Tensor& target = buf_.top();
  buf_.pop();
  Tensor softmax = buf_.top();
  buf_.pop();

  size_t batchsize = 1;
  if (softmax.nDim() > 1)
    batchsize = softmax.shape().at(0);
  size_t dim = softmax.Size() / batchsize;
  float * truth_ptr = reinterpret_cast<float*>(target.blob()->mutable_data());
  float * pre_ptr = reinterpret_cast<float*>(softmax.blob()->mutable_data());
  for (size_t i = 0; i < batchsize; i++) {
    int ilabel = static_cast<int>(truth_ptr[i]);
    // CHECK_GE(ilabel, 0);
    pre_ptr[ilabel] -= 1.0;
    pre_ptr += dim;  // change to the next sample
  }
  return softmax;
}
}  // namespace singa

#endif  // SRC_MODEL_LOSS_CROSS_ENTROPY_H_


