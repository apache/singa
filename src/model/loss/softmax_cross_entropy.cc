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

#include <stack>

#include "singa/model/loss.h"

namespace singa {

Tensor SoftmaxCrossEntropy::Forward(int flag, const Tensor& prediction,
                                    const Tensor& target) {
  CHECK(buf_.empty()) << "Do not call Forward successively for more than twice."
                      << " The calling pattern is [Forward|Evaluate] Backward";
  size_t batchsize = 1;
  if (prediction.nDim() == 2) batchsize = prediction.shape(0);
  size_t dim = prediction.Size() / batchsize;
  const Tensor& input = Reshape(prediction, Shape{batchsize, dim});
  Tensor prob = SoftMax(input);
  // LOG(INFO) << "prob: " << prob.L2();

  // buffer intermediate data
  if (flag & kTrain) {
    buf_.push(prob);
    buf_.push(target);
  }
  Tensor loss(Shape{batchsize}, prob.device(), prob.data_type());

  ComputeCrossEntropy(prob, target, &loss);

  return loss;
}

Tensor SoftmaxCrossEntropy::Backward() {
  const Tensor target = buf_.top();
  buf_.pop();
  Tensor prob = buf_.top();
  buf_.pop();
  SoftmaxCrossEntropyBwd(target, &prob);
  return prob;
}
}  // namespace singa
