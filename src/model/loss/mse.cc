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

#include "singa/model/loss.h"

namespace singa {

Tensor MSE::Forward(int flag, const Tensor& prediction, const Tensor& target) {
  CHECK(buf_.empty()) << "Do not call Forward successively for more than twice."
                      << " The calling pattern is [Forward|Evaluate] Backward";
  Tensor t = prediction - target;
  size_t batchsize = 1;
  if (t.nDim() > 1) batchsize = t.shape().at(0);
  size_t dim = t.Size() / batchsize;
  t.Reshape(Shape{batchsize, dim});
  if (kTrain & flag) buf_.push(t);
  // TODO(wangwei) use CastType for operator/
  return Sum(Square(t), 1) * 0.5f;
}

Tensor MSE::Backward() {
  Tensor ret = buf_.top();
  buf_.pop();
  return ret * (1.0f / ret.shape().at(0));
}
}  // namespace singa
