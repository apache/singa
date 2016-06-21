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
#ifndef SRC_MODEL_OPTIMIZER_ADAGRAD_H_
#define SRC_MODEL_OPTIMIZER_ADAGRAD_H_
#include "singa/model/optimizer.h"
#include <functional>
namespace singa {

void AdaGrad::Setup(const OptimizerConf& conf) { delta_ = conf.delta(); }

// history += grad*grad;
// value = value - lr*grad/sqrt(history+delta)
void AdaGrad::Apply(int step, float lr, const string& name, const Tensor& grad,
                    Tensor* value) {
  if (history_gradient_.find(name) == history_gradient_.end())
    history_gradient_[name].ResetLike(*value);
  Tensor& history = history_gradient_[name];
  Tensor tmp = Square(grad);
  history += tmp;
  Add(history, delta_, &tmp);
  Sqrt(tmp, &tmp);
  Div(grad, tmp, &tmp);
  Axpy(-lr, tmp, value);
}
}  // namespace singa
#endif  // SRC_MODEL_OPTIMIZER_ADAGRAD_H_
