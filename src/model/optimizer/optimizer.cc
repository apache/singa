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

#include "singa/model/optimizer.h"
#include "singa/utils/logging.h"

namespace singa {

void Optimizer::Register(const string& name, const ParamSpec& specs) {
  if (specs.has_constraint()) {
    CHECK(constraints_.find(name) == constraints_.end())
        << "Parameter with name = " << name << " has already registered";
    constraints_[name] = new Constraint(specs.constraint());
  }
  if (specs.has_regularizer()) {
    CHECK(regularizers_.find(name) == regularizers_.end())
        << "Parameter with name = " << name << " has already registered";
    regularizers_[name] = new Regularizer(specs.regularizer());
  }
  if (specs.has_lr_mult()) {
    CHECK(learning_rate_multplier_.find(name) == learning_rate_multplier_.end())
        << "Parameter with name = " << name << " has already registered";
    learning_rate_multplier_[name] = specs.lr_mult();
  }
  /*
  if (specs.has_lr_generator()) {
    LOG(FATAL) << "Not implemented yet";
  }
  */
}

void Optimizer::Apply(int step, const string& name, Tensor* grad,
                      Tensor* param) {
  // TODO(wangwei) need to consider the order of constraint and regularizer
  if (regularizers_.find(name) != regularizers_.end())
    regularizers_.at(name)->Apply(step, param, grad);
  if (constraints_.find(name) != constraints_.end())
    constraints_.at(name)->Apply(step, param, grad);
  float lr = learning_rate_generator_(step);
  if (learning_rate_multplier_.find(name) != learning_rate_multplier_.end())
    lr *= learning_rate_multplier_.at(name);
  Apply(step, lr, name, grad, param);
}

void Regularizer::Setup(const RegularizerConf& conf) {
  type_ = conf.type();
  coefficient_ = conf.coefficient();
}

void Regularizer::Apply(int step, Tensor* value, Tensor* grad) {
  if (type_ == "L2" || type_ == "l2") {
    (*grad) -= (*value) * coefficient_;
  } else {
    CHECK(type_ == "NotSet") << "Unknown regularizer type = " << type_;
  }
}

void Regularizer::Apply(int step, const vector<Tensor*>& values,
                        const vector<Tensor*>& grads) {
  LOG(FATAL) << "Not implemented yet";
}

void Constraint::Setup(const ConstraintConf& conf) {
  type_ = conf.type();
  threshold_ = conf.threshold();
}

void Constraint::Apply(int step, Tensor* value, Tensor* grad) {
  // TODO(wangwei) implement L2 and hard constraint
  CHECK(type_ == "NotSet") << "Unknown regularizer type = " << type_;
}

void Constraint::Apply(int step, const vector<Tensor*>& values,
                       const vector<Tensor*>& grads) {
  LOG(FATAL) << "Not implemented yet";
}

}  // namespace singa
