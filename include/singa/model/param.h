/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#ifndef SINGA_MODEL_PARAM_H_
#define SINGA_MODEL_PARAM_H_
#include "singa/core/tensor.h"
#include <vector>
#include <string>
using std::vector;
using std::string;
namespace singa {
/// Base Param class for storing set of parameters, e.g., a weight matrix or a
/// bias vector.
/// It includes multiple Tensor s for parameter values, gradients, etc.
class Param {
 public:
  ~Param();
  Param(const ParamSpec& conf);
  Param(Param&& p);
  Param(const Param& p);
  void operator=(Param&& p);
  void operator=(const Param& p);

  Tensor& value() {
    return value_;
  }

  Tensor& grad() {
    return grad_;
  }

  void set_value(const Tensor& t) {
    value_ = t;
  }

  void set_value(Tensor&& t) {
    value_ = std::move(t);
  }

  void set_grad(const Tensor& t) {
    isGradValid_ = true;
    grad_ = t;
  }

  void set_grad(Tensor&& t) {
    grad_ = std::move(t);
  }

  // void Compress();
  // string ToString();

 protected:
  string name_;
  Tensor value_;
  float lr_mult_ = 1.0f, decay_mult_ = 1.0f;
};

class ParamGrad {
// return grad tensor or data to recover the grad tensor, e.g., if W = U * V
// then, ParamGrad could just store U and V. provide func for serailize and
// deserialize.
};

// updater just copy the ParamGrad to a device and submit ops to that device, e.g.,
// add grad; check update_condidtion; apply sgd; copy back.
// consider rpc (no rmda).

Param* CreateParam(string type) {
  Param* p = nullptr;
  if (type == "default")
    p = new Param();
  else
    LOG(FATAL) << "Currently param type " << type << " is not implemented."
               << "Pls use the 'default' type";
  return p;
}
#endif  // SINGA_MODEL_PARAM_H_

}  // namespace singa
