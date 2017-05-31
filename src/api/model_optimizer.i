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

/*interface file for swig */

%module model_optimizer
%include "std_vector.i"
%include "std_string.i"
%include "std_pair.i"
%include "std_shared_ptr.i"

%{
#define SWIG_PYTHON_STRICT_BYTE_CHAR
#include "singa/model/optimizer.h"
#include "singa/proto/model.pb.h"
using singa::Tensor;
using singa::ParamSpec;
using singa::OptimizerConf;
%}


%shared_ptr(singa::Optimizer)
%shared_ptr(singa::Regularizer)
%shared_ptr(singa::Constraint)

namespace singa {
class Optimizer {
 public:
  // Optimizer() = default;
  virtual ~Optimizer() = default;
  void Setup(const std::string& str);
  virtual void Apply(int epoch, float lr, const std::string& name,
      Tensor& grad, Tensor& value, int step = -1) = 0;
};
inline std::shared_ptr<Optimizer> CreateOptimizer(const std::string& type);

class Constraint {
 public:
  Constraint() = default;
  void Setup(const std::string& conf_str);
  void Apply(int epoch, const Tensor& value, Tensor& grad, int step = -1);
};

inline std::shared_ptr<Constraint> CreateConstraint(const std::string& type);

class Regularizer {
 public:
  Regularizer() = default;
  void Setup(const std::string& conf_str);
  void Apply(int epoch, const Tensor& value, Tensor& grad, int step = -1);
};
inline std::shared_ptr<Regularizer> CreateRegularizer(const std::string& type);
}
