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

%module model_loss
%include "std_string.i"
%{
#include "singa/model/loss.h"
  using singa::Tensor;
%}

namespace singa {
class Loss {
public:
  Loss() = default;
  virtual ~Loss() {}

  virtual Tensor Forward(int flag, const Tensor &prediction,
                         const Tensor &target) = 0;

  float Evaluate(int flag, const Tensor &prediction, const Tensor &target);

  /// Compute the gradients of the loss values w.r.t. the prediction.
  virtual Tensor Backward() = 0;
};

class MSE : public Loss {
public:
  Tensor Forward(int flag, const Tensor &prediction, const Tensor &target)
      override;

  Tensor Backward() override;
};

class SoftmaxCrossEntropy : public Loss {
public:
  Tensor Forward(int flag, const Tensor &prediction, const Tensor &target)
      override;

  Tensor Backward() override;
};

}
