/*********************************************************
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
************************************************************/

#include "../src/model/operation/batchnorm.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace singa;

#ifdef USE_DNNL
TEST(DNNLOperationBatchNorm, ForwardInference) {
  const float x_data[] = {1, 2,
                          3, 4};
  Tensor in(Shape{2, 2});
  in.CopyDataFromHostPtr(x_data, 2 * 2);

  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {2, 2};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);

  const float mean_[] = {2, 3};
  Tensor moving_mean(Shape{2});
  moving_mean.CopyDataFromHostPtr(mean_, 2);

  const float var_[] = {1, 1};
  Tensor moving_var(Shape{2});
  moving_var.CopyDataFromHostPtr(var_, 2);

  BatchNormHandle batch_norm_handle(0u,in);
  Tensor y = CpuBatchNormForwardInference(batch_norm_handle, in, alpha, beta, moving_mean, moving_var);
}

TEST(DNNLOperationBatchNorm, ForwardTraining) {
  const float x_data[] = {1, 2,
                          3, 4};
  Tensor in(Shape{2, 2});
  in.CopyDataFromHostPtr(x_data, 2 * 2);

  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {2, 2};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);

  const float mean_[] = {2, 3};
  Tensor moving_mean(Shape{2});
  moving_mean.CopyDataFromHostPtr(mean_, 2);

  const float var_[] = {1, 1};
  Tensor moving_var(Shape{2});
  moving_var.CopyDataFromHostPtr(var_, 2);

  BatchNormHandle batch_norm_handle(0u,in);
  auto outputs = CpuBatchNormForwardTraining(batch_norm_handle, in, alpha, beta, moving_mean, moving_var);
}


#endif // USE_DNNL

