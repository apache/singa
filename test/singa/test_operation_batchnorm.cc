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

#include <iostream>

#include "../src/model/operation/batchnorm.h"
#include "gtest/gtest.h"

using namespace singa;

#ifdef USE_DNNL
TEST(DNNLOperationBatchNorm, ForwardInference) {
  Tensor x(Shape{2, 2});
  Tensor alpha(Shape{2});
  Tensor beta(Shape{2});
  Tensor moving_mean(Shape{2});
  Tensor moving_var(Shape{2});

  Gaussian(0.0f, 1.0f, &x);
  Gaussian(0.0f, 1.0f, &alpha);
  Gaussian(0.0f, 1.0f, &beta);
  Gaussian(0.0f, 1.0f, &moving_mean);
  Gaussian(0.0f, 1.0f, &moving_var);

  BatchNormHandle batch_norm_handle(0u, x);
  Tensor y = CpuBatchNormForwardInference(batch_norm_handle, x, alpha, beta,
                                          moving_mean, moving_var);
}

TEST(DNNLOperationBatchNorm, ForwardTraining) {
  Tensor x(Shape{2, 2});
  Tensor alpha(Shape{2});
  Tensor beta(Shape{2});
  Tensor moving_mean(Shape{2});
  Tensor moving_var(Shape{2});

  Gaussian(0.0f, 1.0f, &x);
  Gaussian(0.0f, 1.0f, &alpha);
  Gaussian(0.0f, 1.0f, &beta);
  Gaussian(0.0f, 1.0f, &moving_mean);
  Gaussian(0.0f, 1.0f, &moving_var);

  BatchNormHandle batch_norm_handle(0u, x);
  auto outputs = CpuBatchNormForwardTraining(batch_norm_handle, x, alpha, beta,
                                             moving_mean, moving_var);
}

TEST(DNNLOperationBatchNorm, Backward) {
  Tensor x(Shape{2, 2});
  Tensor y(Shape{2, 2});
  Tensor dy(Shape{2, 2});
  Tensor alpha(Shape{2});
  Tensor beta(Shape{2});
  Tensor moving_mean(Shape{2});
  Tensor moving_var(Shape{2});

  Gaussian(0.0f, 1.0f, &x);
  Gaussian(0.0f, 1.0f, &y);
  Gaussian(0.0f, 1.0f, &dy);
  Gaussian(0.0f, 1.0f, &alpha);
  Gaussian(0.0f, 1.0f, &beta);
  Gaussian(0.0f, 1.0f, &moving_mean);
  Gaussian(0.0f, 1.0f, &moving_var);

  BatchNormHandle batch_norm_handle(0u, x);
  auto outputs = CpuBatchNormBackwardx(batch_norm_handle, y, dy, x, alpha, beta,
                                       moving_mean, moving_var);
}

#endif  // USE_DNNL
