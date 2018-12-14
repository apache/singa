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

#ifdef USE_MKLDNN

TEST(OperationBatchNorm, ForwardInference) {
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

  Tensor moving_mean(Shape{});
  Tensor moving_var(Shape{});


  // momentum
  BatchNormHandle batch_norm_handle(0u,in);
  Tensor y = CpuBatchNormForwardInference(batch_norm_handle, in, alpha, beta, moving_mean,moving_var);


  const float *outptr = y.data<float>();
  const auto &shape = y.shape();
  EXPECT_EQ(2u, shape.size());
  EXPECT_EQ(2u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  EXPECT_NEAR(1.0f, outptr[0], 1e-4f);
  EXPECT_NEAR(1.0f, outptr[1], 1e-4f);
  EXPECT_NEAR(3.0f, outptr[2], 1e-4f);
  EXPECT_NEAR(3.0f, outptr[3], 1e-4f);
}

TEST(OperationBatchNorm, ForwardInference4D) {
  float x_data[] = {
      0.0736655, 0.0459045, 0.0779517, 0.0771059,
      0.0586862, 0.0561263, 0.0708457, 0.0977273,
      0.0405025, -0.170897, 0.0208982, 0.136865,
      -0.0367905, -0.0618205, -0.0103908, -0.0522777,
      -0.122161, -0.025427, -0.0718576, -0.185941,
      0.0166533, 0.178679, -0.0576606, -0.137817,
      0.150676, 0.153442, -0.0929899, -0.148675,
      -0.112459, -0.106284, -0.103074, -0.0668811
  };
  Tensor in(Shape{1, 2, 4, 4});
  in.CopyDataFromHostPtr(x_data, 2*4*4);

  const float alpha_[] = {1,1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {0,0};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);

  Tensor moving_mean(Shape{});
  Tensor moving_var(Shape{});

  // momentum
  BatchNormHandle batch_norm_handle(0.0f,in);
  Tensor y = CpuBatchNormForwardInference(batch_norm_handle, in, alpha, beta, moving_mean, moving_var);


  // y = {1,1,1,1, 3,3,3,3}
  const float *outptr = y.data<float>();
  const auto &shape = y.shape();
  EXPECT_EQ(4u, shape.size());
  EXPECT_EQ(1u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  EXPECT_EQ(4u, shape[2]);
  EXPECT_EQ(4u, shape[3]);
  EXPECT_NEAR(0.637092, outptr[0],  1e-4f);
  EXPECT_NEAR(0.262057, outptr[1],  1e-4f);
  EXPECT_NEAR(0.694995, outptr[2],  1e-4f);
  EXPECT_NEAR(0.683569, outptr[3],  1e-4f);
  EXPECT_NEAR(0.434730, outptr[4],  1e-4f);
  EXPECT_NEAR(0.400147, outptr[5],  1e-4f);
  EXPECT_NEAR(0.598998, outptr[6],  1e-4f);
  EXPECT_NEAR(0.962152, outptr[7],  1e-4f);
  EXPECT_NEAR(0.189079, outptr[8],  1e-4f);
  EXPECT_NEAR(-2.66680, outptr[9],  1e-4f);
  EXPECT_NEAR(-0.07576, outptr[10], 1e-4f);
  EXPECT_NEAR(1.490880, outptr[11], 1e-4f);
  EXPECT_NEAR(-0.85510, outptr[12], 1e-4f);
  EXPECT_NEAR(-1.19324, outptr[13], 1e-4f);
  EXPECT_NEAR(-0.49845, outptr[14], 1e-4f);
  EXPECT_NEAR(-1.06433, outptr[15], 1e-4f);
  EXPECT_NEAR(-0.69664, outptr[16], 1e-4f);
  EXPECT_NEAR(0.185125, outptr[17], 1e-4f);
  EXPECT_NEAR(-0.23810, outptr[18], 1e-4f);
  EXPECT_NEAR(-1.27803, outptr[19], 1e-4f);
  EXPECT_NEAR(0.568704, outptr[20], 1e-4f);
  EXPECT_NEAR(2.045640, outptr[21], 1e-4f);
  EXPECT_NEAR(-0.10869, outptr[22], 1e-4f);
  EXPECT_NEAR(-0.83935, outptr[23], 1e-4f);
  EXPECT_NEAR(1.790380, outptr[24], 1e-4f);
  EXPECT_NEAR(1.815590, outptr[25], 1e-4f);
  EXPECT_NEAR(-0.43073, outptr[26], 1e-4f);
  EXPECT_NEAR(-0.93833, outptr[27], 1e-4f);
  EXPECT_NEAR(-0.60820, outptr[28], 1e-4f);
  EXPECT_NEAR(-0.55192, outptr[29], 1e-4f);
  EXPECT_NEAR(-0.52265, outptr[30], 1e-4f);
  EXPECT_NEAR(-0.19274, outptr[31], 1e-4f);
}

TEST(OperationBatchNorm, ForwardTraining) {
  const float x_data[] = {1, 2, 3, 4};
  Tensor x(Shape{2, 2});
  x.CopyDataFromHostPtr(x_data, 2 * 2);

  const float y_data[] = {9, 9, 9, 9};
  Tensor y(Shape{2, 2});
  y.CopyDataFromHostPtr(y_data, 2 * 2);

  const float dy_data[] = {4, 3, 2, 1};
  Tensor dy(Shape{2, 2});
  dy.CopyDataFromHostPtr(dy_data, 2 * 2);

  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {0, 0};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);


  // 0 momentum will ignore running mean and var
  BatchNormHandle batch_norm_handle(0.3f,x);
  const float running_mean_[] = {0,0};
  Tensor running_mean(Shape{2});
  Tensor running_var(Shape{2});
  running_mean.CopyDataFromHostPtr(running_mean_, 2);
  running_var.CopyDataFromHostPtr(running_mean_, 2);


  // training operation calculate the running mean and var for backward
  auto ret1 = CpuBatchNormForwardTraining(batch_norm_handle, x, alpha, beta, running_mean, running_var);
  const float *meanptr = ret1[1].data<float>();
  EXPECT_NEAR(1.4f, meanptr[0], 1e-4f);
  EXPECT_NEAR(2.1f, meanptr[1], 1e-4f);
  const float *varptr = ret1[2].data<float>();
  EXPECT_NEAR(0.7f, varptr[0], 1e-4f);
  EXPECT_NEAR(0.7f, varptr[1], 1e-4f);
}

TEST(OperationBatchNorm, Backward) {
  const float x_data[] = {1, 2, 3, 4};
  Tensor x(Shape{2, 2});
  x.CopyDataFromHostPtr(x_data, 2 * 2);

  const float y_data[] = {9, 9, 9, 9};
  Tensor y(Shape{2, 2});
  y.CopyDataFromHostPtr(y_data, 2 * 2);

  const float dy_data[] = {4, 3, 2, 1};
  Tensor dy(Shape{2, 2});
  dy.CopyDataFromHostPtr(dy_data, 2 * 2);

  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {0, 0};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);


  // 0 momentum will ignore running mean and var
  BatchNormHandle batch_norm_handle(0.0f,x);
  const float running_mean_[] = {1,2};
  Tensor running_mean(Shape{2});
  Tensor running_var(Shape{2});
  running_mean.CopyDataFromHostPtr(running_mean_, 2);
  running_var.CopyDataFromHostPtr(running_mean_, 2);


  // training operation calculate the running mean and var for backward
  auto ret1 = CpuBatchNormForwardTraining(batch_norm_handle, x, alpha, beta, running_mean, running_var);

  // calculate dx, dscale, dbias
  auto ret2 = CpuBatchNormBackwardx( batch_norm_handle, y, dy, x, alpha, beta, ret1[1],  ret1[2]);

  const auto &shape = ret2[0].shape();
  EXPECT_EQ(2u, shape.size());
  EXPECT_EQ(2u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  const float *dxptr = ret2[0].data<float>();
  EXPECT_NEAR(.0f, dxptr[0], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[1], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[2], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[3], 1e-4f);


  const auto &dbnScaleShape = ret2[1].shape();
  EXPECT_EQ(2u, dbnScaleShape[1]);
  const auto &dbnBiasShape = ret2[2].shape();
  EXPECT_EQ(2u, dbnBiasShape[1]);
  const float *dbnScaleptr = ret2[1].data<float>();
  EXPECT_NEAR(-2.0f, dbnScaleptr[0], 1e-4f);
  EXPECT_NEAR(-2.0f, dbnScaleptr[1], 1e-4f);
  const float *dbnBiasptr = ret2[2].data<float>();
  EXPECT_NEAR(6.0f, dbnBiasptr[0], 1e-4f);
  EXPECT_NEAR(4.0f, dbnBiasptr[1], 1e-4f);
}

#endif // USE_MKLDNN
