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

#include "../src/model/layer/batchnorm.h"
#include "gtest/gtest.h"

using namespace singa;

TEST(BatchNorm, Setup) {
  BatchNorm batchnorm;
  // EXPECT_EQ("BatchNorm", batchnorm.layer_type());

  singa::LayerConf conf;
  singa::BatchNormConf *batchnorm_conf = conf.mutable_batchnorm_conf();
  batchnorm_conf->set_factor(0.01);
  batchnorm.Setup(Shape{2, 4, 4}, conf);

  EXPECT_FLOAT_EQ(0.01f, batchnorm.factor());
  EXPECT_EQ(2u, batchnorm.channels());
  EXPECT_EQ(4u, batchnorm.height());
  EXPECT_EQ(4u, batchnorm.width());
}

TEST(BatchNorm, Forward) {
  BatchNorm batchnorm;
  const float x[] = {1, 2, 3, 4};
  Tensor in(Shape{2, 2});
  in.CopyDataFromHostPtr(x, 2 * 2);
  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {2, 2};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);
  singa::LayerConf conf;
  singa::BatchNormConf *batchnorm_conf = conf.mutable_batchnorm_conf();
  batchnorm_conf->set_factor(1);
  batchnorm.Setup(Shape{2}, conf);
  batchnorm.set_bnScale(alpha);
  batchnorm.set_bnBias(beta);
  batchnorm.set_runningMean(beta);
  batchnorm.set_runningVariance(beta);
  Tensor out = batchnorm.Forward(kTrain, in);
  const float *outptr = out.data<float>();
  const auto &shape = out.shape();
  EXPECT_EQ(2u, shape.size());
  EXPECT_EQ(2u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  EXPECT_NEAR(1.0f, outptr[0], 1e-4f);
  EXPECT_NEAR(1.0f, outptr[1], 1e-4f);
  EXPECT_NEAR(3.0f, outptr[2], 1e-4f);
  EXPECT_NEAR(3.0f, outptr[3], 1e-4f);
}

TEST(BatchNorm, Backward) {
  BatchNorm batchnorm;
  const float x[] = {1, 2, 3, 4};
  Tensor in(Shape{2, 2});
  in.CopyDataFromHostPtr(x, 2 * 2);
  const float dy[] = {4, 3, 2, 1};
  Tensor dy_in(Shape{2, 2});
  dy_in.CopyDataFromHostPtr(dy, 2 * 2);
  const float alpha_[] = {1, 1};
  Tensor alpha(Shape{2});
  alpha.CopyDataFromHostPtr(alpha_, 2);

  const float beta_[] = {0, 0};
  Tensor beta(Shape{2});
  beta.CopyDataFromHostPtr(beta_, 2);
  singa::LayerConf conf;
  singa::BatchNormConf *batchnorm_conf = conf.mutable_batchnorm_conf();
  batchnorm_conf->set_factor(1);
  batchnorm.Setup(Shape{2}, conf);
  batchnorm.set_bnScale(alpha);
  batchnorm.set_bnBias(beta);
  batchnorm.set_runningMean(beta);
  batchnorm.set_runningVariance(beta);
  Tensor out = batchnorm.Forward(kTrain, in);
  auto ret = batchnorm.Backward(kTrain, dy_in);
  Tensor dx = ret.first;
  const auto &shape = dx.shape();
  EXPECT_EQ(2u, shape.size());
  EXPECT_EQ(2u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  const float *dxptr = ret.first.data<float>();
  EXPECT_NEAR(.0f, dxptr[0], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[1], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[2], 1e-4f);
  EXPECT_NEAR(.0f, dxptr[3], 1e-4f);

  Tensor dbnScale = ret.second.at(0);
  const float *dbnScaleptr = dbnScale.data<float>();
  const auto &dbnScaleShape = dbnScale.shape();
  EXPECT_EQ(1u, dbnScaleShape.size());
  EXPECT_EQ(2u, dbnScaleShape[0]);

  EXPECT_NEAR(-2.0f, dbnScaleptr[0], 1e-4f);
  EXPECT_NEAR(-2.0f, dbnScaleptr[1], 1e-4f);

  Tensor dbnBias = ret.second.at(1);
  const float *dbnBiasptr = dbnBias.data<float>();
  const auto &dbnBiasShape = dbnBias.shape();
  EXPECT_EQ(1u, dbnBiasShape.size());
  EXPECT_EQ(2u, dbnBiasShape[0]);

  EXPECT_NEAR(6.0f, dbnBiasptr[0], 1e-4f);
  EXPECT_NEAR(4.0f, dbnBiasptr[1], 1e-4f);
}
