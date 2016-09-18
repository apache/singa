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

#include "gtest/gtest.h"
#include "singa/core/tensor.h"
#include "singa/core/device.h"
#include "singa/model/loss.h"
#include "singa/singa_config.h"

using singa::Tensor;
class TestSoftmaxCrossEntropy : public ::testing::Test {
 protected:
  virtual void SetUp() {
    p.Reshape(singa::Shape{2, 4});
    t.Reshape(singa::Shape{2, 1});
  }
  const float pdat[8] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f };
  const int tdat[2] = {0, 2};

  singa::Tensor p, t;
};

TEST_F(TestSoftmaxCrossEntropy, CppForward) {
  p.CopyDataFromHostPtr(pdat, 8);
  t.AsType(singa::kInt);
  t.CopyDataFromHostPtr(tdat, 2);

  singa::SoftmaxCrossEntropy cross_entropy;
  const Tensor& loss = cross_entropy.Forward(singa::kEval, p, t);
  auto ldat = loss.data<float>();

  const float result_test = -log(0.25);
  EXPECT_FLOAT_EQ(ldat[0], result_test);
  EXPECT_FLOAT_EQ(ldat[1], result_test);
}

TEST_F(TestSoftmaxCrossEntropy, CppBackward) {
  p.CopyDataFromHostPtr(pdat, 8);
  t.AsType(singa::kInt);
  t.CopyDataFromHostPtr(tdat, 2);

  singa::SoftmaxCrossEntropy cross_entropy;
  cross_entropy.Forward(singa::kTrain, p, t);
  const Tensor& grad = cross_entropy.Backward();

  auto gdat = grad.data<float>();
  EXPECT_FLOAT_EQ(gdat[0], -0.75);
  EXPECT_FLOAT_EQ(gdat[1], 0.25);
  EXPECT_FLOAT_EQ(gdat[2], 0.25);
  EXPECT_FLOAT_EQ(gdat[3], 0.25);
  EXPECT_FLOAT_EQ(gdat[4], 0.25);
  EXPECT_FLOAT_EQ(gdat[5], 0.25);
  EXPECT_FLOAT_EQ(gdat[6], -0.75);
  EXPECT_FLOAT_EQ(gdat[7], 0.25);
}

#ifdef USE_CUDA

TEST_F(TestSoftmaxCrossEntropy, CudaForward) {
  singa::SoftmaxCrossEntropy cross_entropy;
  auto dev = std::make_shared<singa::CudaGPU>();
  p.ToDevice(dev);
  t.ToDevice(dev);
  p.CopyDataFromHostPtr(pdat, 8);
  t.CopyDataFromHostPtr(tdat, 2);

  Tensor loss = cross_entropy.Forward(singa::kEval, p, t);
  loss.ToHost();
  auto ldat = loss.data<float>();

  const float result_test = -log(0.25);
  EXPECT_FLOAT_EQ(ldat[0], result_test);
  EXPECT_FLOAT_EQ(ldat[1], result_test);
}

TEST_F(TestSoftmaxCrossEntropy, CudaBackward) {
  singa::SoftmaxCrossEntropy cross_entropy;
  auto dev = std::make_shared<singa::CudaGPU>();
  p.ToDevice(dev);
  t.ToDevice(dev);
  p.CopyDataFromHostPtr(pdat, 8);
  t.CopyDataFromHostPtr(tdat, 2);

  cross_entropy.Forward(singa::kTrain, p, t);
  Tensor grad = cross_entropy.Backward();

  grad.ToHost();
  auto gdat = grad.data<float>();
  EXPECT_FLOAT_EQ(gdat[0], -0.75);
  EXPECT_FLOAT_EQ(gdat[1], 0.25);
  EXPECT_FLOAT_EQ(gdat[2], 0.25);
  EXPECT_FLOAT_EQ(gdat[3], 0.25);
  EXPECT_FLOAT_EQ(gdat[4], 0.25);
  EXPECT_FLOAT_EQ(gdat[5], 0.25);
  EXPECT_FLOAT_EQ(gdat[6], -0.75);
  EXPECT_FLOAT_EQ(gdat[7], 0.25);
}
#endif  // USE_CUDA
