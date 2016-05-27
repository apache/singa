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
#include "../src/model/loss/cross_entropy.h"

using singa::Tensor;
class TestCrossEntropy : public ::testing::Test {
 protected:
  virtual void SetUp() {
    p.Reshape(singa::Shape{2, 4});
    t.Reshape(singa::Shape{2, 1});
    p.CopyDataFromHostPtr(pdat, sizeof(pdat) / sizeof(float));
    t.CopyDataFromHostPtr(tdat, sizeof(pdat) / sizeof(float));
  }
  const float pdat[8] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  const float tdat[2] = {0.0, 2.0};

  singa::Tensor p, t;
};

TEST_F(TestCrossEntropy, CppForward) {
  singa::CrossEntropy cross_entropy;
  const Tensor& loss = cross_entropy.Forward(p, t);
  auto ldat = loss.data<const float*>();

  const float result_test = -log(0.25);
  EXPECT_FLOAT_EQ(ldat[0], result_test);
  EXPECT_FLOAT_EQ(ldat[1], result_test);
}

TEST_F(TestCrossEntropy, CppBackward) {
  singa::CrossEntropy cross_entropy;
  cross_entropy.Forward(p, t);
  const Tensor& grad = cross_entropy.Backward();

  auto gdat = grad.data<const float*>();
  EXPECT_FLOAT_EQ(gdat[0], -0.75);
  EXPECT_FLOAT_EQ(gdat[1], 0.25);
  EXPECT_FLOAT_EQ(gdat[2], 0.25);
  EXPECT_FLOAT_EQ(gdat[3], 0.25);
  EXPECT_FLOAT_EQ(gdat[4], 0.25);
  EXPECT_FLOAT_EQ(gdat[5], 0.25);
  EXPECT_FLOAT_EQ(gdat[6], -0.75);
  EXPECT_FLOAT_EQ(gdat[7], 0.25);
}
