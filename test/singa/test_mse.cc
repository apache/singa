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
#include "singa_config.h"

using singa::Tensor;
class TestMSE : public ::testing::Test {
 protected:
  virtual void SetUp() {
    p.Reshape(singa::Shape{2, 3});
    t.Reshape(singa::Shape{2, 3});
    p.CopyDataFromHostPtr(pdat, sizeof(pdat) / sizeof(float));
    t.CopyDataFromHostPtr(tdat, sizeof(pdat) / sizeof(float));
  }
  const float pdat[6] = {0.1, 1.1, 2.1, 0.3, 2.2, 1.8};
  const float tdat[6] = {0.1, 1.1, 2.0, 0.3, 2.2, 1.8};

  singa::Tensor p, t;
};

#ifdef USE_CBLAS
TEST_F(TestMSE, CppForward) {
  singa::MSE mse;
  const Tensor& loss = mse.Forward(p, t);
  auto ldat = loss.data<const float*>();

  for (size_t i = 0, k = 0; i < loss.Size(); i++) {
    float l = 0.f;
    for (size_t j = 0; j < p.Size() / loss.Size(); j++) {
      l += (pdat[k] - tdat[k]) * (pdat[k] - tdat[k]);
      k++;
    }
    EXPECT_FLOAT_EQ(ldat[i], 0.5 * l);
  }
}

TEST_F(TestMSE, CppBackward) {
  singa::MSE mse;
  mse.Forward(p, t);
  const Tensor& grad = mse.Backward();

  auto gdat = grad.data<const float*>();

  for (size_t i = 0; i < grad.Size(); i++)
    EXPECT_FLOAT_EQ(gdat[i], (1.0f / p.shape().at(0)) * (pdat[i] - tdat[i]));
}
#endif
#ifdef USE_CUDA
TEST_F(TestMSE, CudaForward) {
  singa::MSE mse;
  singa::CudaGPU dev;
  p.ToDevice(&dev);
  t.ToDevice(&dev);
  Tensor loss = mse.Forward(p, t);

  loss.ToHost();
  auto ldat = loss.data<const float*>();

  for (size_t i = 0, k = 0; i < loss.Size(); i++) {
    float l = 0.f;
    for (size_t j = 0; j < p.Size() / loss.Size(); j++) {
      l += (pdat[k] - tdat[k]) * (pdat[k] - tdat[k]);
      k++;
    }
    EXPECT_FLOAT_EQ(ldat[i], 0.5 * l);
  }
}
TEST_F(TestMSE, CudaBackward) {
  singa::MSE mse;
  singa::CudaGPU dev;
  p.ToDevice(&dev);
  t.ToDevice(&dev);
  mse.Forward(p, t);
  Tensor grad = mse.Backward();
  grad.ToHost();
  auto gdat = grad.data<const float*>();

  for (size_t i = 0; i < grad.Size(); i++)
    EXPECT_FLOAT_EQ(gdat[i], (1.0f / p.shape().at(0)) * (pdat[i] - tdat[i]));
}
#endif
