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
#include "singa/core/device.h"
#include "singa/core/tensor.h"
#include "singa/model/loss.h"

using singa::Tensor;
class TestMSE : public ::testing::Test {
 protected:
  virtual void SetUp() {
    p.Resize(singa::Shape{2, 3});
    t.Resize(singa::Shape{2, 3});
    p.CopyDataFromHostPtr(pdat, sizeof(pdat) / sizeof(float));
    t.CopyDataFromHostPtr(tdat, sizeof(pdat) / sizeof(float));
  }
  const float pdat[6] = {0.1f, 1.1f, 2.1f, 0.3f, 2.2f, 1.8f};
  const float tdat[6] = {0.1f, 1.1f, 2.0f, 0.3f, 2.2f, 1.8f};

  singa::Tensor p, t;
};

#ifdef USE_CBLAS
TEST_F(TestMSE, CppForward) {
  singa::MSE mse;
  const Tensor& loss = mse.Forward(singa::kEval, p, t);
  auto ldat = loss.data<float>();

  for (size_t i = 0, k = 0; i < loss.Size(); i++) {
    float l = 0.f;
    for (size_t j = 0; j < p.Size() / loss.Size(); j++) {
      l += (pdat[k] - tdat[k]) * (pdat[k] - tdat[k]);
      k++;
    }
    EXPECT_FLOAT_EQ(ldat[i], 0.5f * l);
  }
}

TEST_F(TestMSE, CppBackward) {
  singa::MSE mse;
  mse.Forward(singa::kTrain, p, t);
  const Tensor& grad = mse.Backward();

  auto gdat = grad.data<float>();

  for (size_t i = 0; i < grad.Size(); i++)
    EXPECT_FLOAT_EQ(gdat[i], (1.0f / p.shape().at(0)) * (pdat[i] - tdat[i]));
}
#endif
#ifdef USE_CUDA
TEST_F(TestMSE, CudaForward) {
  singa::MSE* mse = new singa::MSE();
  auto dev = std::make_shared<singa::CudaGPU>();
  p.ToDevice(dev);
  t.ToDevice(dev);
  Tensor loss = mse->Forward(singa::kEval, p, t);

  loss.ToHost();
  auto ldat = loss.data<float>();

  for (size_t i = 0, k = 0; i < loss.Size(); i++) {
    float l = 0.f;
    for (size_t j = 0; j < p.Size() / loss.Size(); j++) {
      l += (pdat[k] - tdat[k]) * (pdat[k] - tdat[k]);
      k++;
    }
    EXPECT_FLOAT_EQ(ldat[i], 0.5 * l);
  }
  p.ToHost();
  t.ToHost();
  delete mse;
}

TEST_F(TestMSE, CudaBackward) {
  singa::MSE mse;
  auto dev = std::make_shared<singa::CudaGPU>();
  p.ToDevice(dev);
  t.ToDevice(dev);
  mse.Forward(singa::kTrain, p, t);
  Tensor grad = mse.Backward();
  grad.ToHost();
  auto gdat = grad.data<float>();

  for (size_t i = 0; i < grad.Size(); i++)
    EXPECT_FLOAT_EQ(gdat[i], (1.0f / p.shape().at(0)) * (pdat[i] - tdat[i]));
  p.ToHost();
  t.ToHost();
}
#endif
