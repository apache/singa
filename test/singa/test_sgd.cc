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
#include "singa/model/optimizer.h"
#include "singa/singa_config.h"

TEST(SGD, ApplyWithoutMomentum) {
  singa::SGD sgd;
  const float v[4] = {0.1f, 0.2f, 0.3f, 0.4f};
  const float g[4] = {0.1f, 0.1f, 0.1f, 0.1f};

  singa::Tensor value(singa::Shape{4}), grad(singa::Shape{4});
  value.CopyDataFromHostPtr(v, 4);
  grad.CopyDataFromHostPtr(g, 4);

  float lr = 0.1f;
  sgd.Apply(0, lr, "xx", grad, value);

  singa::Tensor v1 = value.Clone();
  const float* newv1 = v1.data<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(newv1[i], v[i] - g[i] * lr);
  }

  lr /= 2;
  grad.CopyDataFromHostPtr(g, 4);
  sgd.Apply(1, lr, "xx", grad, value);
  singa::Tensor v2 = value.Clone();
  const float* newv2 = v2.data<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(newv2[i], newv1[i] - g[i] * lr);
  }
}

TEST(SGD, ApplyWithMomentum) {
  singa::SGD sgd;
  float lr = 0.1f;
  auto func = [](int step) { return step <= 5 ? 0.5f : 0.9f; };
  sgd.SetMomentumGenerator(func);
  const float v[4] = {0.1f, 0.2f, 0.3f, 0.4f};
  const float g[4] = {0.01f, 0.02f, 0.03f, 0.04f};

  singa::Tensor value(singa::Shape{4}), grad(singa::Shape{4});
  value.CopyDataFromHostPtr(v, 4);
  grad.CopyDataFromHostPtr(g, 4);

  sgd.Apply(0, lr, "xx", grad, value);

  singa::Tensor v1 = value.Clone();
  const float* newv1 = v1.data<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(newv1[i], v[i] - g[i] * lr);
  }

  grad.CopyDataFromHostPtr(g, 4);
  sgd.Apply(1, lr, "xx", grad, value);
  singa::Tensor v2 = value.Clone();
  const float* newv2 = v2.data<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(newv2[i], newv1[i] - (g[i] * lr + g[i] * lr * func(1)));
  }
}

#ifdef USE_CUDA
TEST(SGD, ApplyWithoutMomentumCuda) {
  singa::SGD sgd;
  const float v[4] = {0.1, 0.2, 0.3, 0.4};
  const float g[4] = {0.1, 0.1, 0.1, 0.1};

  auto dev = std::make_shared<singa::CudaGPU>();
  singa::Tensor value(singa::Shape{4}, dev), grad(singa::Shape{4}, dev);
  value.CopyDataFromHostPtr(v, 4);
  grad.CopyDataFromHostPtr(g, 4);

  float lr = 0.1f;
  sgd.Apply(0, lr, "xx", grad, value);

  singa::Tensor v1 = value.Clone();
  v1.ToHost();
  const float* newv1 = v1.data<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(newv1[i], v[i] - g[i] * lr);
  }

  lr /= 2;
  grad.CopyDataFromHostPtr(g, 4);
  sgd.Apply(1, lr, "xx", grad, value);
  singa::Tensor v2 = value.Clone();
  v2.ToHost();
  const float* newv2 = v2.data<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(newv2[i], newv1[i] - g[i] * lr);
  }
}

TEST(SGD, ApplyWithMomentumCuda) {
  singa::SGD sgd;
  float lr = 0.1f;
  auto func = [](int step) { return step <= 5 ? 0.5f : 0.9f; };
  sgd.SetMomentumGenerator(func);
  const float v[4] = {0.1, 0.2, 0.3, 0.4};
  const float g[4] = {0.01, 0.02, 0.03, 0.04};

  auto dev = std::make_shared<singa::CudaGPU>();
  singa::Tensor value(singa::Shape{4}, dev), grad(singa::Shape{4}, dev);
  value.CopyDataFromHostPtr(v, 4);
  grad.CopyDataFromHostPtr(g, 4);

  sgd.Apply(0, lr, "xx", grad, value);

  singa::Tensor v1 = value.Clone();
  v1.ToHost();
  const float* newv1 = v1.data<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(newv1[i], v[i] - g[i] * lr);
  }

  grad.CopyDataFromHostPtr(g, 4);
  sgd.Apply(1, lr, "xx", grad, value);
  singa::Tensor v2 = value.Clone();
  v2.ToHost();
  const float* newv2 = v2.data<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(newv2[i], newv1[i] - (g[i] * lr + g[i] * lr * func(1)));
  }
}
#endif
