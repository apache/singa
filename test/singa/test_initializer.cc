/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"
#include "singa/model/initializer.h"

TEST(Initializer, Constant) {
  singa::init::Constant x;
  size_t n = 10;
  singa::Tensor t(singa::Shape{n});
  singa::FillerConf conf;
  conf.set_value(3.1f);
  x.Setup(conf);
  x.Fill(t);
  const float* xPtr = t.data<float>();
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(xPtr[i], 3.1f);
}

TEST(Initializer, Gaussian) {
  singa::init::Gaussian x;
  size_t n = 1000;
  singa::Tensor t(singa::Shape{n});
  singa::FillerConf conf;
  conf.set_mean(0.11f);
  conf.set_std(0.01f);
  x.Setup(conf);
  x.Fill(t);
  const float* xPtr = t.data<float>();
  float mean = 0.0f, std = 0.0f;
  for (size_t i = 0; i < n; i++) mean += xPtr[i];
  mean /= n;
  EXPECT_NEAR(mean, 0.11f, 1e-3);
  for (size_t i = 0; i < n; i++) std += (xPtr[i] - mean) * (xPtr[i] - mean);
  std /= n;
  std = sqrt(std);
  EXPECT_NEAR(std, 0.01f, 1e-3);
}

#ifdef USE_CUDA
TEST(Initializer, ConstantCUDA) {
  singa::init::Constant x;
  auto dev = std::make_shared<singa::CudaGPU>();
  size_t n = 10;
  singa::Tensor t(singa::Shape{n}, dev);
  singa::FillerConf conf;
  conf.set_value(3.1f);
  x.Setup(conf);
  x.Fill(t);
  t.ToHost();
  const float* xPtr = t.data<float>();
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(xPtr[i], 3.1f);

  singa::init::Constant y(-0.1f);
  singa::Tensor s(singa::Shape{n}, dev);
  y.Fill(s);
  s.ToHost();
  const float* sPtr = s.data<float>();
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(sPtr[i], -0.1f);
}

TEST(Initializer, GaussianCUDA) {
  singa::init::Gaussian x;
  auto dev = std::make_shared<singa::CudaGPU>();
  size_t n = 1000;
  singa::Tensor t(singa::Shape{n}, dev);
  singa::FillerConf conf;
  conf.set_mean(0.11f);
  conf.set_std(0.01f);
  x.Setup(conf);
  x.Fill(t);
  t.ToHost();
  const float* tPtr = t.data<float>();
  float mean = 0.0f, std = 0.0f;
  for (size_t i = 0; i < n; i++) mean += tPtr[i];
  mean /= n;
  EXPECT_NEAR(mean, 0.11f, 1e-2);
  for (size_t i = 0; i < n; i++) std += (tPtr[i] - mean) * (tPtr[i] - mean);
  std /= n;
  std = sqrt(std);
  EXPECT_NEAR(std, 0.01f, 1e-2);

  singa::init::Gaussian y(1.5f, 0.1f);
  singa::Tensor s(singa::Shape{n}, dev);
  y.Fill(s);
  s.ToHost();
  const float* sPtr = s.data<float>();
  for (size_t i = 0; i < n; i++) mean += sPtr[i];
  mean /= n;
  EXPECT_NEAR(mean, 1.5f, 0.1f);
  for (size_t i = 0; i < n; i++) std += (sPtr[i] - mean) * (sPtr[i] - mean);
  std /= n;
  std = sqrt(std);
  EXPECT_NEAR(std, 0.1f, 0.1f);
}

TEST(Initializer, XavierCUDA) {
  singa::init::Constant x;
  auto dev = std::make_shared<singa::CudaGPU>();
  size_t m = 30, n = 40;
  singa::Tensor t(singa::Shape{m, n}, dev);
  x.Fill(t);
  t.ToHost();
  const float* xPtr = t.data<float>();
  float mean = 0.0f;
  float high = -100.0f, low = 100.0f;
  for (size_t i = 0; i < n; i++) {
    mean += xPtr[i];
    if (high < xPtr[i]) high = xPtr[i];
    if (low > xPtr[i]) low = xPtr[i];
  }
  mean /= m * n;
  EXPECT_NEAR(mean, 0, 1e-2);
  float scale = sqrt(6.0f / (m + n));
  EXPECT_LT(high, scale);
  EXPECT_GT(low, -scale);
}

#endif
