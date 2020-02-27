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

#include "../src/model/layer/concat.h"
#include "gtest/gtest.h"

using singa::Shape;

TEST(Concat, Setup) {
  Shape s1{2u, 3u};
  Shape s2{1u, 3u};
  singa::LayerConf conf;
  conf.set_type("singa_concat");
  conf.mutable_concat_conf()->set_axis(0);
  singa::Concat layer;
  layer.Setup({{3u}, {3u}}, conf);
  auto s = layer.GetOutputSampleShape();
  EXPECT_EQ(s[0], 3u);
}

void ForwardConcatRowTest(std::shared_ptr<singa::Device> dev) {
  size_t a = 2u, b = 1u, c = 3u;
  singa::Tensor t1({a, c}, dev);
  singa::Tensor t2({b, c}, dev);
  singa::LayerConf conf;
  conf.set_type("singa_concat");
  conf.mutable_concat_conf()->set_axis(0);
  singa::Concat layer;
  layer.Setup({{c}, {c}}, conf);
  layer.ToDevice(dev);

  t1.SetValue(1.0f);
  t2.SetValue(2.0f);
  auto out = layer.Forward(singa::kTrain, {t1, t2});
  EXPECT_EQ(out.size(), 1u);

  out[0].ToHost();
  const float* outptr = out[0].data<float>();
  for (size_t i = 0; i < a; i++) {
    for (size_t j = 0; j < c; j++) EXPECT_FLOAT_EQ(outptr[i * c + j], 1.0f);
  }
  for (size_t i = a; i < a + b; i++) {
    for (size_t j = 0; j < c; j++) EXPECT_FLOAT_EQ(outptr[i * c + j], 2.0f);
  }
}

void ForwardConcatColumnTest(std::shared_ptr<singa::Device> dev) {
  size_t a = 2u, b = 1u, c = 3u;
  singa::Tensor t1({c, a}, dev);
  singa::Tensor t2({c, b}, dev);
  singa::LayerConf conf;
  conf.set_type("singa_concat");
  conf.mutable_concat_conf()->set_axis(1);
  singa::Concat layer;
  layer.Setup({{a}, {b}}, conf);
  layer.ToDevice(dev);

  t1.SetValue(1.0f);
  t2.SetValue(2.0f);
  auto out = layer.Forward(singa::kTrain, {t1, t2});
  EXPECT_EQ(out.size(), 1u);
  out[0].ToHost();
  const float* outptr = out[0].data<float>();
  for (size_t i = 0; i < c; i++) {
    for (size_t j = 0; j < a; j++)
      EXPECT_FLOAT_EQ(outptr[i * (a + b) + j], 1.0f);
  }
  for (size_t i = 0; i < c; i++) {
    for (size_t j = a; j < a + b; j++)
      EXPECT_FLOAT_EQ(outptr[i * (a + b) + j], 2.0f);
  }
}
TEST(Concat, ForwardConcatRowCpp) {
  ForwardConcatRowTest(singa::defaultDevice);
}

TEST(Concat, ForwardConcatColumnCpp) {
  ForwardConcatColumnTest(singa::defaultDevice);
}

#ifdef USE_CUDA
TEST(Concat, ForwardConcatRowCuda) {
  ForwardConcatRowTest(std::make_shared<singa::CudaGPU>());
}

TEST(Concat, ForwardConcatColumnCuda) {
  ForwardConcatColumnTest(std::make_shared<singa::CudaGPU>());
}
#endif  // USE_CUDA

void BackwardConcatRowTest(std::shared_ptr<singa::Device> dev) {
  size_t a = 2u, b = 1u, c = 3u;
  singa::LayerConf conf;
  conf.set_type("singa_concat");
  conf.mutable_concat_conf()->set_axis(0);
  singa::Concat layer;
  layer.Setup({{c}, {c}}, conf);
  layer.ToDevice(dev);

  singa::Tensor t1({a, c}, dev);
  singa::Tensor t2({b, c}, dev);
  t1.SetValue(1.0f);
  t2.SetValue(2.0f);
  layer.Forward(singa::kTrain, {t1, t2});

  singa::Tensor t({a + b, c}, dev);
  singa::Uniform(-1.f, 1.f, &t);
  auto out = layer.Backward(singa::kTrain, {t});
  auto grads = out.first;
  EXPECT_EQ(grads.size(), 2u);

  t.ToHost();
  const float* tptr = t.data<float>();

  grads[0].ToHost();
  const float* outa = grads[0].data<float>();
  for (size_t i = 0; i < a; i++)
    for (size_t j = 0; j < c; j++)
      EXPECT_FLOAT_EQ(outa[i * c + j], tptr[i * c + j]);
  grads[1].ToHost();
  const float* outb = grads[1].data<float>();
  for (size_t i = 0; i < b; i++)
    for (size_t j = 0; j < c; j++)
      EXPECT_FLOAT_EQ(outb[i * c + j], tptr[(i + a) * c + j]);
}

void BackwardConcatColumnTest(std::shared_ptr<singa::Device> dev) {
  size_t a = 2u, b = 1u, c = 3u;
  singa::LayerConf conf;
  conf.set_type("singa_concat");
  conf.mutable_concat_conf()->set_axis(1);
  singa::Concat layer;
  layer.Setup({{a}, {b}}, conf);
  layer.ToDevice(dev);

  singa::Tensor t1({c, a}, dev);
  singa::Tensor t2({c, b}, dev);
  t1.SetValue(1.0f);
  t2.SetValue(2.0f);
  layer.Forward(singa::kTrain, {t1, t2});

  singa::Tensor t({c, a + b}, dev);
  singa::Uniform(-1.f, 1.f, &t);
  auto out = layer.Backward(singa::kTrain, {t});
  auto grads = out.first;
  EXPECT_EQ(grads.size(), 2u);

  t.ToHost();
  const float* tptr = t.data<float>();

  grads[0].ToHost();
  const float* outa = grads[0].data<float>();
  for (size_t i = 0; i < c; i++)
    for (size_t j = 0; j < a; j++)
      EXPECT_FLOAT_EQ(outa[i * a + j], tptr[i * (a + b) + j]);
  grads[1].ToHost();
  const float* outb = grads[1].data<float>();
  for (size_t i = 0; i < c; i++)
    for (size_t j = 0; j < b; j++)
      EXPECT_FLOAT_EQ(outb[i * b + j], tptr[i * (a + b) + a + j]);
}

TEST(Concat, BackwardConcatRowCpp) {
  BackwardConcatRowTest(singa::defaultDevice);
}

TEST(Concat, BackwardConcatColumn) {
  BackwardConcatColumnTest(singa::defaultDevice);
}

#ifdef USE_CUDA
TEST(Concat, BackwardConcatRowCuda) {
  BackwardConcatRowTest(std::make_shared<singa::CudaGPU>());
}

TEST(Concat, BackwardConcatColumnCuda) {
  BackwardConcatColumnTest(std::make_shared<singa::CudaGPU>());
}
#endif  // USE_CUDA
