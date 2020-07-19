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

#include "../src/model/layer/slice.h"
#include "gtest/gtest.h"

using singa::Shape;
TEST(Slice, Setup) {
  singa::LayerConf conf;
  conf.set_type("singa_slice");
  auto slice_conf = conf.mutable_slice_conf();
  slice_conf->set_axis(1);
  slice_conf->add_slice_point(2);
  singa::Slice layer;
  layer.Setup({3u}, conf);
  auto s1 = layer.GetOutputSampleShape(0);
  EXPECT_EQ(s1[0], 2u);
  auto s2 = layer.GetOutputSampleShape(1);
  EXPECT_EQ(s2[0], 1u);
}

void ForwardSliceRowTest(std::shared_ptr<singa::Device> dev) {
  size_t a = 2u, b = 1u, c = 3u;
  singa::LayerConf conf;
  conf.set_type("singa_slice");
  auto slice_conf = conf.mutable_slice_conf();
  slice_conf->set_axis(0);
  slice_conf->add_slice_point(a);
  singa::Slice layer;
  layer.Setup({c}, conf);
  layer.ToDevice(dev);

  singa::Tensor t({a + b, c}, dev);
  singa::Uniform(-1.f, 1.f, &t);
  auto grads = layer.Forward(singa::kTrain, {t});
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

void ForwardSliceColumnTest(std::shared_ptr<singa::Device> dev) {
  size_t a = 2u, b = 1u, c = 3u;
  singa::LayerConf conf;
  conf.set_type("singa_slice");
  auto slice_conf = conf.mutable_slice_conf();
  slice_conf->set_axis(1);
  slice_conf->add_slice_point(a);
  singa::Slice layer;
  layer.Setup({a + b}, conf);
  layer.ToDevice(dev);

  singa::Tensor t({c, a + b}, dev);
  singa::Uniform(-1.f, 1.f, &t);
  auto out = layer.Forward(singa::kTrain, {t});
  EXPECT_EQ(out.size(), 2u);

  t.ToHost();
  const float* tptr = t.data<float>();

  out[0].ToHost();
  const float* outa = out[0].data<float>();
  for (size_t i = 0; i < c; i++)
    for (size_t j = 0; j < a; j++)
      EXPECT_FLOAT_EQ(outa[i * a + j], tptr[i * (a + b) + j]);
  out[1].ToHost();
  const float* outb = out[1].data<float>();
  for (size_t i = 0; i < c; i++)
    for (size_t j = 0; j < b; j++)
      EXPECT_FLOAT_EQ(outb[i * b + j], tptr[i * (a + b) + a + j]);
}

TEST(Slice, ForwardSliceRowCpp) { ForwardSliceRowTest(singa::defaultDevice); }

TEST(Slice, ForwardSliceColumn) {
  ForwardSliceColumnTest(singa::defaultDevice);
}

#ifdef USE_CUDA
TEST(Slice, ForwardSliceRowCuda) {
  ForwardSliceRowTest(std::make_shared<singa::CudaGPU>());
}

TEST(Slice, ForwardSliceColumnCuda) {
  ForwardSliceColumnTest(std::make_shared<singa::CudaGPU>());
}
#endif  // USE_CUDA

void BackwardSliceRowTest(std::shared_ptr<singa::Device> dev) {
  size_t a = 2u, b = 1u, c = 3u;
  singa::LayerConf conf;
  conf.set_type("singa_slice");
  auto slice_conf = conf.mutable_slice_conf();
  slice_conf->set_axis(0);
  slice_conf->add_slice_point(2);
  singa::Slice layer;
  layer.Setup({c}, conf);
  layer.ToDevice(dev);

  singa::Tensor t1({a, c}, dev);
  singa::Tensor t2({b, c}, dev);
  t1.SetValue(1.0f);
  t2.SetValue(2.0f);
  auto out = layer.Backward(singa::kTrain, {t1, t2});
  auto grad = out.first[0];

  grad.ToHost();
  const float* outptr = grad.data<float>();
  for (size_t i = 0; i < a; i++) {
    for (size_t j = 0; j < c; j++) EXPECT_FLOAT_EQ(outptr[i * c + j], 1.0f);
  }
  for (size_t i = a; i < a + b; i++) {
    for (size_t j = 0; j < c; j++) EXPECT_FLOAT_EQ(outptr[i * c + j], 2.0f);
  }
}

void BackwardSliceColumnTest(std::shared_ptr<singa::Device> dev) {
  size_t a = 2u, b = 1u, c = 3u;
  singa::LayerConf conf;
  conf.set_type("singa_slice");
  auto slice_conf = conf.mutable_slice_conf();
  slice_conf->set_axis(1);
  slice_conf->add_slice_point(2);
  singa::Slice layer;
  layer.Setup({a + b}, conf);
  layer.ToDevice(dev);

  singa::Tensor t1({c, a}, dev);
  singa::Tensor t2({c, b}, dev);
  t1.SetValue(1.0f);
  t2.SetValue(2.0f);
  auto out = layer.Backward(singa::kTrain, {t1, t2});
  auto grad = out.first[0];
  grad.ToHost();
  const float* outptr = grad.data<float>();
  for (size_t i = 0; i < c; i++) {
    for (size_t j = 0; j < a; j++)
      EXPECT_FLOAT_EQ(outptr[i * (a + b) + j], 1.0f);
  }
  for (size_t i = 0; i < c; i++) {
    for (size_t j = a; j < a + b; j++)
      EXPECT_FLOAT_EQ(outptr[i * (a + b) + j], 2.0f);
  }
}

TEST(Slice, BackwardSliceRowCpp) { BackwardSliceRowTest(singa::defaultDevice); }

TEST(Slice, BackwardSliceColumn) {
  BackwardSliceColumnTest(singa::defaultDevice);
}

#ifdef USE_CUDA
TEST(Slice, BackwardSliceRowCuda) {
  BackwardSliceRowTest(std::make_shared<singa::CudaGPU>());
}

TEST(Slice, BackwardSliceColumnCuda) {
  BackwardSliceColumnTest(std::make_shared<singa::CudaGPU>());
}
#endif  // USE_CUDA
