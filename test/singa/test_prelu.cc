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

#include "../src/model/layer/prelu.h"
#include "gtest/gtest.h"
#include "singa/singa_config.h"

using singa::PReLU;
using singa::Shape;
TEST(PReLU, Setup) {
  PReLU prelu;
  // EXPECT_EQ("PReLU", prelu.layer_type());

  singa::LayerConf conf;
  singa::PReLUConf *preluconf = conf.mutable_prelu_conf();
  preluconf->set_channel_shared(true);
  preluconf->set_format("NHWC");

  prelu.Setup(Shape{4}, conf);
  EXPECT_EQ(true, prelu.Channel_shared());
  EXPECT_EQ("NHWC", prelu.Format());
}

TEST(PReLU, ForwardCPU) {
  const float x[] = {1.f,  2.f, 3.f,  -2.f, -3.f, -1.f,
                     -1.f, 2.f, -1.f, -2.f, -2.f, -1.f};
  size_t n = sizeof(x) / sizeof(float);
  size_t batchsize = 2, c = 3, h = 2, w = 1;
  singa::Tensor in(singa::Shape{batchsize, h, w, c});
  in.CopyDataFromHostPtr<float>(x, n);

  PReLU prelu;
  singa::LayerConf conf;
  singa::PReLUConf *preluconf = conf.mutable_prelu_conf();
  preluconf->set_channel_shared(false);
  preluconf->set_format("NHWC");
  prelu.Setup(Shape{h, w, c}, conf);

  const float neg_slope[] = {0.25f, 0.5f, 0.75f};
  singa::Tensor a(singa::Shape{c});
  a.CopyDataFromHostPtr<float>(neg_slope, c);
  prelu.Set_a(a);

  singa::Tensor out = prelu.Forward(singa::kTrain, in);
  const float *yptr = out.data<float>();
  EXPECT_EQ(n, out.Size());

  float *y = new float[n];
  size_t div_factor = prelu.Channel_shared() ? c : 1;
  if (prelu.Format() == "NCHW") {
    for (size_t i = 0; i < n; i++) {
      size_t pos = i / (h * w) % c / div_factor;
      y[i] = std::max(x[i], 0.f) + neg_slope[pos] * std::min(x[i], 0.f);
    }
  } else if (prelu.Format() == "NHWC") {
    for (size_t i = 0; i < n; i++) {
      size_t pos = i % c / div_factor;
      y[i] = std::max(x[i], 0.f) + neg_slope[pos] * std::min(x[i], 0.f);
    }
  }
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(y[i], yptr[i]);
  delete[] y;
}

TEST(PReLU, BackwardCPU) {
  const float x[] = {1.f,  2.f, 3.f,  -2.f, -3.f, -1.f,
                     -1.f, 2.f, -1.f, -2.f, -2.f, -1.f};
  size_t n = sizeof(x) / sizeof(float);
  size_t batchsize = 2, c = 3, h = 2, w = 1;
  singa::Tensor in(singa::Shape{batchsize, c, h, w});
  in.CopyDataFromHostPtr<float>(x, n);

  PReLU prelu;
  singa::LayerConf conf;
  singa::PReLUConf *preluconf = conf.mutable_prelu_conf();
  preluconf->set_channel_shared(false);
  preluconf->set_format("NCHW");
  prelu.Setup(Shape{c, h, w}, conf);

  const float neg_slope[] = {0.25f, 0.5f, 0.75f};
  singa::Tensor a(singa::Shape{c});
  a.CopyDataFromHostPtr<float>(neg_slope, c);
  prelu.Set_a(a);

  singa::Tensor out = prelu.Forward(singa::kTrain, in);

  const float grad[] = {1.f, 2.f,  -2.f, -1.f, -1.f, -3.f,
                        2.f, -2.f, 1.f,  1.f,  -2.f, 0.f};
  singa::Tensor out_diff(singa::Shape{batchsize, c, h, w});
  out_diff.CopyDataFromHostPtr<float>(grad, n);
  const auto ret = prelu.Backward(singa::kTrain, out_diff);
  const float *xptr = ret.first.data<float>();
  const float *aptr = ret.second.at(0).data<float>();
  float *dx = new float[n];
  size_t div_factor = prelu.Channel_shared() ? c : 1;
  size_t params = prelu.Channel_shared() ? 1 : c;
  float da[] = {0.f, 0.f, 0.f};
  if (prelu.Format() == "NCHW") {
    for (size_t i = 0; i < n; i++) {
      size_t pos = i / (h * w) % c / div_factor;
      dx[i] = grad[i] *
              (std::max(x[i], 0.f) + neg_slope[pos] * std::min(x[i], 0.f));
    }
    for (size_t i = 0; i < n; i++) {
      size_t pos = i / (h * w) % c / div_factor;
      da[pos] += grad[i] * std::min(x[i], 0.f);
    }
  } else if (prelu.Format() == "NHWC") {
    for (size_t i = 0; i < n; i++) {
      size_t pos = i % c / div_factor;
      dx[i] = grad[i] *
              (std::max(x[i], 0.f) + neg_slope[pos] * std::min(x[i], 0.f));
    }
    for (size_t i = 0; i < n; i++) {
      size_t pos = i % c / div_factor;
      da[pos] += grad[i] * std::min(x[i], 0.f);
    }
  }
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(dx[i], xptr[i]);
  for (size_t i = 0; i < params; i++) EXPECT_FLOAT_EQ(da[i], aptr[i]);
  delete[] dx;
}

#ifdef USE_CUDA
TEST(PReLU, ForwardGPU) {
  const float x[] = {1.f,  2.f, 3.f,  -2.f, -3.f, -1.f,
                     -1.f, 2.f, -1.f, -2.f, -2.f, -1.f};
  size_t n = sizeof(x) / sizeof(float);
  size_t batchsize = 2, c = 3, h = 2, w = 1;
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(singa::Shape{batchsize, h, w, c}, cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  PReLU prelu;
  singa::LayerConf conf;
  singa::PReLUConf *preluconf = conf.mutable_prelu_conf();
  preluconf->set_channel_shared(false);
  preluconf->set_format("NHWC");
  prelu.Setup(Shape{h, w, c}, conf);

  const float neg_slope[] = {0.25f, 0.5f, 0.75f};
  singa::Tensor a(singa::Shape{c}, cuda);
  a.CopyDataFromHostPtr<float>(neg_slope, c);
  prelu.Set_a(a);

  singa::Tensor out = prelu.Forward(singa::kTrain, in);
  out.ToHost();
  const float *yptr = out.data<float>();
  EXPECT_EQ(n, out.Size());

  float *y = new float[n];
  size_t div_factor = prelu.Channel_shared() ? c : 1;
  if (prelu.Format() == "NCHW") {
    for (size_t i = 0; i < n; i++) {
      size_t pos = i / (h * w) % c / div_factor;
      y[i] = std::max(x[i], 0.f) + neg_slope[pos] * std::min(x[i], 0.f);
    }
  } else if (prelu.Format() == "NHWC") {
    for (size_t i = 0; i < n; i++) {
      size_t pos = i % c / div_factor;
      y[i] = std::max(x[i], 0.f) + neg_slope[pos] * std::min(x[i], 0.f);
    }
  }
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(y[i], yptr[i]);
  delete[] y;
}

TEST(PReLU, BackwardGPU) {
  const float x[] = {1.f,  2.f, 3.f,  -2.f, -3.f, -1.f,
                     -1.f, 2.f, -1.f, -2.f, -2.f, -1.f};
  size_t n = sizeof(x) / sizeof(float);
  size_t batchsize = 2, c = 3, h = 2, w = 1;
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(singa::Shape{batchsize, c, h, w}, cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  PReLU prelu;
  singa::LayerConf conf;
  singa::PReLUConf *preluconf = conf.mutable_prelu_conf();
  preluconf->set_channel_shared(false);
  preluconf->set_format("NCHW");
  prelu.Setup(Shape{c, h, w}, conf);

  const float neg_slope[] = {0.25f, 0.5f, 0.75f};
  singa::Tensor a(singa::Shape{c}, cuda);
  a.CopyDataFromHostPtr<float>(neg_slope, c);
  prelu.Set_a(a);

  singa::Tensor out = prelu.Forward(singa::kTrain, in);
  const float grad[] = {1.f, 2.f,  -2.f, -1.f, -1.f, -3.f,
                        2.f, -2.f, 1.f,  1.f,  -2.f, 0.f};
  singa::Tensor out_diff(singa::Shape{batchsize, c, h, w}, cuda);
  out_diff.CopyDataFromHostPtr<float>(grad, n);
  const auto ret = prelu.Backward(singa::kTrain, out_diff);

  singa::Tensor in_diff = ret.first;
  in_diff.ToHost();
  const float *xptr = in_diff.data<float>();
  singa::Tensor a_diff = ret.second.at(0);
  a_diff.ToHost();
  const float *aptr = a_diff.data<float>();
  float *dx = new float[n];
  size_t div_factor = prelu.Channel_shared() ? c : 1;
  size_t params = prelu.Channel_shared() ? 1 : c;
  float da[] = {0.f, 0.f, 0.f};
  if (prelu.Format() == "NCHW") {
    for (size_t i = 0; i < n; i++) {
      size_t pos = i / (h * w) % c / div_factor;
      dx[i] = grad[i] *
              (std::max(x[i], 0.f) + neg_slope[pos] * std::min(x[i], 0.f));
    }
    for (size_t i = 0; i < n; i++) {
      size_t pos = i / (h * w) % c / div_factor;
      da[pos] += grad[i] * std::min(x[i], 0.f);
    }
  } else if (prelu.Format() == "NHWC") {
    for (size_t i = 0; i < n; i++) {
      size_t pos = i % c / div_factor;
      dx[i] = grad[i] *
              (std::max(x[i], 0.f) + neg_slope[pos] * std::min(x[i], 0.f));
    }
    for (size_t i = 0; i < n; i++) {
      size_t pos = i % c / div_factor;
      da[pos] += grad[i] * std::min(x[i], 0.f);
    }
  }
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(dx[i], xptr[i]);
  for (size_t i = 0; i < params; i++) EXPECT_FLOAT_EQ(da[i], aptr[i]);
  delete[] dx;
}
#endif
