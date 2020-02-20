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
#include "singa/singa_config.h"
#ifdef USE_CUDNN

#include <cudnn.h>
#include <math.h>  // exp

#include "../src/model/layer/cudnn_softmax.h"
#include "gtest/gtest.h"

// TODO(wangwei) add test for matrix input
using singa::CudnnSoftmax;
using singa::Shape;
TEST(CudnnSoftmax, Setup) {
  CudnnSoftmax sft;
  // EXPECT_EQ("CudnnSoftmax", sft.layer_type());

  singa::LayerConf conf;
  singa::SoftmaxConf* softmaxconf = conf.mutable_softmax_conf();
  softmaxconf->set_algorithm("fast");
  sft.Setup(Shape{1}, conf);
  EXPECT_EQ(CUDNN_SOFTMAX_FAST, sft.Algorithm());
}

TEST(CudnnSoftmax, Forward1D) {
  const float x[] = {1.f, 2.f, 0.f, -2.f, -3.f, -1.f};
  size_t n = sizeof(x) / sizeof(float);
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Shape shape = {n};
  singa::Tensor in(shape, cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  CudnnSoftmax sft;
  singa::LayerConf conf;
  singa::SoftmaxConf* softmaxconf = conf.mutable_softmax_conf();
  softmaxconf->set_algorithm("accurate");
  sft.Setup(Shape{1}, conf);
  singa::Tensor out = sft.Forward(singa::kTrain, in);
  out.ToHost();
  const float* yptr = out.data<float>();
  EXPECT_EQ(n, out.Size());

  float* y = new float[n];
  float sigma = 0.f;
  for (size_t i = 0; i < n; i++) sigma += exp(x[i]);
  for (size_t i = 0; i < n; i++) y[i] = exp(x[i]) / sigma;
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(y[i], yptr[i]);
  delete[] y;
}

TEST(CudnnSoftmax, Backward1D) {
  const float x[] = {1.f, 2.f, 3.f, -2.f, -3.f, -1.f};
  size_t n = sizeof(x) / sizeof(float);
  singa::Shape shape = {n};
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(shape, cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  CudnnSoftmax sft;
  singa::LayerConf conf;
  singa::SoftmaxConf* softmaxconf = conf.mutable_softmax_conf();
  softmaxconf->set_algorithm("accurate");
  sft.Setup(Shape{1}, conf);

  singa::Tensor out = sft.Forward(singa::kTrain, in);
  out.ToHost();
  const float* yptr = out.data<float>();

  const float grad[] = {2.f, -3.f, 1.f, 3.f, -1.f, -2.f};
  singa::Tensor out_diff(shape, cuda);
  out_diff.CopyDataFromHostPtr<float>(grad, n);
  const auto ret = sft.Backward(singa::kTrain, out_diff);
  singa::Tensor in_diff = ret.first;
  in_diff.ToHost();
  const float* xptr = in_diff.data<float>();

  float* dx = new float[n];
  float sigma = 0.f;
  for (size_t i = 0; i < n; i++) sigma += grad[i] * yptr[i];
  for (size_t i = 0; i < n; i++) dx[i] = (grad[i] - sigma) * yptr[i];
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(dx[i], xptr[i]);
  delete[] dx;
}

TEST(CudnnSoftmax, Forward2D) {
  const float x[] = {1.f, 2.f, 0.f, -2.f, -3.f, -1.f};
  size_t n = sizeof(x) / sizeof(float);
  size_t batch = 2, c = 3;
  singa::Shape shape = {batch, c};
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(shape, cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  CudnnSoftmax sft;
  singa::LayerConf conf;
  singa::SoftmaxConf* softmaxconf = conf.mutable_softmax_conf();
  softmaxconf->set_algorithm("accurate");
  sft.Setup(Shape{c}, conf);

  singa::Tensor out = sft.Forward(singa::kTrain, in);
  out.ToHost();
  const float* yptr = out.data<float>();
  EXPECT_EQ(n, out.Size());

  float* y = new float[n];
  float* sigma = new float[batch];
  for (size_t i = 0; i < batch; i++) sigma[i] = 0.f;
  for (size_t i = 0; i < n; i++) sigma[i / c] += exp(x[i]);
  for (size_t i = 0; i < n; i++) y[i] = exp(x[i]) / sigma[i / c];
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(y[i], yptr[i]);
  delete[] y;
  delete[] sigma;
}

TEST(CudnnSoftmax, Backward2D) {
  const float x[] = {1.f, 2.f, 3.f, -2.f, -3.f, -1.f};
  size_t n = sizeof(x) / sizeof(float);
  size_t batch = 2, c = 3;
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Shape shape = {batch, c};
  singa::Tensor in(shape, cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  CudnnSoftmax sft;
  singa::LayerConf conf;
  singa::SoftmaxConf* softmaxconf = conf.mutable_softmax_conf();
  softmaxconf->set_algorithm("accurate");
  sft.Setup(Shape{c}, conf);

  singa::Tensor out = sft.Forward(singa::kTrain, in);
  out.ToHost();
  const float* yptr = out.data<float>();

  const float grad[] = {2.f, -3.f, 1.f, 3.f, -1.f, -2.f};
  singa::Tensor out_diff(shape, cuda);
  out_diff.CopyDataFromHostPtr<float>(grad, n);
  const auto ret = sft.Backward(singa::kTrain, out_diff);
  singa::Tensor in_diff = ret.first;
  in_diff.ToHost();
  const float* xptr = in_diff.data<float>();

  float* dx = new float[n];
  float* sigma = new float[batch];
  for (size_t i = 0; i < batch; i++) sigma[i] = 0.f;
  for (size_t i = 0; i < n; i++) sigma[i / c] += grad[i] * yptr[i];
  for (size_t i = 0; i < n; i++) dx[i] = (grad[i] - sigma[i / c]) * yptr[i];
  for (size_t i = 0; i < n; i++) EXPECT_FLOAT_EQ(dx[i], xptr[i]);
  delete[] dx;
  delete[] sigma;
}
#endif  // USE_CUDNN
