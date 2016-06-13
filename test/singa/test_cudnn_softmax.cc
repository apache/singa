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
#include "singa_config.h"
#ifdef USE_CUDNN

#include "../src/model/layer/cudnn_softmax.h"
#include "gtest/gtest.h"
#include <math.h>  // exp
#include <cudnn.h>

using singa::CudnnSoftmax;
TEST(CudnnSoftmax, Setup) {
  CudnnSoftmax sft;
  EXPECT_EQ("CudnnSoftmax", sft.layer_type());

  singa::LayerConf conf;
  singa::SoftmaxConf* softmaxconf = conf.mutable_softmax_conf();
  softmaxconf->set_axis(2);

  sft.Setup(conf);
  sft.InitCudnn(1, singa::kFloat32);
  EXPECT_EQ(2, sft.Axis());
}

TEST(CudnnSoftmax, Forward) {
  const float x[] = {1.0f, 2.0f, 0.0f, -2.0f, -3.0f, -1.0};
  size_t n = sizeof(x) / sizeof(float);
  singa::CudaGPU cuda(0, 1);
  singa::Tensor in(singa::Shape{n}, &cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  int axis = 1;
  CudnnSoftmax sft;
  singa::LayerConf conf;
  singa::SoftmaxConf* softmaxconf = conf.mutable_softmax_conf();
  softmaxconf->set_axis(axis);
  sft.Setup(conf);
  sft.InitCudnn(n, singa::kFloat32);

  singa::Tensor out = sft.Forward(singa::kTrain, in);
  singa::CppCPU host(0, 1);
  out.ToDevice(&host);
  const float* yptr = out.data<const float*>();
  EXPECT_EQ(n, out.Size());

  float* y = new float[n];
  float sigma = 0.f;
  for (size_t i = 0; i < n; i++) sigma += exp(x[i]);
  for (size_t i = 0; i < n; i++) y[i] = exp(x[i]) / sigma;
  EXPECT_FLOAT_EQ(y[0], yptr[0]);
  EXPECT_FLOAT_EQ(y[4], yptr[4]);
  EXPECT_FLOAT_EQ(y[5], yptr[5]);
}

TEST(CudnnSoftmax, Backward) {
  const float x[] = {1.0f, 2.0f, 3.0f, -2.0f, -3.0f, -1.0};
  size_t n = sizeof(x) / sizeof(float);
  singa::CudaGPU cuda(0, 1);
  singa::Tensor in(singa::Shape{n}, &cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  int axis = 1;
  CudnnSoftmax sft;
  singa::LayerConf conf;
  singa::SoftmaxConf* softmaxconf = conf.mutable_softmax_conf();
  softmaxconf->set_axis(axis);
  sft.Setup(conf);
  singa::Tensor out = sft.Forward(singa::kTrain, in);
  singa::CppCPU host(0, 1);
  out.ToDevice(&host);
  const float* yptr = out.data<const float*>();

  const float grad[] = {2.0f, -3.0f, 1.0f, 3.0f, -1.0f, -2.0};
  singa::Tensor out_diff(singa::Shape{n}, &cuda);
  out_diff.CopyDataFromHostPtr<float>(grad, n);
  const auto ret = sft.Backward(singa::kTrain, out_diff);
  singa::Tensor in_diff = ret.first;
  in_diff.ToDevice(&host);
  const float* xptr = in_diff.data<const float*>();

  float* dx = new float[n];
  float sigma = 0.f;
  for (size_t i = 0; i < n; i++) sigma += grad[i] * yptr[i];
  for (size_t i = 0; i < n; i++) dx[i] = (grad[i] - sigma) * yptr[i];
  EXPECT_FLOAT_EQ(dx[0], xptr[0]);
  EXPECT_FLOAT_EQ(dx[4], xptr[4]);
  EXPECT_FLOAT_EQ(dx[5], xptr[5]);
}
#endif  // USE_CUDNN
