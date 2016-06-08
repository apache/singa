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

#include "../src/model/layer/flatten.h"
#include "gtest/gtest.h"

using singa::Flatten;
TEST(Flatten, Setup) {
  Flatten flt;
  EXPECT_EQ("Flatten", flt.layer_type());

  singa::LayerConf conf;
  singa::FlattenConf *flattenconf = conf.mutable_flatten_conf();
  flattenconf->set_axis(1);

  flt.Setup(conf);
  EXPECT_EQ(1, flt.Axis());
}

TEST(Flatten, ForwardCPU) {
  const float x[] = { 1.f, 2.f, 3.f, -2.f, -3.f, -4.f, 1.5f, -1.5f, 0.f, -0.5f,
                      -2.f, -1.f };
  size_t n = sizeof(x) / sizeof(float);
  singa::Shape s = { 2, 1, 3, 2 };
  singa::Tensor in(s);
  in.CopyDataFromHostPtr<float>(x, n);

  int axis = 3;
  Flatten flt;
  singa::LayerConf conf;
  singa::FlattenConf *flattenconf = conf.mutable_flatten_conf();
  flattenconf->set_axis(axis);
  flt.Setup(conf);

  singa::Tensor out = flt.Forward(singa::kTrain, in);
  EXPECT_EQ(n, out.Size());
  EXPECT_EQ(6, out.shape(0));
  EXPECT_EQ(2, out.shape(1));
  const float *yptr = out.data<const float *>();
  for (size_t i = 0; i < n; i++)
    EXPECT_FLOAT_EQ(x[i], yptr[i]);
}

TEST(Flatten, BackwardCPU) {
  // directly use input as the output_grad for backward
  // note that only the shape of input really matters
  const float dy[] = { 1.f, 2.f, 3.f, -2.f, -3.f, -4.f, 1.5f, -1.5f, 0.f, -0.5f,
                       -2.f, -1.f };
  size_t n = sizeof(dy) / sizeof(float);
  singa::Tensor in(singa::Shape {
    2, 1, 3, 2
  });
  in.CopyDataFromHostPtr<float>(dy, n);

  int axis = 2;
  Flatten flt;
  singa::LayerConf conf;
  singa::FlattenConf *flattenconf = conf.mutable_flatten_conf();
  flattenconf->set_axis(axis);
  flt.Setup(conf);

  singa::Tensor temp = flt.Forward(singa::kTrain, in);
  const auto out = flt.Backward(singa::kTrain, temp);
  const float *xptr = out.first.data<const float *>();
  EXPECT_EQ(n, out.first.Size());
  EXPECT_EQ(2, out.first.shape(0));
  EXPECT_EQ(1, out.first.shape(1));
  EXPECT_EQ(3, out.first.shape(2));
  EXPECT_EQ(2, out.first.shape(3));
  for (size_t i = 0; i < n; i++)
    EXPECT_FLOAT_EQ(dy[i], xptr[i]);
}

#ifdef USE_CUDA
TEST(Flatten, ForwardGPU) {
  const float x[] = { 1.f, 2.f, 3.f, -2.f, -3.f, -4.f, 1.5f, -1.5f, 0.f, -0.5f,
                      -2.f, -1.f };
  size_t n = sizeof(x) / sizeof(float);
  singa::CudaGPU cuda(0, 1);
  singa::Tensor in(singa::Shape {
    2, 1, 3, 2
  },
                   &cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  int axis = 3;
  Flatten flt;
  singa::LayerConf conf;
  singa::FlattenConf *flattenconf = conf.mutable_flatten_conf();
  flattenconf->set_axis(axis);
  flt.Setup(conf);

  singa::Tensor out = flt.Forward(singa::kTrain, in);
  singa::CppCPU host(0, 1);
  out.ToDevice(&host);
  EXPECT_EQ(n, out.Size());
  EXPECT_EQ(6, out.shape(0));
  EXPECT_EQ(2, out.shape(1));
  const float *yptr = out.data<const float *>();
  for (size_t i = 0; i < n; i++)
    EXPECT_FLOAT_EQ(x[i], yptr[i]);
}

TEST(Flatten, BackwardGPU) {
  // directly use input as the output_grad for backward
  // note that only the shape of input really matters
  const float dy[] = { 1.f, 2.f, 3.f, -2.f, -3.f, -4.f, 1.5f, -1.5f, 0.f, -0.5f,
                       -2.f, -1.f };
  size_t n = sizeof(dy) / sizeof(float);
  singa::CudaGPU cuda(0, 1);
  singa::Tensor in(singa::Shape {
    2, 1, 3, 2
  },
                   &cuda);
  in.CopyDataFromHostPtr<float>(dy, n);

  int axis = 2;
  Flatten flt;
  singa::LayerConf conf;
  singa::FlattenConf *flattenconf = conf.mutable_flatten_conf();
  flattenconf->set_axis(axis);
  flt.Setup(conf);

  singa::Tensor out = flt.Forward(singa::kTrain, in);
  const auto ret = flt.Backward(singa::kTrain, out);
  singa::CppCPU host(0, 1);
  singa::Tensor in_diff = ret.first;
  in_diff.ToDevice(&host);
  const float *xptr = in_diff.data<const float *>();
  EXPECT_EQ(n, in_diff.Size());
  EXPECT_EQ(2, in_diff.shape(0));
  EXPECT_EQ(1, in_diff.shape(1));
  EXPECT_EQ(3, in_diff.shape(2));
  EXPECT_EQ(2, in_diff.shape(3));
  for (size_t i = 0; i < n; i++)
    EXPECT_FLOAT_EQ(dy[i], xptr[i]);
}
#endif // USE_CUDA
