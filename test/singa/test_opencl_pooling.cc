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

#include "../src/model/layer/opencl_pooling.h"
#include "../src/model/layer/pooling.h"
#include "gtest/gtest.h"

#ifdef USE_OPENCL

using singa::OpenclDevice;
using singa::OpenclPooling;
using singa::Shape;

TEST(OpenclPooling, Setup) {
  OpenclPooling pool;
  EXPECT_EQ("OpenclPooling", pool.layer_type());

  singa::LayerConf conf;
  singa::PoolingConf *poolconf = conf.mutable_pooling_conf();
  poolconf->set_pool(singa::PoolingConf_PoolMethod_MAX);
  poolconf->set_kernel_h(1);
  poolconf->set_kernel_w(2);
  poolconf->set_pad_h(1);
  poolconf->set_pad_w(0);
  poolconf->set_stride_h(2);
  poolconf->set_stride_w(1);
  pool.Setup(Shape{1, 3, 3}, conf);

  EXPECT_EQ(singa::PoolingConf_PoolMethod_MAX, pool.pool_method());
  EXPECT_EQ(1u, pool.kernel_h());
  EXPECT_EQ(2u, pool.kernel_w());
  EXPECT_EQ(1u, pool.pad_h());
  EXPECT_EQ(0u, pool.pad_w());
  EXPECT_EQ(2u, pool.stride_h());
  EXPECT_EQ(1u, pool.stride_w());
  EXPECT_EQ(1u, pool.channels());
  EXPECT_EQ(3u, pool.height());
  EXPECT_EQ(3u, pool.width());
}

TEST(OpenclPooling, Forward) {
  const size_t batchsize = 2, c = 1, h = 3, w = 3;
  const float x[batchsize * c * h * w] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                          7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  auto ocl = std::make_shared<OpenclDevice>();
  singa::Tensor in(singa::Shape{batchsize, c, h, w}, ocl);
  in.CopyDataFromHostPtr(x, batchsize * c * h * w);

  OpenclPooling pool;
  singa::LayerConf conf;
  singa::PoolingConf *poolconf = conf.mutable_pooling_conf();
  poolconf->set_pool(singa::PoolingConf_PoolMethod_MAX);
  poolconf->set_kernel_h(2);
  poolconf->set_kernel_w(2);
  poolconf->set_pad_h(0);
  poolconf->set_pad_w(0);
  poolconf->set_stride_h(1);
  poolconf->set_stride_w(1);
  pool.Setup(Shape{1, 3, 3}, conf);

  // Parameter "flag" does not influence pooling
  singa::Tensor out1 = pool.Forward(singa::kTrain, in);
  out1.ToHost();
  const float *outptr1 = out1.data<float>();
  // Input: 3*3; kernel: 2*2; stride: 1*1; no padding.
  EXPECT_EQ(8u, out1.Size());
  EXPECT_EQ(5.0f, outptr1[0]);
  EXPECT_EQ(6.0f, outptr1[1]);
  EXPECT_EQ(8.0f, outptr1[2]);
  EXPECT_EQ(9.0f, outptr1[3]);
  EXPECT_EQ(5.0f, outptr1[4]);
  EXPECT_EQ(6.0f, outptr1[5]);
  EXPECT_EQ(8.0f, outptr1[6]);
  EXPECT_EQ(9.0f, outptr1[7]);
}

TEST(OpenclPooling, Backward) {
  // src_data
  const size_t batchsize = 2, c = 1, src_h = 3, src_w = 3;
  const float x[batchsize * c * src_h * src_w] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  auto ocl = std::make_shared<OpenclDevice>();
  singa::Tensor in(singa::Shape{batchsize, c, src_h, src_w}, ocl);
  in.CopyDataFromHostPtr(x, batchsize * c * src_h * src_w);

  OpenclPooling pool;
  singa::LayerConf conf;
  singa::PoolingConf *poolconf = conf.mutable_pooling_conf();
  poolconf->set_pool(singa::PoolingConf_PoolMethod_MAX);
  poolconf->set_kernel_h(2);
  poolconf->set_kernel_w(2);
  poolconf->set_pad_h(0);
  poolconf->set_pad_w(0);
  poolconf->set_stride_h(1);
  poolconf->set_stride_w(1);
  pool.Setup(Shape{1, 3, 3}, conf);

  singa::Tensor out1 = pool.Forward(singa::kTrain, in);

  // grad
  const size_t grad_h = 2, grad_w = 2;
  const float dy[batchsize * c * grad_h * grad_w] = {0.1f, 0.2f, 0.3f, 0.4f,
                                                     0.1f, 0.2f, 0.3f, 0.4f};
  singa::Tensor grad(singa::Shape{batchsize, c, grad_h, grad_w}, ocl);
  grad.CopyDataFromHostPtr(dy, batchsize * c * grad_h * grad_w);

  const auto ret = pool.Backward(singa::kTrain, grad);
  singa::Tensor in_grad = ret.first;
  in_grad.ToHost();
  const float *dx = in_grad.data<float>();
  EXPECT_EQ(18u, in_grad.Size());
  EXPECT_EQ(0.0f, dx[0]);
  EXPECT_EQ(0.0f, dx[1]);
  EXPECT_EQ(0.0f, dx[2]);
  EXPECT_EQ(0.0f, dx[3]);
  EXPECT_EQ(0.1f, dx[4]);
  EXPECT_EQ(0.2f, dx[5]);
  EXPECT_EQ(0.0f, dx[6]);
  EXPECT_EQ(0.3f, dx[7]);
  EXPECT_EQ(0.4f, dx[8]);
  EXPECT_EQ(0.0f, dx[9]);
  EXPECT_EQ(0.0f, dx[10]);
  EXPECT_EQ(0.0f, dx[11]);
  EXPECT_EQ(0.0f, dx[12]);
  EXPECT_EQ(0.1f, dx[13]);
  EXPECT_EQ(0.2f, dx[14]);
  EXPECT_EQ(0.0f, dx[15]);
  EXPECT_EQ(0.3f, dx[16]);
  EXPECT_EQ(0.4f, dx[17]);
}

#endif  // USE_OPENCL
