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
#include "../src/model/operation/pooling.h"

#include "gtest/gtest.h"

using namespace singa;

#ifdef USE_MKLDNN
TEST(OperationPooling, Forward) {
  const size_t batchsize = 2, c = 1, h = 3, w = 3;
  const float x[batchsize * c * h * w] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                          7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  Tensor in(Shape{batchsize, c, h, w});
  in.CopyDataFromHostPtr(x, batchsize * c * h * w);


  PoolingHandle pool_handle(in, {2, 2}, {1,1}, {0,0}, true);
  Tensor out1 = CpuPoolingForward(pool_handle, in);

  // Parameter "flag" does not influence pooling
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

TEST(OperationPooling, ForwardAverage) {
  const size_t batchsize = 2, c = 1, h = 3, w = 3;
  const float x[batchsize * c * h * w] = {1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f,
                                          7.0f, 8.0f, 9.0f,

                                          1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f,
                                          7.0f, 8.0f, 9.0f};
  Tensor in(Shape{batchsize, c, h, w});
  in.CopyDataFromHostPtr(x, batchsize * c * h * w);


  PoolingHandle pool_handle(in, {2, 2}, {1,1}, {0,0}, false);
  Tensor out1 = CpuPoolingForward(pool_handle, in);

  // Parameter "flag" does not influence pooling
  const float *outptr1 = out1.data<float>();
  // Input: 3*3; kernel: 2*2; stride: 1*1; no padding.
  EXPECT_EQ(8u, out1.Size());
  EXPECT_EQ(3.0f, outptr1[0]);
  EXPECT_EQ(4.0f, outptr1[1]);
  EXPECT_EQ(6.0f, outptr1[2]);
  EXPECT_EQ(7.0f, outptr1[3]);
  EXPECT_EQ(3.0f, outptr1[4]);
  EXPECT_EQ(4.0f, outptr1[5]);
  EXPECT_EQ(6.0f, outptr1[6]);
  EXPECT_EQ(7.0f, outptr1[7]);

}


TEST(OperationPooling, Backward) {
  // src_data
  const size_t batchsize = 2, c = 1, src_h = 3, src_w = 3;
  const float x[batchsize * c * src_h * src_w] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  Tensor in(Shape{batchsize, c, src_h, src_w});
  in.CopyDataFromHostPtr(x, batchsize * c * src_h * src_w);


  PoolingHandle pool_handle(in, {2, 2}, {1,1}, {0,0}, true);

  Tensor out = CpuPoolingForward(pool_handle, in);

  // grad - bwd
  const size_t grad_h = 2, grad_w = 2;
  const float dy[batchsize * c * grad_h * grad_w] = {0.1f, 0.2f, 0.3f, 0.4f,
                                                     0.1f, 0.2f, 0.3f, 0.4f};
  Tensor grad(Shape{batchsize, c, grad_h, grad_w});
  grad.CopyDataFromHostPtr(dy, batchsize * c * grad_h * grad_w);

  Tensor in_grad = CpuPoolingBackward(pool_handle, grad, in, out);


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

TEST(OperationPooling, BackwardAvg) {
  // src_data
  const size_t batchsize = 2, c = 1, src_h = 3, src_w = 3;
  const float x[batchsize * c * src_h * src_w] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,

      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f};
  Tensor in(Shape{batchsize, c, src_h, src_w});
  in.CopyDataFromHostPtr(x, batchsize * c * src_h * src_w);


  PoolingHandle pool_handle(in, {2, 2}, {1,1}, {0,0}, false);

  Tensor out = CpuPoolingForward(pool_handle, in);

  // grad - bwd
  const size_t grad_h = 2, grad_w = 2;
  const float dy[batchsize * c * grad_h * grad_w] = {0.1f, 0.2f, 0.3f, 0.4f,
                                                     0.1f, 0.2f, 0.3f, 0.4f};
  Tensor grad(Shape{batchsize, c, grad_h, grad_w});
  grad.CopyDataFromHostPtr(dy, batchsize * c * grad_h * grad_w);

  Tensor in_grad = CpuPoolingBackward(pool_handle, grad, in, out);

  const float *dx = in_grad.data<float>();
  EXPECT_EQ(18u, in_grad.Size());
  EXPECT_NEAR(0.0250f, dx[0], 1e-5f);
  EXPECT_NEAR(0.0750f, dx[1], 1e-5f);
  EXPECT_NEAR(0.0500f, dx[2], 1e-5f);
  EXPECT_NEAR(0.1000f, dx[3], 1e-5f);
  EXPECT_NEAR(0.2500f, dx[4], 1e-5f);
  EXPECT_NEAR(0.1500f, dx[5], 1e-5f);
  EXPECT_NEAR(0.0750f, dx[6], 1e-5f);
  EXPECT_NEAR(0.1750f, dx[7], 1e-5f);
  EXPECT_NEAR(0.1000f, dx[8], 1e-5f);
  EXPECT_NEAR(0.0250f, dx[9], 1e-5f);
  EXPECT_NEAR(0.0750f, dx[10], 1e-5f);
  EXPECT_NEAR(0.0500f, dx[11], 1e-5f);
  EXPECT_NEAR(0.1000f, dx[12], 1e-5f);
  EXPECT_NEAR(0.2500f, dx[13], 1e-5f);
  EXPECT_NEAR(0.1500f, dx[14], 1e-5f);
  EXPECT_NEAR(0.0750f, dx[15], 1e-5f);
  EXPECT_NEAR(0.1750f, dx[16], 1e-5f);
  EXPECT_NEAR(0.1000f, dx[17], 1e-5f);
}

#endif // USE_MKLDNN
