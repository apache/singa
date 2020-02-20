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

#ifdef USE_DNNL
TEST(DNNLOperationPooling, Forward) {
  const size_t batchsize = 2, c = 1, h = 3, w = 3;
  const float x[batchsize * c * h * w] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                          7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  Tensor in(Shape{batchsize, c, h, w});
  in.CopyDataFromHostPtr(x, batchsize * c * h * w);

  PoolingHandle pool_handle(in, {2, 2}, {1, 1}, {0, 0}, true);
  Tensor out1 = CpuPoolingForward(pool_handle, in);
}
TEST(DNNLOperationPooling, ForwardAverage) {
  const size_t batchsize = 2, c = 1, h = 3, w = 3;
  const float x[batchsize * c * h * w] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,

      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  Tensor in(Shape{batchsize, c, h, w});
  in.CopyDataFromHostPtr(x, batchsize * c * h * w);

  PoolingHandle pool_handle(in, {2, 2}, {1, 1}, {0, 0}, false);
  Tensor out1 = CpuPoolingForward(pool_handle, in);
}

TEST(DNNLOperationPooling, Backward) {
  // src_data
  const size_t batchsize = 2, c = 1, src_h = 3, src_w = 3;
  const float x[batchsize * c * src_h * src_w] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  Tensor in(Shape{batchsize, c, src_h, src_w});
  in.CopyDataFromHostPtr(x, batchsize * c * src_h * src_w);

  PoolingHandle pool_handle(in, {2, 2}, {1, 1}, {0, 0}, true);

  Tensor out = CpuPoolingForward(pool_handle, in);

  // grad - bwd
  const size_t grad_h = 2, grad_w = 2;
  const float dy[batchsize * c * grad_h * grad_w] = {0.1f, 0.2f, 0.3f, 0.4f,
                                                     0.1f, 0.2f, 0.3f, 0.4f};
  Tensor grad(Shape{batchsize, c, grad_h, grad_w});
  grad.CopyDataFromHostPtr(dy, batchsize * c * grad_h * grad_w);

  Tensor in_grad = CpuPoolingBackward(pool_handle, grad, in, out);
}
TEST(DNNLOperationPooling, BackwardAvg) {
  // src_data
  const size_t batchsize = 2, c = 1, src_h = 3, src_w = 3;
  const float x[batchsize * c * src_h * src_w] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,

      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  Tensor in(Shape{batchsize, c, src_h, src_w});
  in.CopyDataFromHostPtr(x, batchsize * c * src_h * src_w);

  PoolingHandle pool_handle(in, {2, 2}, {1, 1}, {0, 0}, false);

  Tensor out = CpuPoolingForward(pool_handle, in);

  // grad - bwd
  const size_t grad_h = 2, grad_w = 2;
  const float dy[batchsize * c * grad_h * grad_w] = {0.1f, 0.2f, 0.3f, 0.4f,
                                                     0.1f, 0.2f, 0.3f, 0.4f};
  Tensor grad(Shape{batchsize, c, grad_h, grad_w});
  grad.CopyDataFromHostPtr(dy, batchsize * c * grad_h * grad_w);

  Tensor in_grad = CpuPoolingBackward(pool_handle, grad, in, out);
}

#endif  // USE_DNNL
