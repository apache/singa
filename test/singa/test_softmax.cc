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

#include <math.h>  // exp

#include "../src/model/layer/softmax.h"
#include "gtest/gtest.h"

using singa::Shape;
using singa::Softmax;
TEST(Softmax, Setup) {
  Softmax sft;
  // EXPECT_EQ("Softmax", sft.layer_type());

  singa::LayerConf conf;
  sft.Setup(Shape{3}, conf);
}

#ifdef USE_CBLAS
TEST(Softmax, Forward) {
  const float x[] = {1.0f, 2.0f, 0.0f, -2.0f, -3.0f, -1.0};
  size_t row = 2;
  size_t col = 3;
  size_t n = row * col;
  singa::Tensor in(singa::Shape{row, col});
  in.CopyDataFromHostPtr<float>(x, row * col);

  Softmax sft;
  singa::LayerConf conf;
  sft.Setup(Shape{col}, conf);

  singa::Tensor out = sft.Forward(singa::kTrain, in);
  const float* yptr = out.data<float>();
  EXPECT_EQ(n, out.Size());

  float* sigma = new float[row];
  for (size_t i = 0; i < row; i++) sigma[i] = 0.f;
  for (size_t i = 0; i < n; i++) sigma[i / col] += exp(x[i]);
  // EXPECT_EQ(0, sigma[1]);
  for (size_t i = 0; i < row; i++)
    for (size_t j = 0; j < col; j++) {
      EXPECT_FLOAT_EQ(yptr[i * col + j], exp(x[i * col + j]) / sigma[i]);
    }
  delete[] sigma;
}

TEST(Softmax, Backward) {
  const float x[] = {1.0f, 2.0f, 0.0f, -2.0f, -3.0f, -1.0};
  size_t n = sizeof(x) / sizeof(float);
  size_t row = 2;
  size_t col = 3;
  singa::Tensor in(singa::Shape{row, col});
  in.CopyDataFromHostPtr<float>(x, n);

  Softmax sft;
  singa::LayerConf conf;
  sft.Setup(Shape{col}, conf);
  singa::Tensor out = sft.Forward(singa::kTrain, in);
  const float* yptr = out.data<float>();

  const float grad[] = {2.0f, -3.0f, 1.0f, 3.0f, -1.0f, -2.0};
  singa::Tensor out_diff(singa::Shape{row, col});
  out_diff.CopyDataFromHostPtr<float>(grad, n);
  const auto in_diff = sft.Backward(singa::kTrain, out_diff);
  const float* xptr = in_diff.first.data<float>();

  float* dx = new float[n];
  float* sigma = new float[row];
  for (size_t i = 0; i < row; i++) sigma[i] = 0.f;
  for (size_t i = 0; i < n; i++) sigma[i / col] += grad[i] * yptr[i];
  // EXPECT_EQ(0, sigma[0]);
  // EXPECT_EQ(0, sigma[1]);
  for (size_t i = 0; i < row; i++)
    for (size_t j = 0; j < col; j++)
      dx[i * col + j] = (grad[i * col + j] - sigma[i]) * yptr[i * col + j];
  EXPECT_FLOAT_EQ(dx[0], xptr[0]);
  EXPECT_FLOAT_EQ(dx[4], xptr[4]);
  EXPECT_FLOAT_EQ(dx[5], xptr[5]);
  delete[] dx;
  delete[] sigma;
}
#endif
