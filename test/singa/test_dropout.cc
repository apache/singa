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

#include "../src/model/layer/dropout.h"
#include "gtest/gtest.h"

using singa::Dropout;
using singa::Shape;
TEST(Dropout, Setup) {
  Dropout drop;
  // EXPECT_EQ("Dropout", drop.layer_type());

  singa::LayerConf conf;
  singa::DropoutConf* dropconf = conf.mutable_dropout_conf();
  dropconf->set_dropout_ratio(0.8f);

  drop.Setup(Shape{3}, conf);
  EXPECT_EQ(0.8f, drop.dropout_ratio());
}

TEST(Dropout, Forward) {
  const float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  size_t n = sizeof(x) / sizeof(float);
  singa::Tensor in(singa::Shape{n});
  in.CopyDataFromHostPtr(x, n);

  float pdrop = 0.5;
  Dropout drop;
  singa::LayerConf conf;
  singa::DropoutConf* dropconf = conf.mutable_dropout_conf();
  dropconf->set_dropout_ratio(pdrop);
  drop.Setup(Shape{1}, conf);
  float scale = 1.0f / (1.0f - pdrop);

  singa::Tensor out1 = drop.Forward(singa::kTrain, in);

  const float* mptr = drop.mask().data<float>();
  for (size_t i = 0; i < n; i++)
    EXPECT_FLOAT_EQ(0, mptr[i] * (mptr[i] - scale));

  const float* outptr1 = out1.data<float>();
  EXPECT_EQ(n, out1.Size());
  // the output value should be 0 or the same as the input
  EXPECT_EQ(0.f, outptr1[0] * (outptr1[0] - scale * x[0]));
  EXPECT_EQ(0.f, outptr1[1] * (outptr1[1] - scale * x[1]));
  EXPECT_EQ(0.f, outptr1[7] * (outptr1[7] - scale * x[7]));

  singa::Tensor out2 = drop.Forward(singa::kEval, in);
  EXPECT_EQ(n, out2.Size());
  const float* outptr2 = out2.data<float>();
  // the output value should be the same as the input
  EXPECT_EQ(x[0], outptr2[0]);
  EXPECT_EQ(x[1], outptr2[1]);
  EXPECT_EQ(x[7], outptr2[7]);
}

TEST(Dropout, Backward) {
  const float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  size_t n = sizeof(x) / sizeof(float);
  singa::Tensor in(singa::Shape{n});
  in.CopyDataFromHostPtr(x, n);

  float pdrop = 0.5;
  float scale = 1.0f / (1.0f - pdrop);

  Dropout drop;
  singa::LayerConf conf;
  singa::DropoutConf* dropconf = conf.mutable_dropout_conf();
  dropconf->set_dropout_ratio(pdrop);
  drop.Setup(Shape{1}, conf);
  singa::Tensor out1 = drop.Forward(singa::kTrain, in);

  const float dy[] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f};
  singa::Tensor grad(singa::Shape{n});
  grad.CopyDataFromHostPtr(dy, n);

  const float* mptr = drop.mask().data<float>();
  const auto ret = drop.Backward(singa::kTrain, grad);
  const float* dx = ret.first.data<float>();
  EXPECT_FLOAT_EQ(dx[0], dy[0] * (mptr[0] > 0 ? 1.0f : 0.0f) * scale);
  EXPECT_FLOAT_EQ(dx[1], dy[1] * (mptr[1] > 0) * scale);
  EXPECT_FLOAT_EQ(dx[7], dy[7] * (mptr[7] > 0) * scale);
}
