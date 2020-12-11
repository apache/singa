/*********************************************************
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
 ************************************************************/

#include "../src/model/layer/lrn.h"
#include "gtest/gtest.h"

using namespace singa;

TEST(LRN, Setup) {
  LRN lrn;
  // EXPECT_EQ("LRN", lrn.layer_type());

  LayerConf conf;
  LRNConf *lrn_conf = conf.mutable_lrn_conf();
  lrn_conf->set_k(1.0);
  lrn_conf->set_local_size(3);
  lrn_conf->set_alpha(0.1f);
  lrn_conf->set_beta(0.75f);
  lrn.Setup(Shape{1}, conf);

  EXPECT_FLOAT_EQ(1.0, lrn.k());
  EXPECT_EQ(3, lrn.local_size());
  EXPECT_FLOAT_EQ(0.1f, lrn.alpha());
  EXPECT_FLOAT_EQ(0.75f, lrn.beta());
}

TEST(LRN, Forward) {
  LRN lrn;
  const float x[] = {1, 2, 3, 4, 5, 6, 7, 8};
  Tensor in(Shape{2, 4, 1, 1});
  in.CopyDataFromHostPtr(x, 8);

  singa::LayerConf conf;
  singa::LRNConf *lrn_conf = conf.mutable_lrn_conf();
  lrn_conf->set_k(1.0);
  lrn_conf->set_local_size(3);
  lrn_conf->set_alpha(0.1f);
  lrn_conf->set_beta(0.75f);
  lrn.Setup(Shape{4, 1, 1}, conf);

  Tensor out = lrn.Forward(kTrain, in);
  const float *outptr = out.data<float>();
  const auto &shape = out.shape();
  EXPECT_EQ(4u, shape.size());
  EXPECT_EQ(2u, shape[0]);
  EXPECT_EQ(4u, shape[1]);
  EXPECT_EQ(1u, shape[2]);
  EXPECT_EQ(1u, shape[3]);

  EXPECT_NEAR(0.737787, outptr[0], 1e-6f);
  EXPECT_NEAR(1.037221, outptr[1], 1e-6f);
  EXPECT_NEAR(1.080992, outptr[2], 1e-6f);
  EXPECT_NEAR(1.563179, outptr[3], 1e-6f);
  EXPECT_NEAR(1.149545, outptr[4], 1e-6f);
  EXPECT_NEAR(0.930604, outptr[5], 1e-6f);
  EXPECT_NEAR(0.879124, outptr[6], 1e-6f);
  EXPECT_NEAR(1.218038, outptr[7], 1e-6f);
}

TEST(LRN, Backward) {
  LRN lrn;
  const float x[] = {1, 2, 3, 4, 5, 6, 7, 8};
  Tensor in(Shape{2, 4, 1, 1});
  in.CopyDataFromHostPtr(x, 8);

  singa::LayerConf conf;
  singa::LRNConf *lrn_conf = conf.mutable_lrn_conf();
  lrn_conf->set_k(1.0);
  lrn_conf->set_local_size(3);
  lrn_conf->set_alpha(0.1f);
  lrn_conf->set_beta(0.75f);
  lrn.Setup(Shape{4, 1, 1}, conf);

  Tensor out = lrn.Forward(kTrain, in);

  const float dy_arr[] = {8, 7, 6, 5, 4, 3, 2, 1};
  Tensor dy(Shape{2, 4, 1, 1});
  dy.CopyDataFromHostPtr(dy_arr, 8);

  const auto ret = lrn.Backward(singa::kTrain, dy);
  singa::Tensor dx = ret.first;
  const float *dxptr = dx.data<float>();
  const auto &shape = dx.shape();
  EXPECT_EQ(4u, shape.size());
  EXPECT_EQ(2u, shape[0]);
  EXPECT_EQ(4u, shape[1]);
  EXPECT_EQ(1u, shape[2]);
  EXPECT_EQ(1u, shape[3]);

  EXPECT_NEAR(4.858288752f, dxptr[0], 1e-6f);
  EXPECT_NEAR(1.04332631f, dxptr[1], 1e-6f);
  EXPECT_NEAR(-0.952648779f, dxptr[2], 1e-6f);
  EXPECT_NEAR(-0.38373312f, dxptr[3], 1e-6f);
  EXPECT_NEAR(0.259424615f, dxptr[4], 1e-6f);
  EXPECT_NEAR(-0.426475393f, dxptr[5], 1e-6f);
  EXPECT_NEAR(-0.213195118f, dxptr[6], 1e-6f);
  EXPECT_NEAR(-0.099276183f, dxptr[7], 1e-6f);
}
