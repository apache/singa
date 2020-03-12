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

#include "../src/model/layer/cudnn_lrn.h"

#ifdef USE_CUDNN
// cudnn lrn is added in cudnn 4
#if CUDNN_MAJOR >= 4
#include "gtest/gtest.h"

using singa::CudnnLRN;
using singa::Shape;
TEST(CudnnLRN, Setup) {
  CudnnLRN lrn;
  // EXPECT_EQ("CudnnLRN", lrn.layer_type());

  singa::LayerConf conf;
  singa::LRNConf *lrn_conf = conf.mutable_lrn_conf();
  lrn_conf->set_k(1.0);
  lrn_conf->set_local_size(3);
  lrn_conf->set_alpha(0.1);
  lrn_conf->set_beta(0.75);
  lrn.Setup(Shape{1}, conf);

  EXPECT_FLOAT_EQ(1.0, lrn.k());
  EXPECT_EQ(3, lrn.local_size());
  EXPECT_FLOAT_EQ(0.1, lrn.alpha());
  EXPECT_FLOAT_EQ(0.75, lrn.beta());
}

TEST(CudnnLRN, Forward) {
  CudnnLRN lrn;
  const float x[] = {
      0.00658502,  -0.0496967,  -0.0333733, -0.0263094, -0.044298,  0.0211638,
      0.0829358,   -0.0172312,  -0.0665471, -0.10017,   -0.0750333, -0.104551,
      -0.00981208, -0.0583349,  -0.0751652, 0.011747,   0.0151165,  0.0304321,
      0.0736639,   -0.00652653, 0.00962833, 0.169646,   -0.044588,  -0.00244141,
      0.0597329,   -0.0530868,  0.0124246,  0.108429,   0.0451175,  0.0247055,
      0.0304345,   0.0179575};
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(singa::Shape{1, 2, 4, 4}, cuda);
  in.CopyDataFromHostPtr(x, 1 * 2 * 4 * 4);

  singa::LayerConf conf;
  singa::LRNConf *lrn_conf = conf.mutable_lrn_conf();
  lrn_conf->set_k(1.0);
  lrn_conf->set_local_size(3);
  lrn_conf->set_alpha(0.1);
  lrn_conf->set_beta(0.75);
  lrn.Setup(Shape{2, 4, 4}, conf);

  singa::Tensor out = lrn.Forward(singa::kTrain, in);
  out.ToHost();
  const float *outptr = out.data<float>();
  const auto &shape = out.shape();
  EXPECT_EQ(4u, shape.size());
  EXPECT_EQ(1u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  EXPECT_EQ(4u, shape[2]);
  EXPECT_EQ(4u, shape[3]);

  EXPECT_NEAR(0.00658498f, outptr[0], 1e-6f);
  EXPECT_NEAR(-0.0496925f, outptr[1], 1e-6f);
  EXPECT_NEAR(-0.0333678f, outptr[2], 1e-6f);
  EXPECT_NEAR(-0.0263089f, outptr[3], 1e-6f);
  EXPECT_NEAR(-0.0442958f, outptr[4], 1e-6f);
  EXPECT_NEAR(0.0211483f, outptr[5], 1e-6f);
  EXPECT_NEAR(0.0829174f, outptr[6], 1e-6f);
  EXPECT_NEAR(-0.0172311f, outptr[7], 1e-6f);
  EXPECT_NEAR(-0.0665338f, outptr[8], 1e-6f);
  EXPECT_NEAR(-0.100138f, outptr[9], 1e-6f);
  EXPECT_NEAR(-0.0750224f, outptr[10], 1e-6f);
  EXPECT_NEAR(-0.104492f, outptr[11], 1e-6f);
  EXPECT_NEAR(-0.00981155f, outptr[12], 1e-6f);
  EXPECT_NEAR(-0.058329f, outptr[13], 1e-6f);
  EXPECT_NEAR(-0.0751528f, outptr[14], 1e-6f);
  EXPECT_NEAR(0.0117468f, outptr[15], 1e-6f);
  EXPECT_NEAR(0.0151164f, outptr[16], 1e-6f);
  EXPECT_NEAR(0.0304296f, outptr[17], 1e-6f);
  EXPECT_NEAR(0.0736518f, outptr[18], 1e-6f);
  EXPECT_NEAR(-0.00652641f, outptr[19], 1e-6f);
  EXPECT_NEAR(0.00962783f, outptr[20], 1e-6f);
  EXPECT_NEAR(0.169522f, outptr[21], 1e-6f);
  EXPECT_NEAR(-0.0445781f, outptr[22], 1e-6f);
  EXPECT_NEAR(-0.00244139f, outptr[23], 1e-6f);
  EXPECT_NEAR(0.0597209f, outptr[24], 1e-6f);
  EXPECT_NEAR(-0.0530697f, outptr[25], 1e-6f);
  EXPECT_NEAR(0.0124228f, outptr[26], 1e-6f);
  EXPECT_NEAR(0.108367f, outptr[27], 1e-6f);
  EXPECT_NEAR(0.045115f, outptr[28], 1e-6f);
  EXPECT_NEAR(0.024703f, outptr[29], 1e-6f);
  EXPECT_NEAR(0.0304295f, outptr[30], 1e-6f);
  EXPECT_NEAR(0.0179573f, outptr[31], 1e-6f);
}

TEST(CudnnLRN, Backward) {
  CudnnLRN lrn;

  const float x[] = {
      0.00658502,  -0.0496967,  -0.0333733, -0.0263094, -0.044298,  0.0211638,
      0.0829358,   -0.0172312,  -0.0665471, -0.10017,   -0.0750333, -0.104551,
      -0.00981208, -0.0583349,  -0.0751652, 0.011747,   0.0151165,  0.0304321,
      0.0736639,   -0.00652653, 0.00962833, 0.169646,   -0.044588,  -0.00244141,
      0.0597329,   -0.0530868,  0.0124246,  0.108429,   0.0451175,  0.0247055,
      0.0304345,   0.0179575};
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor x_tensor(singa::Shape{1, 2, 4, 4}, cuda);
  x_tensor.CopyDataFromHostPtr(x, 1 * 2 * 4 * 4);

  const float dy[] = {
      -0.103178,   -0.0326904, 0.293932,   0.355288,   -0.0288079, -0.0543308,
      -0.0668226,  0.0462216,  -0.0448064, -0.068982,  -0.0509133, -0.0721143,
      0.0959078,   -0.0389037, -0.0510071, -0.178793,  0.00428248, -0.001132,
      -0.19928,    0.011935,   0.00622313, 0.143793,   0.0253894,  0.0104906,
      -0.170673,   0.0283919,  0.00523488, -0.0455003, 0.177807,   0.000892812,
      -0.00113197, 0.00327798};

  singa::Tensor dy_tensor(singa::Shape{1, 2, 4, 4}, cuda);
  dy_tensor.CopyDataFromHostPtr(dy, 1 * 2 * 4 * 4);

  singa::LayerConf conf;
  singa::LRNConf *lrn_conf = conf.mutable_lrn_conf();
  lrn_conf->set_k(1.0);
  lrn_conf->set_local_size(3);
  lrn_conf->set_alpha(0.1);
  lrn_conf->set_beta(0.75);
  lrn.Setup(Shape{2, 4, 4}, conf);

  lrn.Forward(singa::kTrain, x_tensor);
  const auto ret = lrn.Backward(singa::kTrain, dy_tensor);
  singa::Tensor dx = ret.first;
  dx.ToHost();
  const float *dxptr = dx.data<float>();
  const auto &shape = dx.shape();
  EXPECT_EQ(4u, shape.size());
  EXPECT_EQ(1u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  EXPECT_EQ(4u, shape[2]);
  EXPECT_EQ(4u, shape[3]);

  EXPECT_NEAR(-0.103177, dxptr[0], 1e-6f);
  EXPECT_NEAR(-0.0326837, dxptr[1], 1e-6f);
  EXPECT_NEAR(0.293844, dxptr[2], 1e-6f);
  EXPECT_NEAR(0.355269, dxptr[3], 1e-6f);
  EXPECT_NEAR(-0.0288034, dxptr[4], 1e-6f);
  EXPECT_NEAR(-0.0543157, dxptr[5], 1e-6f);
  EXPECT_NEAR(-0.0667802, dxptr[6], 1e-6f);
  EXPECT_NEAR(0.0462206, dxptr[7], 1e-6f);
  EXPECT_NEAR(-0.0448215, dxptr[8], 1e-6f);
  EXPECT_NEAR(-0.0689328, dxptr[9], 1e-6f);
  EXPECT_NEAR(-0.0508914, dxptr[10], 1e-6f);
  EXPECT_NEAR(-0.0720598, dxptr[11], 1e-6f);
  EXPECT_NEAR(0.0959062, dxptr[12], 1e-6f);
  EXPECT_NEAR(-0.0388931, dxptr[13], 1e-6f);
  EXPECT_NEAR(-0.0509844, dxptr[14], 1e-6f);
  EXPECT_NEAR(-0.17879, dxptr[15], 1e-6f);
  EXPECT_NEAR(0.00428292, dxptr[16], 1e-6f);
  EXPECT_NEAR(-0.00113432, dxptr[17], 1e-6f);
  EXPECT_NEAR(-0.199158, dxptr[18], 1e-6f);
  EXPECT_NEAR(0.0119317, dxptr[19], 1e-6f);
  EXPECT_NEAR(0.00622216, dxptr[20], 1e-6f);
  EXPECT_NEAR(0.143491, dxptr[21], 1e-6f);
  EXPECT_NEAR(0.0253689, dxptr[22], 1e-6f);
  EXPECT_NEAR(0.0104904, dxptr[23], 1e-6f);
  EXPECT_NEAR(-0.170617, dxptr[24], 1e-6f);
  EXPECT_NEAR(0.0283971, dxptr[25], 1e-6f);
  EXPECT_NEAR(0.00523171, dxptr[26], 1e-6f);
  EXPECT_NEAR(-0.0454887, dxptr[27], 1e-6f);
  EXPECT_NEAR(0.177781, dxptr[28], 1e-6f);
  EXPECT_NEAR(0.000889893, dxptr[29], 1e-6f);
  EXPECT_NEAR(-0.00113756, dxptr[30], 1e-6f);
  EXPECT_NEAR(0.00327978, dxptr[31], 1e-6f);
}

#endif  //  CUDNN_MAJOR >= 4
#endif  //  USE_CUDNN
