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

#include "../src/model/layer/cudnn_batchnorm.h"

#ifdef USE_CUDNN
#include "gtest/gtest.h"

using singa::CudnnBatchNorm;
using singa::Shape;
TEST(CudnnBatchNorm, Setup) {
  CudnnBatchNorm batchnorm;
  // EXPECT_EQ("CudnnBatchNorm", batchnorm.layer_type());

  singa::LayerConf conf;
  singa::BatchNormConf *batchnorm_conf = conf.mutable_batchnorm_conf();
  batchnorm_conf->set_factor(0.01);
  batchnorm.Setup(Shape{2, 4, 4}, conf);

  EXPECT_FLOAT_EQ(0.01, batchnorm.factor());
  EXPECT_EQ(2u, batchnorm.channels());
  EXPECT_EQ(4u, batchnorm.height());
  EXPECT_EQ(4u, batchnorm.width());
}

TEST(CudnnBatchNorm, Forward) {
  CudnnBatchNorm batchnorm;
  const float x[] = {0.0736655,  0.0459045,  0.0779517,  0.0771059,  0.0586862,
                     0.0561263,  0.0708457,  0.0977273,  0.0405025,  -0.170897,
                     0.0208982,  0.136865,   -0.0367905, -0.0618205, -0.0103908,
                     -0.0522777, -0.122161,  -0.025427,  -0.0718576, -0.185941,
                     0.0166533,  0.178679,   -0.0576606, -0.137817,  0.150676,
                     0.153442,   -0.0929899, -0.148675,  -0.112459,  -0.106284,
                     -0.103074,  -0.0668811};
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(singa::Shape{1, 2, 4, 4}, cuda);
  in.CopyDataFromHostPtr(x, 1 * 2 * 4 * 4);
  const float alpha_[] = {1, 1};
  singa::Tensor alpha(singa::Shape{1, 2, 1, 1}, cuda);
  alpha.CopyDataFromHostPtr(alpha_, 1 * 2 * 1 * 1);

  const float beta_[] = {0, 0};
  singa::Tensor beta(singa::Shape{1, 2, 1, 1}, cuda);
  beta.CopyDataFromHostPtr(beta_, 1 * 2 * 1 * 1);

  singa::LayerConf conf;
  singa::BatchNormConf *batchnorm_conf = conf.mutable_batchnorm_conf();
  batchnorm_conf->set_factor(0.9);
  batchnorm.Setup(Shape{2, 4, 4}, conf);

  batchnorm.ToDevice(cuda);
  batchnorm.set_bnScale(alpha);
  batchnorm.set_bnBias(beta);
  batchnorm.set_runningMean(beta);
  batchnorm.set_runningVariance(beta);
  singa::Tensor out = batchnorm.Forward(singa::kTrain, in);
  out.ToHost();
  const float *outptr = out.data<float>();
  const auto &shape = out.shape();
  EXPECT_EQ(4u, shape.size());
  EXPECT_EQ(1u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  EXPECT_EQ(4u, shape[2]);
  EXPECT_EQ(4u, shape[3]);
  EXPECT_NEAR(0.637092, outptr[0], 1e-4f);
  EXPECT_NEAR(0.262057, outptr[1], 1e-4f);
  EXPECT_NEAR(0.694995, outptr[2], 1e-4f);
  EXPECT_NEAR(0.683569, outptr[3], 1e-4f);
  EXPECT_NEAR(0.43473, outptr[4], 1e-4f);
  EXPECT_NEAR(0.400147, outptr[5], 1e-4f);
  EXPECT_NEAR(0.598998, outptr[6], 1e-4f);
  EXPECT_NEAR(0.962152, outptr[7], 1e-4f);
  EXPECT_NEAR(0.189079, outptr[8], 1e-4f);
  EXPECT_NEAR(-2.6668, outptr[9], 1e-4f);
  EXPECT_NEAR(-0.0757632, outptr[10], 1e-4f);
  EXPECT_NEAR(1.49088, outptr[11], 1e-4f);
  EXPECT_NEAR(-0.855104, outptr[12], 1e-4f);
  EXPECT_NEAR(-1.19324, outptr[13], 1e-4f);
  EXPECT_NEAR(-0.498459, outptr[14], 1e-4f);
  EXPECT_NEAR(-1.06433, outptr[15], 1e-4f);
  EXPECT_NEAR(-0.696646, outptr[16], 1e-4f);
  EXPECT_NEAR(0.185125, outptr[17], 1e-4f);
  EXPECT_NEAR(-0.238109, outptr[18], 1e-4f);
  EXPECT_NEAR(-1.27803, outptr[19], 1e-4f);
  EXPECT_NEAR(0.568704, outptr[20], 1e-4f);
  EXPECT_NEAR(2.04564, outptr[21], 1e-4f);
  EXPECT_NEAR(-0.108697, outptr[22], 1e-4f);
  EXPECT_NEAR(-0.839356, outptr[23], 1e-4f);
  EXPECT_NEAR(1.79038, outptr[24], 1e-4f);
  EXPECT_NEAR(1.81559, outptr[25], 1e-4f);
  EXPECT_NEAR(-0.430738, outptr[26], 1e-4f);
  EXPECT_NEAR(-0.938335, outptr[27], 1e-4f);
  EXPECT_NEAR(-0.608203, outptr[28], 1e-4f);
  EXPECT_NEAR(-0.551921, outptr[29], 1e-4f);
  EXPECT_NEAR(-0.522658, outptr[30], 1e-4f);
  EXPECT_NEAR(-0.192746, outptr[31], 1e-4f);
}

TEST(CudnnBatchNorm, Backward) {
  CudnnBatchNorm batchnorm;
  const float x[] = {0.0736655,  0.0459045,  0.0779517,  0.0771059,  0.0586862,
                     0.0561263,  0.0708457,  0.0977273,  0.0405025,  -0.170897,
                     0.0208982,  0.136865,   -0.0367905, -0.0618205, -0.0103908,
                     -0.0522777, -0.122161,  -0.025427,  -0.0718576, -0.185941,
                     0.0166533,  0.178679,   -0.0576606, -0.137817,  0.150676,
                     0.153442,   -0.0929899, -0.148675,  -0.112459,  -0.106284,
                     -0.103074,  -0.0668811};
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor x_tensor(singa::Shape{1, 2, 4, 4}, cuda);
  x_tensor.CopyDataFromHostPtr(x, 1 * 2 * 4 * 4);

  singa::LayerConf conf;
  singa::BatchNormConf *batchnorm_conf = conf.mutable_batchnorm_conf();
  batchnorm_conf->set_factor(1);
  batchnorm.Setup(Shape{2, 4, 4}, conf);

  const float dy[] = {
      -0.0064714,  0,           0,          0,          0,          -0.00297655,
      -0.0195729,  0,           0,          0,          0,          0,
      0,           0,           0,          -0.0032594, 0,          0,
      0,           0,           0,          0,          0.0125562,  0,
      0.00041933,  0.000386108, -0.0074611, 0.0015929,  0.00468428, 0.00735506,
      -0.00682525, 0.00342023};

  singa::Tensor dy_tensor(singa::Shape{1, 2, 4, 4}, cuda);
  dy_tensor.CopyDataFromHostPtr(dy, 1 * 2 * 4 * 4);
  const float alpha_[] = {1, 1};
  singa::Tensor alpha(singa::Shape{2}, cuda);
  alpha.CopyDataFromHostPtr(alpha_, 1 * 2 * 1 * 1);

  const float beta_[] = {0, 0};
  singa::Tensor beta(singa::Shape{2}, cuda);
  beta.CopyDataFromHostPtr(beta_, 1 * 2 * 1 * 1);

  const float mean_[] = {0.0123405, -0.0622333};
  singa::Tensor mean(singa::Shape{2}, cuda);
  mean.CopyDataFromHostPtr(mean_, 1 * 2 * 1 * 1);

  const float var_[] = {15.9948, 8.68198};
  singa::Tensor var(singa::Shape{2}, cuda);
  var.CopyDataFromHostPtr(var_, 1 * 2 * 1 * 1);

  batchnorm.ToDevice(cuda);
  batchnorm.set_bnScale(alpha);
  batchnorm.set_bnBias(beta);
  batchnorm.set_runningMean(beta);
  batchnorm.set_runningVariance(beta);
  batchnorm.Forward(singa::kTrain, x_tensor);
  const auto ret = batchnorm.Backward(singa::kTrain, dy_tensor);
  singa::Tensor dx = ret.first;
  dx.ToHost();
  const float *dxptr = dx.data<float>();
  const auto &shape = dx.shape();
  EXPECT_EQ(4u, shape.size());
  EXPECT_EQ(1u, shape[0]);
  EXPECT_EQ(2u, shape[1]);
  EXPECT_EQ(4u, shape[2]);
  EXPECT_EQ(4u, shape[3]);
  EXPECT_NEAR(-0.0528703, dxptr[0], 1e-4f);
  EXPECT_NEAR(0.0302578, dxptr[1], 1e-4f);
  EXPECT_NEAR(0.0352178, dxptr[2], 1e-4f);
  EXPECT_NEAR(0.0350869, dxptr[3], 1e-4f);
  EXPECT_NEAR(0.032236, dxptr[4], 1e-4f);
  EXPECT_NEAR(-0.00837157, dxptr[5], 1e-4f);
  EXPECT_NEAR(-0.2303, dxptr[6], 1e-4f);
  EXPECT_NEAR(0.0382786, dxptr[7], 1e-4f);
  EXPECT_NEAR(0.0294217, dxptr[8], 1e-4f);
  EXPECT_NEAR(-0.00329757, dxptr[9], 1e-4f);
  EXPECT_NEAR(0.0263874, dxptr[10], 1e-4f);
  EXPECT_NEAR(0.0443361, dxptr[11], 1e-4f);
  EXPECT_NEAR(0.0174587, dxptr[12], 1e-4f);
  EXPECT_NEAR(0.0135847, dxptr[13], 1e-4f);
  EXPECT_NEAR(0.0215447, dxptr[14], 1e-4f);
  EXPECT_NEAR(-0.0289709, dxptr[15], 1e-4f);
  EXPECT_NEAR(-0.0100591, dxptr[16], 1e-4f);
  EXPECT_NEAR(-0.00895677, dxptr[17], 1e-4f);
  EXPECT_NEAR(-0.00948587, dxptr[18], 1e-4f);
  EXPECT_NEAR(-0.0107859, dxptr[19], 1e-4f);
  EXPECT_NEAR(-0.00847725, dxptr[20], 1e-4f);
  EXPECT_NEAR(-0.0066309, dxptr[21], 1e-4f);
  EXPECT_NEAR(0.105131, dxptr[22], 1e-4f);
  EXPECT_NEAR(-0.0102375, dxptr[23], 1e-4f);
  EXPECT_NEAR(-0.00312763, dxptr[24], 1e-4f);
  EXPECT_NEAR(-0.00339895, dxptr[25], 1e-4f);
  EXPECT_NEAR(-0.0777377, dxptr[26], 1e-4f);
  EXPECT_NEAR(0.00415871, dxptr[27], 1e-4f);
  EXPECT_NEAR(0.0327506, dxptr[28], 1e-4f);
  EXPECT_NEAR(0.0571663, dxptr[29], 1e-4f);
  EXPECT_NEAR(-0.0720566, dxptr[30], 1e-4f);
  EXPECT_NEAR(0.0217477, dxptr[31], 1e-4f);

  singa::Tensor dbnScale = ret.second.at(0);
  dbnScale.ToHost();
  const float *dbnScaleptr = dbnScale.data<float>();
  const auto &dbnScaleShape = dbnScale.shape();
  EXPECT_EQ(1u, dbnScaleShape.size());
  EXPECT_EQ(2u, dbnScaleShape[0]);

  EXPECT_NEAR(-0.013569f, dbnScaleptr[0], 1e-4f);
  EXPECT_NEAR(-0.00219431f, dbnScaleptr[1], 1e-4f);

  singa::Tensor dbnBias = ret.second.at(1);
  dbnBias.ToHost();
  const float *dbnBiasptr = dbnBias.data<float>();
  const auto &dbnBiasShape = dbnBias.shape();
  EXPECT_EQ(1u, dbnBiasShape.size());
  EXPECT_EQ(2u, dbnBiasShape[0]);

  EXPECT_NEAR(-0.0322803f, dbnBiasptr[0], 1e-4f);
  EXPECT_NEAR(0.0161278f, dbnBiasptr[1], 1e-4f);
}

#endif  //  USE_CUDNN
