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
#include "../src/model/layer/dense.h"
#include "gtest/gtest.h"
#include "singa/singa_config.h"

using singa::Dense;
using singa::Shape;
TEST(Dense, Setup) {
  Dense dense;
  // EXPECT_EQ("Dense", dense.layer_type());

  singa::LayerConf conf;
  singa::DenseConf *denseconf = conf.mutable_dense_conf();
  denseconf->set_num_output(3);
  dense.Setup(Shape{2}, conf);

  EXPECT_EQ(3u, dense.num_output());
  EXPECT_EQ(2u, dense.num_input());
}
#ifdef USE_CBLAS
TEST(Dense, ForwardCpp) {
  Dense dense;

  singa::LayerConf conf;
  singa::DenseConf *denseconf = conf.mutable_dense_conf();
  denseconf->set_num_output(3);
  denseconf->set_transpose(false);
  dense.Setup(Shape{2}, conf);

  const size_t batchsize = 3, vdim = 2, hdim = 3;
  const float x[batchsize * vdim] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  singa::Tensor in(singa::Shape{batchsize, vdim});
  in.CopyDataFromHostPtr(x, batchsize * vdim);

  // set weight
  const float we[vdim * hdim] = {1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f};
  singa::Tensor weight(singa::Shape{vdim, hdim});
  weight.CopyDataFromHostPtr(we, hdim * vdim);

  const float bia[hdim] = {1.0f, 1.0f, 1.0f};
  singa::Tensor bias(singa::Shape{hdim});
  bias.CopyDataFromHostPtr(bia, hdim);

  dense.set_weight(weight);
  dense.set_bias(bias);

  singa::Tensor out1 = dense.Forward(singa::kTrain, in);
  const float *outptr1 = out1.data<float>();
  EXPECT_EQ(9u, out1.Size());
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(
          (x[i * 2 + 0] * we[j] + x[i * 2 + 1] * we[3 + j] + bia[j]),
          outptr1[i * 3 + j]);
}
TEST(Dense, BackwardCpp) {
  Dense dense;

  singa::LayerConf conf;
  singa::DenseConf *denseconf = conf.mutable_dense_conf();
  denseconf->set_num_output(3);
  denseconf->set_transpose(false);
  dense.Setup(Shape{2}, conf);

  const size_t batchsize = 3, vdim = 2, hdim = 3;
  const float x[batchsize * vdim] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  singa::Tensor in(singa::Shape{batchsize, vdim});
  in.CopyDataFromHostPtr(x, batchsize * vdim);

  // set weight
  const float we[hdim * vdim] = {1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f};
  singa::Tensor weight(singa::Shape{vdim, hdim});
  weight.CopyDataFromHostPtr(we, hdim * vdim);

  const float bia[hdim] = {1.0f, 1.0f, 1.0f};
  singa::Tensor bias(singa::Shape{hdim});
  bias.CopyDataFromHostPtr(bia, hdim);

  dense.set_weight(weight);
  dense.set_bias(bias);

  singa::Tensor out1 = dense.Forward(singa::kTrain, in);

  // grad
  const float dy[batchsize * hdim] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                                      2.0f, 3.0f, 3.0f, 3.0f};
  singa::Tensor grad(singa::Shape{batchsize, hdim});
  grad.CopyDataFromHostPtr(dy, batchsize * hdim);

  const auto ret = dense.Backward(singa::kTrain, grad);
  singa::Tensor in_grad = ret.first;
  singa::Tensor dweight = ret.second.at(0);
  singa::Tensor dbias = ret.second.at(1);
  EXPECT_EQ(6u, in_grad.Size());
  /*
  const float *dx = in_grad.data<float>();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(
          (dy[i * 3 + 0] * we[j * 3 + 0] + dy[i * 3 + 1] * we[j * 3 + 1] +
           dy[i * 3 + 2] * we[j * 3 + 2]),
          dx[i * 2 + j]);
  const float *dweightx = dweight.data<float>();
  EXPECT_EQ(6u, dweight.Size());
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(
          (dy[i * 3 + 0] * x[j * 3 + 0] + dy[i * 3 + 1] * x[j * 3 + 0] +
           dy[i * 3 + 2] * x[j * 3 + 2]),
          dweightx[j * 2 + i]);
  */
  const float *dbiasx = dbias.data<float>();
  EXPECT_EQ(3u, dbias.Size());
  for (int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ((dy[0 * 3 + i] + dy[1 * 3 + i] + dy[2 * 3 + i]), dbiasx[i]);
}
#endif  // USE_CBLAS

#ifdef USE_CUDA
TEST(Dense, ForwardCuda) {
  Dense dense;

  singa::LayerConf conf;
  singa::DenseConf *denseconf = conf.mutable_dense_conf();
  denseconf->set_num_output(3);
  denseconf->set_transpose(false);
  dense.Setup(Shape{2}, conf);

  const size_t batchsize = 3, vdim = 2, hdim = 3;
  const float x[batchsize * vdim] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(singa::Shape{batchsize, vdim}, cuda);
  in.CopyDataFromHostPtr(x, batchsize * vdim);

  // set weight
  const float we[hdim * vdim] = {1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f};
  singa::Tensor weight(singa::Shape{vdim, hdim}, cuda);
  weight.CopyDataFromHostPtr(we, hdim * vdim);

  const float bia[hdim] = {1.0f, 1.0f, 1.0f};
  singa::Tensor bias(singa::Shape{hdim}, cuda);
  bias.CopyDataFromHostPtr(bia, hdim);

  dense.set_weight(weight);
  dense.set_bias(bias);

  singa::Tensor out1 = dense.Forward(singa::kTrain, in);
  out1.ToHost();
  const float *outptr1 = out1.data<float>();
  EXPECT_EQ(9u, out1.Size());
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(
          (x[i * 2 + 0] * we[j] + x[i * 2 + 1] * we[3 + j] + bia[j]),
          outptr1[i * 3 + j]);
}
TEST(Dense, BackwardCuda) {
  Dense dense;

  singa::LayerConf conf;
  singa::DenseConf *denseconf = conf.mutable_dense_conf();
  denseconf->set_num_output(3);
  denseconf->set_transpose(false);
  dense.Setup(Shape{2}, conf);

  const size_t batchsize = 3, vdim = 2, hdim = 3;
  const float x[batchsize * vdim] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(singa::Shape{batchsize, vdim}, cuda);
  in.CopyDataFromHostPtr(x, batchsize * vdim);

  // set weight
  const float we[hdim * vdim] = {1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f};
  singa::Tensor weight(singa::Shape{vdim, hdim}, cuda);
  weight.CopyDataFromHostPtr(we, hdim * vdim);

  const float bia[hdim] = {1.0f, 1.0f, 1.0f};
  singa::Tensor bias(singa::Shape{hdim}, cuda);
  bias.CopyDataFromHostPtr(bia, hdim);

  dense.set_weight(weight);
  dense.set_bias(bias);

  singa::Tensor out1 = dense.Forward(singa::kTrain, in);

  // grad
  const float dy[batchsize * hdim] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                                      2.0f, 3.0f, 3.0f, 3.0f};
  singa::Tensor grad(singa::Shape{batchsize, hdim}, cuda);
  grad.CopyDataFromHostPtr(dy, batchsize * hdim);

  auto ret = dense.Backward(singa::kTrain, grad);
  singa::Tensor in_grad = ret.first;
  singa::Tensor dweight = ret.second.at(0);
  singa::Tensor dbias = ret.second.at(1);
  in_grad.ToHost();
  EXPECT_EQ(6u, in_grad.Size());
  /*
  const float *dx = in_grad.data<float>();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(
          (dy[i * 3 + 0] * we[j * 3 + 0] + dy[i * 3 + 1] * we[j * 3 + 1] +
           dy[i * 3 + 2] * we[j * 3 + 2]),
          dx[i * 2 + j]);
  */
  dweight.ToHost();
  EXPECT_EQ(6u, dweight.Size());
  /*
  const float *dweightx = dweight.data<float>();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(
          (dy[0 * 3 + i] * x[0 * 2 + j] + dy[1 * 3 + i] * x[1 * 2 + j] +
           dy[2 * 3 + i] * x[2 * 2 + j]),
          dweightx[j * 2 + i]);
  */
  dbias.ToHost();
  const float *dbiasx = dbias.data<float>();
  EXPECT_EQ(3u, dbias.Size());
  for (int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ((dy[0 * 3 + i] + dy[1 * 3 + i] + dy[2 * 3 + i]), dbiasx[i]);
}
#endif
