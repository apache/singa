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

#include "../src/model/layer/cudnn_rnn.h"
#ifdef USE_CUDNN

#include "gtest/gtest.h"

using singa::CudnnRNN;
using singa::Shape;
TEST(CudnnRNN, Setup) {
  CudnnRNN rnn;
  EXPECT_EQ("CudnnRNN", rnn.layer_type());

  singa::LayerConf conf;
  singa::RNNConf *rnnconf = conf.mutable_rnn_conf();
  rnnconf->set_hiddensize(2);
  rnnconf->set_numlayers(1);
  rnnconf->set_dropout(0); 
  rnnconf->set_inputmode("cudnn_linear_input");
  rnnconf->set_direction("cudnn_undirectional");
  rnnconf->set_mode("cudnn_rnn_tanh");
  // MB
  rnnconf->set_workspace_byte_limit(256);
  rnn.Setup(Shape{4, 1, 2}, conf);

  EXPECT_EQ(2u, rnn.hiddenSize());
  EXPECT_EQ(1u, rnn.numLayers());
  EXPECT_EQ(0u, rnn.dropout());
  EXPECT_EQ("cudnn_linear_input", rnn.inputMode());
  EXPECT_EQ("cudnn_undirectional", rnn.direction());
  EXPECT_EQ("cudnn_rnn_tanh", rnn.mode());
  EXPECT_EQ(256u << 20, rnn.workspace_byte_limit());
}

TEST(CudnnRNN, Forward) {
  auto cuda = std::make_shared<singa::CudaGPU>();
  const size_t seqLength = 4, batchsize = 1, dim = 2;
  const size_t numLayers = 1, hiddensize = 2, numDirections = 1;
  const float x[seqLength * batchsize * dim] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                          1.0f, 1.0f, 1.0f};
  singa::Tensor in(singa::Shape{seqLength, batchsize, dim}, cuda);
  in.CopyDataFromHostPtr(x, seqLength * batchsize * dim);


  
  const float hx_data[numLayers * batchsize * hiddensize * numDirections] = {1.0f, 1.0f};
  singa::Tensor hx(singa::Shape{numLayers, batchsize, hiddensize * numDirections}, cuda);
  hx.CopyDataFromHostPtr(hx_data, numLayers * batchsize * hiddensize * numDirections);

  const float cx_data[numLayers * batchsize * hiddensize * numDirections] = {1.0f, 1.0f};
  singa::Tensor cx(singa::Shape{numLayers, batchsize, hiddensize * numDirections}, cuda);
  cx.CopyDataFromHostPtr(cx_data, numLayers * batchsize * hiddensize * numDirections);
  
  CudnnRNN rnn;
  
  singa::LayerConf conf;
  singa::RNNConf *rnnconf = conf.mutable_rnn_conf();
  rnnconf->set_hiddensize(2);
  rnnconf->set_numlayers(1);
  rnnconf->set_dropout(0);
  rnnconf->set_inputmode("cudnn_linear_input");
  rnnconf->set_direction("cudnn_undirectional");
  rnnconf->set_mode("cudnn_rnn_tanh");
  // MB
  rnnconf->set_workspace_byte_limit(256);
  rnn.Setup(Shape{4, 1, 2}, conf);
 
  
  size_t weightSize = rnn.weightSize();
  float we[weightSize];
  for (size_t i = 0; i < weightSize; i++)
    we[i] = 1.0f;
  singa::Tensor weight(singa::Shape{weightSize, 1, 1}, cuda);
  weight.CopyDataFromHostPtr(we, weightSize);
  rnn.set_weight(weight);
 
  vector<singa::Tensor> input_array;
  input_array.push_back(in);
  input_array.push_back(hx);
  input_array.push_back(cx);
  const auto ret = rnn.Forward(singa::kTrain, input_array);
  // singa::CppCPU host(0, 1);
  singa::Tensor out1 = ret[0];
  out1.ToHost();
  const float *outptr1 = out1.data<float>();
  EXPECT_EQ(8u, out1.Size());
  EXPECT_NEAR(1.0f, outptr1[0], 0.0001); // tanh 6
  EXPECT_NEAR(1.0f, outptr1[1], 0.0001);
  EXPECT_NEAR(1.0f, outptr1[2], 0.0001);
  EXPECT_NEAR(1.0f, outptr1[3], 0.0001);
  EXPECT_NEAR(1.0f, outptr1[4], 0.0001);
  EXPECT_NEAR(1.0f, outptr1[5], 0.0001);
  EXPECT_NEAR(1.0f, outptr1[6], 0.0001);
  EXPECT_NEAR(1.0f, outptr1[7], 0.0001);

  singa::Tensor hy1 = ret[1];
  hy1.ToHost();
  const float *hyptr1 = hy1.data<float>();
  EXPECT_EQ(2u, hy1.Size());
  EXPECT_NEAR(1.0f, hyptr1[0], 0.0001);
  EXPECT_NEAR(1.0f, hyptr1[1], 0.0001);
}

TEST(CudnnRNN, Backward) {
  // src_data
  auto cuda = std::make_shared<singa::CudaGPU>();
  const size_t seqLength = 4, batchsize = 1, dim = 2;
  const size_t numLayers = 1, hiddensize = 2, numDirections = 1;
  const float x[seqLength * batchsize * dim] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                          1.0f, 1.0f, 1.0f};
  singa::Tensor in(singa::Shape{seqLength, batchsize, dim}, cuda);
  in.CopyDataFromHostPtr(x, seqLength * batchsize * dim);

  const float hx_data[numLayers * batchsize * hiddensize * numDirections] = {1.0f, 1.0f};
  singa::Tensor hx(singa::Shape{numLayers, batchsize, hiddensize * numDirections}, cuda);
  hx.CopyDataFromHostPtr(hx_data, numLayers * batchsize * hiddensize * numDirections);

  const float cx_data[numLayers * batchsize * hiddensize * numDirections] = {1.0f, 1.0f};
  singa::Tensor cx(singa::Shape{numLayers, batchsize, hiddensize * numDirections}, cuda);
  cx.CopyDataFromHostPtr(cx_data, numLayers * batchsize * hiddensize * numDirections);

  CudnnRNN rnn;

  singa::LayerConf conf;
  singa::RNNConf *rnnconf = conf.mutable_rnn_conf();
  rnnconf->set_hiddensize(2);
  rnnconf->set_numlayers(1);
  rnnconf->set_dropout(0);
  rnnconf->set_inputmode("cudnn_linear_input");
  rnnconf->set_direction("cudnn_undirectional");
  rnnconf->set_mode("cudnn_rnn_tanh");
  // MB
  rnnconf->set_workspace_byte_limit(256);
  rnn.Setup(Shape{4, 1, 2}, conf);

  size_t weightSize = rnn.weightSize();
  float we[weightSize];
  for (size_t i = 0; i < weightSize; i++)
    we[i] = 1.0f;
  singa::Tensor weight(singa::Shape{weightSize, 1, 1}, cuda);
  weight.CopyDataFromHostPtr(we, weightSize);
  rnn.set_weight(weight);


  vector<singa::Tensor> input_array;
  input_array.push_back(in);
  input_array.push_back(hx);
  input_array.push_back(cx);
  const auto ret = rnn.Forward(singa::kTrain, input_array);

  // grad
  const float dy[seqLength * batchsize * hiddensize * numDirections] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  singa::Tensor grad(singa::Shape{seqLength, batchsize, hiddensize * numDirections},
                     cuda);
  grad.CopyDataFromHostPtr(dy, seqLength * batchsize * hiddensize * numDirections);

  const float dhy_data[numLayers * batchsize * hiddensize * numDirections] = {1.0f, 1.0f};
  singa::Tensor dhy(singa::Shape{numLayers, batchsize, hiddensize * numDirections},
                     cuda);
  dhy.CopyDataFromHostPtr(dhy_data, numLayers * batchsize * hiddensize * numDirections);

  const float dcy_data[numLayers * batchsize * hiddensize * numDirections] = {1.0f, 1.0f};
  singa::Tensor dcy(singa::Shape{numLayers, batchsize, hiddensize * numDirections},
                     cuda);
  dcy.CopyDataFromHostPtr(dcy_data, numLayers * batchsize * hiddensize * numDirections);

  vector<singa::Tensor> grad_array;
  grad_array.push_back(grad);
  grad_array.push_back(dhy);
  grad_array.push_back(dcy);
  const auto ret_back = rnn.Backward(singa::kTrain, grad_array);
  // singa::CppCPU host(0, 1);
  singa::Tensor in_grad = ret_back.first[0];
  in_grad.ToHost();
  const float *dx = in_grad.data<float>();
  EXPECT_EQ(8u, in_grad.Size());
  EXPECT_NEAR(0.14, dx[0], 0.0001);
  EXPECT_NEAR(0.14, dx[1], 0.0001);
  EXPECT_NEAR(0.1596, dx[2], 0.0001);
  EXPECT_NEAR(0.1596, dx[3], 0.0001);
  EXPECT_NEAR(0.1623, dx[4], 0.0001);
  EXPECT_NEAR(0.1623, dx[5], 0.0001);
  EXPECT_NEAR(0.1627, dx[6], 0.0001);
  EXPECT_NEAR(0.1627, dx[7], 0.0001);

  singa::Tensor dhx_grad = ret_back.first[1];
  dhx_grad.ToHost();
  const float *dhx = dhx_grad.data<float>();
  EXPECT_EQ(2u, dhx_grad.Size());
  EXPECT_NEAR(0.1627, dhx[0], 0.0001);
  EXPECT_NEAR(0.1627, dhx[1], 0.0001);
}
#endif  // USE_CUDNN
