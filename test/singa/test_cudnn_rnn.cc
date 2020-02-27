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
#if CUDNN_VERSION >= 5005

#include "gtest/gtest.h"

using singa::CudnnRNN;
using singa::Shape;
using singa::Tensor;
class TestCudnnRNN : public ::testing::Test {
 protected:
  virtual void SetUp() {
    singa::RNNConf *rnnconf = conf.mutable_rnn_conf();
    rnnconf->set_hidden_size(hidden_size);
    rnnconf->set_num_stacks(1);
    rnnconf->set_dropout(0);
    rnnconf->set_input_mode("linear");
    rnnconf->set_direction("unidirectional");
    rnnconf->set_rnn_mode("tanh");
  }
  singa::LayerConf conf;
  size_t hidden_size = 4;
};

TEST_F(TestCudnnRNN, Setup) {
  CudnnRNN rnn;
  // EXPECT_EQ("CudnnRNN", rnn.layer_type());
  rnn.Setup(Shape{2}, conf);
  auto weight = rnn.param_values().at(0);
  EXPECT_EQ(weight.Size(), hidden_size * (2 + hidden_size + 2));
}

TEST_F(TestCudnnRNN, Forward) {
  auto cuda = std::make_shared<singa::CudaGPU>();
  const size_t seqLength = 4, batchsize = 1, dim = 2;
  const float x[seqLength * batchsize * dim] = {1.0f, 1.0f, 1.0f, 1.0f,
                                                1.0f, 1.0f, 1.0f, 1.0f};

  vector<Tensor> inputs;
  for (size_t i = 0; i < seqLength; i++) {
    Tensor t(Shape{batchsize, dim}, cuda);
    t.CopyDataFromHostPtr(x + i * t.Size(), t.Size());
    inputs.push_back(t);
  }

  singa::Tensor hx;
  inputs.push_back(hx);

  CudnnRNN rnn;
  rnn.Setup(Shape{dim}, conf);
  rnn.ToDevice(cuda);

  auto weight = rnn.param_values().at(0);
  size_t weightSize = weight.Size();
  float we[weightSize];
  float wvalue = 0.1f;
  for (size_t i = 0; i < weightSize; i++) we[i] = wvalue;
  weight.CopyDataFromHostPtr(we, weightSize);

  const auto ret = rnn.Forward(singa::kEval, inputs);
  EXPECT_EQ(ret.size(), seqLength + 1);
  vector<float> hxptr(hidden_size, 0.0f);
  for (size_t i = 0; i < seqLength; i++) {
    auto y = ret[i];
    y.ToHost();
    auto yptr = y.data<float>();
    vector<float> tmp;
    for (size_t j = 0; j < hidden_size; j++) {
      float ty = 0;
      for (size_t k = 0; k < dim; k++) {
        ty += x[i * dim + k] * wvalue;
      }
      ty += wvalue;
      for (size_t k = 0; k < hidden_size; k++) {
        ty += hxptr[k] * wvalue;
      }
      ty += wvalue;
      ty = tanh(ty);
      EXPECT_NEAR(ty, yptr[j], 1e-4);
      tmp.push_back(ty);
    }
    std::copy(tmp.begin(), tmp.end(), hxptr.begin());
  }
}

TEST_F(TestCudnnRNN, Backward) {
  auto cuda = std::make_shared<singa::CudaGPU>();
  const size_t seqLength = 4, batchsize = 1, dim = 2;
  const float x[seqLength * batchsize * dim] = {1.0f, 1.0f, 1.0f, 1.0f,
                                                1.0f, 1.0f, 1.0f, 1.0f};

  vector<Tensor> inputs;
  for (size_t i = 0; i < seqLength; i++) {
    Tensor t(Shape{batchsize, dim}, cuda);
    t.CopyDataFromHostPtr(x + i * t.Size(), t.Size());
    inputs.push_back(t);
  }

  singa::Tensor hx;
  inputs.push_back(hx);

  CudnnRNN rnn;
  rnn.Setup(Shape{dim}, conf);
  rnn.ToDevice(cuda);

  auto weight = rnn.param_values().at(0);
  size_t weightSize = weight.Size();
  float we[weightSize];
  float wvalue = 0.1f;
  for (size_t i = 0; i < weightSize; i++) we[i] = wvalue;
  weight.CopyDataFromHostPtr(we, weightSize);

  const auto outs = rnn.Forward(singa::kTrain, inputs);

  float dyptr[seqLength * batchsize * hidden_size];
  for (size_t i = 0; i < seqLength * batchsize * hidden_size; i++)
    dyptr[i] = i * 0.1f;
  vector<Tensor> grads;
  for (size_t i = 0; i < seqLength; i++) {
    Tensor dy(Shape{batchsize, hidden_size}, cuda);
    dy.CopyDataFromHostPtr(dyptr + i * dy.Size(), dy.Size());
    grads.push_back(dy);
  }
  Tensor dhy;
  grads.push_back(dhy);
  vector<float> dhyptr(hidden_size, 0.0f);
  const auto ret = rnn.Backward(singa::kTrain, grads);
  for (size_t i = seqLength - 1; i > 0; i--) {
    auto dx = ret.first[i];
    auto y = outs[i].Clone();
    y.ToHost();
    dx.ToHost();
    auto dxptr = dx.data<float>();
    auto yptr = y.data<float>();
    for (size_t j = 0; j < hidden_size; j++) {
      dhyptr[j] += dyptr[i * hidden_size + j];
      dhyptr[j] *= 1 - yptr[j] * yptr[j];
    }
    for (size_t k = 0; k < dim; k++) {
      float tdx = 0;
      for (size_t j = 0; j < hidden_size; j++) {
        tdx += dhyptr[j] * wvalue;
      }
      EXPECT_NEAR(tdx, dxptr[k], 1e-4);
    }
    vector<float> tmp;
    for (size_t k = 0; k < hidden_size; k++) {
      float tdhy = 0;
      for (size_t j = 0; j < hidden_size; j++) {
        tdhy += dhyptr[j] * wvalue;
      }
      tmp.push_back(tdhy);
    }
    std::copy(tmp.begin(), tmp.end(), dhyptr.begin());
  }
}
#endif  // CUDNN_VERSION >= 5005
#endif  // USE_CUDNN
