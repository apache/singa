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
#include "singa/singa_config.h"
#ifdef USE_CUDNN

#include <cudnn.h>
#include <math.h>  // exp tanh

#include "../src/model/layer/cudnn_activation.h"
#include "gtest/gtest.h"
#include "singa/proto/core.pb.h"

using singa::CudnnActivation;
using singa::Shape;
TEST(CudnnActivation, Setup) {
  CudnnActivation acti;
  // EXPECT_EQ("CudnnActivation", acti.layer_type());

  singa::LayerConf conf;
  conf.set_type("cudnn_relu");
  singa::ReLUConf* reluconf = conf.mutable_relu_conf();
  reluconf->set_negative_slope(0.5f);

  acti.Setup(Shape{3}, conf);
  //  EXPECT_EQ(CUDNN_ACTIVATION_RELU, acti.CudnnMode());
  EXPECT_EQ(0.5f, acti.Negative_slope());
}

TEST(CudnnActivation, Forward) {
  const float x[] = {1.0f, 2.0f, 3.0f, -2.0f, -3.0f, -4.0};
  size_t n = sizeof(x) / sizeof(float);
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(singa::Shape{n}, cuda);
  in.CopyDataFromHostPtr<float>(x, n);

  float neg_slope = 0.5f;
  std::string types[] = {"cudnn_sigmoid", "cudnn_tanh", "cudnn_relu"};
  for (int j = 0; j < 3; j++) {
    CudnnActivation acti;
    singa::LayerConf conf;
    std::string layertype = types[j];
    conf.set_type(layertype);
    if (layertype == "relu") {
      singa::ReLUConf* reluconf = conf.mutable_relu_conf();
      reluconf->set_negative_slope(neg_slope);
    }
    acti.Setup(Shape{n}, conf);

    singa::Tensor out = acti.Forward(singa::kTrain, in);
    EXPECT_EQ(n, out.Size());
    out.ToHost();
    const float* yptr = out.data<float>();
    float* y = new float[n];
    if (acti.Mode() == "sigmoid") {
      for (size_t i = 0; i < n; i++) y[i] = 1.f / (1.f + exp(-x[i]));
    } else if (acti.Mode() == "tanh") {
      for (size_t i = 0; i < n; i++) y[i] = tanh(x[i]);
    } else if (acti.Mode() == "relu") {
      for (size_t i = 0; i < n; i++) y[i] = (x[i] >= 0.f) ? x[i] : 0.f;
    } else {
      LOG(FATAL) << "Unkown activation: " << acti.Mode();
    }
    EXPECT_FLOAT_EQ(y[0], yptr[0]);
    EXPECT_FLOAT_EQ(y[4], yptr[4]);
    EXPECT_FLOAT_EQ(y[5], yptr[5]);
    delete[] y;
  }
}

TEST(CudnnActivation, Backward) {
  const float x[] = {2.0f, 3.0f, 3.0f, 7.f, 0.0f, 5.0, 1.5, 2.5, -2.5, 1.5};
  size_t n = sizeof(x) / sizeof(float);
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor in(singa::Shape{n}, cuda);
  in.CopyDataFromHostPtr<float>(x, n);
  float neg_slope = 0.5f;
  std::string types[] = {"cudnn_sigmoid", "cudnn_tanh", "cudnn_relu"};
  for (int j = 0; j < 3; j++) {
    CudnnActivation acti;
    singa::LayerConf conf;
    std::string layertype = types[j];
    conf.set_type(layertype);
    if (layertype == "RELU") {
      singa::ReLUConf* reluconf = conf.mutable_relu_conf();
      reluconf->set_negative_slope(neg_slope);
    }
    acti.Setup(Shape{n}, conf);
    singa::Tensor out = acti.Forward(singa::kTrain, in);
    EXPECT_EQ(n, out.Size());
    out.ToHost();
    const float* yptr = out.data<float>();

    const float grad[] = {2.0f, 1.0f, 2.0f, 0.0f, -2.0f,
                          -1.0, 1.5,  2.5,  -1.5, -2.5};
    singa::Tensor out_diff(singa::Shape{n}, cuda);
    out_diff.CopyDataFromHostPtr<float>(grad, n);
    const auto ret = acti.Backward(singa::kTrain, out_diff);
    singa::Tensor in_diff = ret.first;
    in_diff.ToHost();
    const float* xptr = in_diff.data<float>();
    float* dx = new float[n];
    if (acti.Mode() == "sigmoid") {
      for (size_t i = 0; i < n; i++) dx[i] = grad[i] * yptr[i] * (1. - yptr[i]);
    } else if (acti.Mode() == "tanh") {
      for (size_t i = 0; i < n; i++) dx[i] = grad[i] * (1. - yptr[i] * yptr[i]);
    } else if (acti.Mode() == "relu") {
      for (size_t i = 0; i < n; i++)
        dx[i] =
            grad[i] * (x[i] > 0.f);  //+ acti.Negative_slope() * (x[i] <= 0.f);
    } else {
      LOG(FATAL) << "Unkown activation: " << acti.Mode();
    }
    for (size_t i = 0; i < n; i++) {
      EXPECT_NEAR(dx[i], xptr[i], 1e-7);
    }
    delete[] dx;
  }
}
#endif  // USE_CUDNN
