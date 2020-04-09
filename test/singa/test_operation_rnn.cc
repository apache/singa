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
#include "../src/model/operation/rnn.h"
#include "gtest/gtest.h"
#include "singa/core/tensor.h"
#include "singa/singa_config.h"

using namespace singa;

#ifdef USE_CUDNN
TEST(Operation_RNN, tranining) {

  auto cuda = std::make_shared<singa::CudaGPU>();

  int hidden_size = 7;
  int seq_length = 5;
  int batch_size = 6;
  int feature_size = 3;

  vector<Tensor> x;
  for (int i=0;i<seq_length;i++){
    Tensor tmp(Shape{batch_size, feature_size}, cuda);
    Gaussian(0.0f, 1.0f, &tmp);
    x.push_back(tmp);
  }

  CudnnRNNHandle rnn_handle(x, feature_size,hidden_size, 2);

  Tensor W(Shape{rnn_handle.weights_size}, cuda);
  Gaussian(0.0f, 1.0f, &W);

  std::cout<<"forward training\n";
  auto y1 = GpuRNNForwardTraining(x, W, rnn_handle);
  auto y2 = GpuRNNForwardInference(x, W, rnn_handle);
  auto dx = GpuRNNBackwardx(y1, y2, W, rnn_handle);
  auto dW = GpuRNNBackwardW(x, y1, rnn_handle);
}
#endif  // USE_CUDNN
