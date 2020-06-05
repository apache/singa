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
  int bidirectional = 0;
  int num_layers = 1;

  Tensor x(Shape{seq_length, batch_size, feature_size}, cuda);
  Gaussian(0.0f, 1.0f, &x);

  CudnnRNNHandle rnn_handle(x, hidden_size, 0);

  Tensor W(Shape{rnn_handle.weights_size}, cuda);
  Gaussian(0.0f, 1.0f, &W);

  Tensor hx(
      Shape{num_layers * (bidirectional ? 2 : 1), batch_size, hidden_size},
      cuda);
  Tensor cx(
      Shape{num_layers * (bidirectional ? 2 : 1), batch_size, hidden_size},
      cuda);
  hx.SetValue(0.0f);
  cx.SetValue(0.0f);

  auto outputs = GpuRNNForwardTraining(x, hx, cx, W, rnn_handle);
  auto outputs2 = GpuRNNForwardInference(x, hx, cx, W, rnn_handle);

  Tensor y(Shape{seq_length, batch_size, hidden_size * (bidirectional ? 2 : 1)},
           cuda);
  Tensor dy(
      Shape{seq_length, batch_size, hidden_size * (bidirectional ? 2 : 1)},
      cuda);
  Tensor dhy(
      Shape{num_layers * (bidirectional ? 2 : 1), batch_size, hidden_size},
      cuda);
  Tensor dcy(
      Shape{num_layers * (bidirectional ? 2 : 1), batch_size, hidden_size},
      cuda);
  Gaussian(0.0f, 1.0f, &y);
  Gaussian(0.0f, 1.0f, &dy);
  Gaussian(0.0f, 1.0f, &dhy);
  Gaussian(0.0f, 1.0f, &dcy);
  auto output3 = GpuRNNBackwardx(y, dy, dhy, dcy, W, hx, cx, rnn_handle);
  auto dW = GpuRNNBackwardW(x, hx, y, rnn_handle);
}
#endif  // USE_CUDNN
