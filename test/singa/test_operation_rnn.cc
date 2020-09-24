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
TEST(OperationRNN, tranining) {
  auto cuda = std::make_shared<singa::CudaGPU>();

  size_t hidden_size = 7;
  int seq_length = 5;
  size_t batch_size = 6;
  size_t feature_size = 3;
  size_t num_layers = 1;
  int bdirect = 0;

  Shape s_s{num_layers * (bdirect ? 2 : 1), batch_size, hidden_size};
  Shape y_s{seq_length, batch_size, hidden_size * (bdirect ? 2 : 1)};

  // x
  Tensor x(Shape{seq_length, batch_size, feature_size}, cuda);
  Gaussian(0.0f, 1.0f, &x);

  // x hidden states and cell states
  Tensor hx(s_s, cuda);
  Tensor cx(s_s, cuda);
  hx.SetValue(0.0f);
  cx.SetValue(0.0f);

  // y dy
  Tensor y(y_s, cuda);
  Tensor dy(y_s, cuda);
  Gaussian(0.0f, 1.0f, &y);
  Gaussian(0.0f, 1.0f, &dy);

  // y hidden states and cell states
  Tensor dhy(s_s, cuda);
  Tensor dcy(s_s, cuda);
  Gaussian(0.0f, 1.0f, &dhy);
  Gaussian(0.0f, 1.0f, &dcy);

  // init handle and weights
  CudnnRNNHandle rnn_handle(x, hidden_size);
  Tensor W(Shape{rnn_handle.weights_size}, cuda);
  Gaussian(0.0f, 1.0f, &W);

  // forward and backward passes
  auto outputs = GpuRNNForwardTraining(x, hx, cx, W, rnn_handle);
  auto outputs2 = GpuRNNForwardInference(x, hx, cx, W, rnn_handle);
  auto output3 = GpuRNNBackwardx(y, dy, dhy, dcy, W, hx, cx, rnn_handle);
  auto dW = GpuRNNBackwardW(x, hx, y, rnn_handle);
}

TEST(OperationRNNEx, tranining) {
  auto cuda = std::make_shared<singa::CudaGPU>();

  size_t hidden_size = 2;
  size_t seq_length = 6;
  size_t batch_size = 6;
  size_t feature_size = 4;
  int bdirect = 0;  // 0 or 1
  size_t num_layers = 1;

  Shape s_s{num_layers * (bdirect ? 2 : 1), batch_size, hidden_size};
  Shape y_s{seq_length, batch_size, hidden_size * (bdirect ? 2 : 1)};
  Shape x_s{seq_length, batch_size, feature_size};

  // x
  Tensor x(x_s, cuda);
  Gaussian(0.0f, 1.0f, &x);

  // x hidden states and cell states
  Tensor hx(s_s, cuda);
  Tensor cx(s_s, cuda);
  hx.SetValue(0.0f);
  cx.SetValue(0.0f);

  // y hidden states and cell states
  Tensor dhy(s_s, cuda);
  Tensor dcy(s_s, cuda);
  Gaussian(0.0f, 1.0f, &dhy);
  Gaussian(0.0f, 1.0f, &dcy);

  // y dy
  Tensor y(y_s, cuda);
  Tensor dy(y_s, cuda);
  Gaussian(0.0f, 1.0f, &y);
  Gaussian(0.0f, 1.0f, &dy);

  // seq lengths
  Tensor seq_lengths(
      Shape{
          batch_size,
      },
      cuda, singa::kInt);
  vector<int> data(batch_size, seq_length);
  seq_lengths.CopyDataFromHostPtr(data.data(), batch_size);

  // init handle and weights
  CudnnRNNHandle rnn_handle(x, hidden_size, 0);
  Tensor W(Shape{rnn_handle.weights_size}, cuda);
  Gaussian(0.0f, 1.0f, &W);

  // forward and backward passes for batch first format
  /* TODO: WARNING: Logging before InitGoogleLogging() is written to STDERR
    F0619 07:11:43.435175  1094 rnn.cc:658] Check failed: status ==
    CUDNN_STATUS_SUCCESS (8 vs. 0)  CUDNN_STATUS_EXECUTION_FAILED
    *** Check failure stack trace: ***
    Aborted (core dumped)
    */
  auto outputs = GpuRNNForwardTrainingEx(x, hx, cx, W, seq_lengths, rnn_handle);
  auto outputs2 =
      GpuRNNForwardInferenceEx(x, hx, cx, W, seq_lengths, rnn_handle);
  auto outputs3 =
      GpuRNNBackwardxEx(y, dy, dhy, dcy, W, hx, cx, seq_lengths, rnn_handle);
  auto dW = GpuRNNBackwardWEx(x, hx, y, seq_lengths, rnn_handle);
}

#endif  // USE_CUDNN
