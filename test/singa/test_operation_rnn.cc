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

struct hdl_t {
  int bs, data_s, seq_len, bidir, num_lay, hid_s;
  cudnnDataType_t dtype;
};

void init_descs(cudnnTensorDescriptor_t *descs, hdl_t &h) {
  auto type = CUDNN_DATA_FLOAT;
  int dimA[] = {h.bs, h.data_s, 1};
  int strideA[] = {dimA[1] * dimA[2], dimA[2], 1};
  for (int i = 0; i < h.seq_len; i++) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs[i]));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(descs[i], type, 3, dimA,
                                           strideA));
  }
}

void init_state_desc(cudnnTensorDescriptor_t &desc, hdl_t &h) {
  /* If direction is CUDNN_BIDIRECTIONAL then the first dimension should match
  double the numLayers argument passed to cudnnSetRNNDescriptor(). */
  /* The second dimension must match the batchSize parameter in xDesc */
  /* the third dimension must match the hiddenSize argument passed to the
  cudnnSetRNNDescriptor() call used to initialize rnnDesc. */
  int dim[] = {h.num_lay * (h.bidir ? 2 : 1), h.bs, h.hid_s};
  int stride[] = {dim[2] * dim[1], dim[2], 1};
  auto type = CUDNN_DATA_FLOAT;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CHECK( cudnnSetTensorNdDescriptor(desc, type, 3, dim, stride));
}

void init_dropout(cudnnDropoutDescriptor_t &dropoutDesc, Context* ctx) {
  size_t seed = 0x1234567;
  size_t stateSize;
  float dropout = 0.0f;
  CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropoutDesc));
  CUDNN_CHECK(cudnnDropoutGetStatesSize(ctx->cudnn_handle, &stateSize));
  // memory
  void *states;
  CUDA_CHECK(cudaMalloc(&states, stateSize));
  CUDNN_CHECK(cudnnSetDropoutDescriptor(dropoutDesc, ctx->cudnn_handle, dropout,
                                        states, stateSize, seed));
  // CUDA_CHECK(cudaFree(states));
}

void init_rnn_desc(cudnnRNNDescriptor_t &rnnDesc,
                   cudnnDropoutDescriptor_t &dropoutDesc, Context* ctx,
                   hdl_t &h) {
  /* rnn desc */
  CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnnDesc));
  auto RNNMode = CUDNN_RNN_RELU;
  // RNNMode = CUDNN_RNN_TANH;
  // RNNMode = CUDNN_LSTM;
  // RNNMode = CUDNN_GRU;
  auto type = CUDNN_DATA_FLOAT;
  auto algo = CUDNN_RNN_ALGO_STANDARD;
  CUDNN_CHECK(cudnnSetRNNDescriptor(
      ctx->cudnn_handle, rnnDesc, h.hid_s, h.num_lay, dropoutDesc,
      CUDNN_LINEAR_INPUT,
      h.bidir ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, RNNMode,
      algo,  // CUDNN_RNN_ALGO_STANDARD,
      type));
}

void init_parameters_desc(cudnnFilterDescriptor_t &wDesc,
                          cudnnTensorDescriptor_t *xDesc,
                          int weights_size_bytes,
                          hdl_t &h) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
  int weights_size = weights_size_bytes / sizeof(float);  // TODO different types
  int dimW[] = {weights_size, 1, 1};
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(wDesc, h.dtype,
                                         CUDNN_TENSOR_NCHW, 3, dimW));
}

int prod(vector<int> shape) {
  int prod = 1;
  for (const auto &s : shape) prod *= s;
  return prod;
}

void printt(Tensor &t, std::string mes="tensor "){
  std::shared_ptr<Device> dev = t.device();
  t.ToHost();
  const float* dptr = static_cast<const float*>(t.block()->data());
  std::cout<< mes << "\n --> size:" << t.size() << " shape: ";
  for (int i =0; i<t.n_dim();i++){
    std::cout<<t.shape()[i]<<",";
  }
  std::cout<<"\n --> val: ";
  for (int i =0; i<t.size();i++){
    std::cout<<dptr[i]<<",";
  }
  std::cout<<"\n";
  t.ToDevice(dev);
}

void setBias(int linLayerID, int pseudoLayer, Tensor &weights,
                vector<float> biasValues, cudnnRNNDescriptor_t &rnnDesc,
                cudnnTensorDescriptor_t *xDesc, Context *ctx,
                cudnnFilterDescriptor_t &wDesc) {

  cudnnFilterDescriptor_t linLayerBiasDesc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
  void *linLayerBias = nullptr;
  void *W_ptr = weights.block()->mutable_data();
  CUDNN_CHECK(cudnnGetRNNLinLayerBiasParams(
      ctx->cudnn_handle, rnnDesc, pseudoLayer, xDesc[0], wDesc, W_ptr,
      linLayerID, linLayerBiasDesc, &linLayerBias));

  int dims[] = {1, 1, 1};
  cudnnDataType_t data_type;
  cudnnTensorFormat_t tensor_format;
  int n_dims;

  CUDNN_CHECK(cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, &data_type,
                                         &tensor_format, &n_dims, dims));

  size_t offset = (float*)linLayerBias - (float*)W_ptr;
  weights.CopyDataFromHostPtr(biasValues.data(), dims[0] * dims[1] * dims[2],
                              offset);
}

void setWeight(int linLayerID, int pseudoLayer, Tensor &weights,
                vector<float> weightValues, cudnnRNNDescriptor_t &rnnDesc,
                cudnnTensorDescriptor_t *xDesc, Context *ctx,
                cudnnFilterDescriptor_t &wDesc) {
  // TODO: assert input value

  cudnnFilterDescriptor_t linLayerMatDesc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&linLayerMatDesc));
  void *linLayerMat = nullptr;
  // linLayID 0 or 1
  //   0 input from the previous layer - cudnn name:W - singa name:Wx
  //   1 recurrent input - R - Wh
  //   cudnn fomula: ht = ReLU(Wixt + Riht-1 + bWi + bRi)
  //   0 -> Wx -> {data_s, hid_s}
  //      b-> {hid,}
  //   1 -> Wh {hid_s,hid_s}

  //   int linLayerID=0;

  // pseudoLayer:
  //   if uni directional
  //   pseudoLayer=0 is the RNN input layer,
  //   pseudoLayer=1 is the first hidden layer
  //   and so on

  //   int pseudoLayer=0;

  void* W_ptr = weights.block()->mutable_data();
  CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
      ctx->cudnn_handle, rnnDesc, pseudoLayer, xDesc[0], wDesc, W_ptr,
      linLayerID, linLayerMatDesc, &linLayerMat));


  int dims[] = {1, 1, 1};
  cudnnDataType_t data_type;
  cudnnTensorFormat_t tensor_format;
  int n_dims;
  CUDNN_CHECK(cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &data_type,
                                         &tensor_format, &n_dims, dims));

  size_t offset = (float*)linLayerMat - (float*)W_ptr;

  weights.CopyDataFromHostPtr(weightValues.data(), dims[0]*dims[1]*dims[2], offset);
}

TEST(cudnnrnn, setWeights) {
  auto cuda = std::make_shared<singa::CudaGPU>();
  size_t hid_s = 2;
  int seq = 2;
  size_t bs = 2;
  size_t data_s = 2;
  size_t num_lay = 1;
  int bidir = 0;

  float x_val[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  Tensor x(Shape{seq, bs, data_s}, cuda);
  x.CopyDataFromHostPtr(x_val, x.size());

  CudnnRNNHandle rnn_handle(x, hid_s);

  Tensor W(Shape{rnn_handle.weights_size}, cuda);
  W.SetValue(0.0f);

  float Wx_val[] = {1.0f, 1.0f, 1.0f, 1.0f};
  Tensor Wx(Shape{data_s, hid_s}, cuda);
  Wx.CopyDataFromHostPtr(Wx_val, Wx.size());

  float Wh_val[] = {2.0f, 2.0f, 2.0f, 2.0f};
  Tensor Wh(Shape{hid_s, hid_s}, cuda);
  Wh.CopyDataFromHostPtr(Wh_val, Wh.size());

  float Bx_val[] = {3.0f, 3.0f};
  Tensor Bx(Shape{hid_s}, cuda);
  Bx.CopyDataFromHostPtr(Bx_val, Bx.size());

  float Bh_val[] = {4.0f, 4.0f};
  Tensor Bh(Shape{hid_s}, cuda);
  Bh.CopyDataFromHostPtr(Bh_val, Bh.size());

  GpuRNNSetParam(0, 0, W, Wx, false, rnn_handle);
  GpuRNNSetParam(1, 0, W, Wh, false, rnn_handle);
  GpuRNNSetParam(0, 0, W, Bx, true, rnn_handle);
  GpuRNNSetParam(1, 0, W, Bh, true, rnn_handle);

  printt(W, "weights");
}

TEST(cudnnrnn, tranining) {
  shared_ptr<Device> dev = std::make_shared<singa::CudaGPU>();
  Context* ctx = dev->context(0);

  // int bs, data_s, seq_len, bidir, num_lay, hid_s, dtype;
  hdl_t h = {2, 2, 2, 0, 1, 2, CUDNN_DATA_FLOAT};


  cudnnTensorDescriptor_t *xDesc = new cudnnTensorDescriptor_t[h.seq_len];
  cudnnTensorDescriptor_t *yDesc = new cudnnTensorDescriptor_t[h.seq_len];
  init_descs(xDesc, h);
  init_descs(yDesc, h);

  cudnnTensorDescriptor_t hxDesc, cxDesc, hyDesc, cyDesc;
  init_state_desc(hxDesc, h);
  init_state_desc(cxDesc, h);
  init_state_desc(hyDesc, h);
  init_state_desc(cyDesc, h);


  // dropout
  cudnnDropoutDescriptor_t dropoutDesc;
  init_dropout(dropoutDesc, ctx);

  // rnndesc
  cudnnRNNDescriptor_t rnnDesc;
  init_rnn_desc(rnnDesc, dropoutDesc, ctx, h);

  // not existing in cudnn 7.4.5
  // CUDNN_CHECK(cudnnSetRNNBiasMode(rnnDesc, CUDNN_RNN_NO_BIAS));

  // w size
  size_t weights_size_bytes, workspace_size_bytes, reserve_size_bytes;
  CUDNN_CHECK(cudnnGetRNNParamsSize(ctx->cudnn_handle, rnnDesc, xDesc[0],
                                    &weights_size_bytes, h.dtype));

  // w and dw
  cudnnFilterDescriptor_t wDesc, dwDesc;
  init_parameters_desc(wDesc, xDesc, weights_size_bytes, h);
  init_parameters_desc(dwDesc, xDesc, weights_size_bytes, h);

  // workspace and reserve 
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(ctx->cudnn_handle, rnnDesc, h.seq_len,
                                       xDesc, &workspace_size_bytes));
  CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
      ctx->cudnn_handle, rnnDesc, h.seq_len, xDesc, &reserve_size_bytes));

  int weights_size = weights_size_bytes / sizeof(float);
  int workspace_size = workspace_size_bytes / sizeof(float);
  int reserve_size = reserve_size_bytes / sizeof(float);

  vector<int> xshape = {h.bs, h.seq_len, h.data_s};
  vector<int> yshape = {h.bs, h.seq_len, (h.bidir ? 2 : 1) * h.hid_s};
  vector<int> stateshape = {h.bs, h.num_lay * (h.bidir ? 2 : 1), h.hid_s};

  // create
  Tensor x(Shape{xshape.begin(), xshape.end()}, dev);
  Tensor hx(Shape{stateshape.begin(),stateshape.end()}, dev);
  Tensor cx(Shape{stateshape.begin(),stateshape.end()}, dev);
  Tensor W(Shape{weights_size,}, dev);
  Tensor y(Shape{yshape.begin(), yshape.end()}, dev);
  Tensor hy(Shape{stateshape.begin(),stateshape.end()}, dev);
  Tensor cy(Shape{stateshape.begin(),stateshape.end()}, dev);
  Tensor ws(Shape{workspace_size,}, dev);
  Tensor rs(Shape{reserve_size,}, dev);

  // init values
  x.SetValue(0.0f);
  hx.SetValue(0.0f);
  cx.SetValue(0.0f);
  W.SetValue(0.0f);
  y.SetValue(1.0f);
  hy.SetValue(0.0f);
  cy.SetValue(0.0f);
  ws.SetValue(0.0f);
  rs.SetValue(0.0f);
  auto x_ptr = x.block()->data();
  auto hx_ptr = hx.block()->data();
  auto cx_ptr = cx.block()->data();
  auto W_ptr = W.block()->data();
  auto y_ptr = y.block()->mutable_data();
  auto hy_ptr = hy.block()->mutable_data();
  auto cy_ptr = cy.block()->mutable_data();
  auto ws_ptr = ws.block()->mutable_data();
  auto rs_ptr = rs.block()->mutable_data();

  float x_val[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  x.CopyDataFromHostPtr(x_val, x.size());

  vector<float> wx_val{1.0f,1.0f,1.0f,1.0f};
  vector<float> bx_val{1.0f,1.0f,1.0f,1.0f};
  setWeight(0, 0, W, wx_val, rnnDesc, xDesc, ctx, wDesc);
  setWeight(1, 0, W, wx_val, rnnDesc, xDesc, ctx, wDesc);
  setBias(0, 0, W, bx_val, rnnDesc, xDesc, ctx, wDesc);
  setBias(1, 0, W, bx_val, rnnDesc, xDesc, ctx, wDesc);

  CUDNN_CHECK(cudnnRNNForwardTraining(
      ctx->cudnn_handle, rnnDesc, h.seq_len, xDesc, x_ptr, hxDesc, hx_ptr,
      cxDesc, cx_ptr, wDesc, W_ptr, yDesc, y_ptr, hyDesc, hy_ptr, cyDesc, cy_ptr,
      ws_ptr, workspace_size_bytes, rs_ptr, reserve_size_bytes));

  printt(x, "after forward x ");
  printt(y, "after forward y ");

  delete[] xDesc;
  delete[] yDesc;
}

#endif  // USE_CUDNN
