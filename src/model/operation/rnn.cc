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

#include "rnn.h"
namespace singa {
#ifdef USE_CUDNN
CudnnRNNHandle::CudnnRNNHandle(const Tensor &x, const int hidden_size,
                               const int mode, const int num_layers,
                               const int bias, const float dropout,
                               const int bidirectional)
    : bias(bias),
      dropout(dropout),
      bidirectional(bidirectional),
      hidden_size(hidden_size),
      mode(mode),
      num_layers(num_layers) {
  CHECK_EQ(bias, 1) << "Current implementation always include bias";
  CHECK(bidirectional == 0 || bidirectional == 1)
      << "bidirectional should be 0 or 1 not " << bidirectional;

  dev = x.device();
  ctx = x.device()->context(0);

  seq_length = x.shape(0);
  batch_size = x.shape(1);
  feature_size = x.shape(2);

  cudnnRNNAlgo = CUDNN_RNN_ALGO_STANDARD;
  cudnnDataType = CUDNN_DATA_FLOAT;

  init_data_desc();
  update_data_desc();
  init_dropout_desc();
  init_rnn_desc();
  init_parameters_desc();
  init_workspace();
}

void CudnnRNNHandle::init_workspace() {
  /* workspace data */
  // Need for every pass
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(ctx->cudnn_handle, rnnDesc, seq_length,
                                       xDesc, &workspace_size));
  // Only needed in training, shouldn't be touched between passes.
  CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(ctx->cudnn_handle, rnnDesc,
                                             seq_length, xDesc, &reserve_size));

  workspace = Tensor(Shape{workspace_size}, dev);
  reserve_space = Tensor(Shape{reserve_size}, dev);
}

void CudnnRNNHandle::init_parameters_desc() {
  /* weights size
   *   depends on rnn desc */
  CUDNN_CHECK(cudnnGetRNNParamsSize(ctx->cudnn_handle, rnnDesc, xDesc[0],
                                    &weights_size, cudnnDataType));
  /* weights desc
   *   depends on weights size */
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&dwDesc));

  int dimW[3];
  dimW[0] = weights_size / sizeof(float);  // TODO different types
  dimW[1] = 1;
  dimW[2] = 1;
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(wDesc, cudnnDataType,
                                         CUDNN_TENSOR_NCHW, 3, dimW));
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(dwDesc, cudnnDataType,
                                         CUDNN_TENSOR_NCHW, 3, dimW));
}

void CudnnRNNHandle::init_rnn_desc() {
  /* rnn desc */
  CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnnDesc));
  if (mode == 0)
    RNNMode = CUDNN_RNN_RELU;
  else if (mode == 1)
    RNNMode = CUDNN_RNN_TANH;
  else if (mode == 2)
    RNNMode = CUDNN_LSTM;
  else if (mode == 3)
    RNNMode = CUDNN_GRU;
  CUDNN_CHECK(cudnnSetRNNDescriptor(
      ctx->cudnn_handle, rnnDesc, hidden_size, num_layers, dropoutDesc,
      CUDNN_LINEAR_INPUT,
      bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, RNNMode,
      cudnnRNNAlgo,  // CUDNN_RNN_ALGO_STANDARD,
      cudnnDataType));
}
void CudnnRNNHandle::init_dropout_desc() {
  /* drop out */
  size_t seed = 0x1234567;
  CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropoutDesc));
  size_t stateSize;
  CUDNN_CHECK(cudnnDropoutGetStatesSize(ctx->cudnn_handle, &stateSize));
  CUDA_CHECK(cudaMalloc(&states, stateSize));
  CUDNN_CHECK(cudnnSetDropoutDescriptor(dropoutDesc, ctx->cudnn_handle, dropout,
                                        states, stateSize, seed));
}

void CudnnRNNHandle::init_data_desc() {
  /* xDesc, yDesc */
  xDesc = (cudnnTensorDescriptor_t *)malloc(seq_length *
                                            sizeof(cudnnTensorDescriptor_t));
  yDesc = (cudnnTensorDescriptor_t *)malloc(seq_length *
                                            sizeof(cudnnTensorDescriptor_t));
  dxDesc = (cudnnTensorDescriptor_t *)malloc(seq_length *
                                             sizeof(cudnnTensorDescriptor_t));
  dyDesc = (cudnnTensorDescriptor_t *)malloc(seq_length *
                                             sizeof(cudnnTensorDescriptor_t));

  for (int i = 0; i < seq_length; i++) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc[i]));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc[i]));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc[i]));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc[i]));
  }
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&hxDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&cxDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&hyDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&cyDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&dhxDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&dcxDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&dhyDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&dcyDesc));
}

void CudnnRNNHandle::update_data_desc() {
  int dimA[3];
  int strideA[3];

  // init list of desc for x, y
  for (int i = 0; i < seq_length; i++) {
    // dimA[0] = x[i].shape(0); // batch size
    dimA[0] = batch_size;  // TODO bs changes
    dimA[1] = feature_size;
    dimA[2] = 1;
    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;
    CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(xDesc[i], cudnnDataType, 3, dimA, strideA));
    CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(dxDesc[i], cudnnDataType, 3, dimA, strideA));

    // dimA[0] = x[i].shape(0); // batch size
    dimA[0] = batch_size;  // TODO bs changes
    dimA[1] = bidirectional ? hidden_size * 2 : hidden_size;
    dimA[2] = 1;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;
    CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(yDesc[i], cudnnDataType, 3, dimA, strideA));
    CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(dyDesc[i], cudnnDataType, 3, dimA, strideA));
  }

  dimA[0] = num_layers * (bidirectional ? 2 : 1);
  dimA[1] = batch_size;
  dimA[2] = hidden_size;

  strideA[0] = dimA[2] * dimA[1];
  strideA[1] = dimA[2];
  strideA[2] = 1;

  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(hxDesc, cudnnDataType, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(cxDesc, cudnnDataType, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(hyDesc, cudnnDataType, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(cyDesc, cudnnDataType, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(dhxDesc, cudnnDataType, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(dcxDesc, cudnnDataType, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(dhyDesc, cudnnDataType, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(dcyDesc, cudnnDataType, 3, dimA, strideA));
}

// reserve for masking
Tensor CudnnRNNHandle::merge_inputs(size_t num, const vector<Tensor> &in) {
  if (num == 1) return in.at(0);
  size_t size = 0;
  for (size_t i = 0; i < num; i++) size += in.at(i).Size();
  Tensor out(Shape{size}, in.at(0).device(), in.at(0).data_type());
  for (size_t i = 0, offset = 0; i < num; i++) {
    CopyDataToFrom(&out, in.at(i), in.at(i).Size(), offset);
    offset += in.at(i).Size();
  }
  return out;
}
vector<Tensor> CudnnRNNHandle::split_output(size_t num, size_t dim,
                                            const vector<Tensor> &in,
                                            const Tensor output) {
  vector<Tensor> outputs;
  if (num == 1) {
    outputs.push_back(Reshape(output, Shape{in.at(0).shape(0), dim}));
  } else {
    for (size_t i = 0, offset = 0; offset < output.Size(); i++) {
      Shape s{in.at(i).shape(0), dim};
      Tensor out(s, output.device(), output.data_type());
      CopyDataToFrom(&out, output, out.Size(), 0, offset);
      outputs.push_back(out);
      offset += out.Size();
    }
    CHECK_EQ(num, outputs.size());
  }
  return outputs;
}

CudnnRNNHandle::~CudnnRNNHandle() {
  free(xDesc);
  free(yDesc);
  free(dxDesc);
  free(dyDesc);
}

vector<Tensor> GpuRNNForwardInference(const Tensor &x, const Tensor &hx,
                                      const Tensor &cx, const Tensor &W,
                                      CudnnRNNHandle &h) {
  Tensor y(Shape{h.seq_length, h.batch_size,
                 h.hidden_size * (h.bidirectional ? 2 : 1)},
           x.device());
  Tensor hy(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                  h.hidden_size},
            x.device());
  Tensor cy(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                  h.hidden_size},
            x.device());
  y.device()->Exec(
      [&y, &hy, &cy, &x, &hx, &cx, &W, &h](Context *ctx) {
        CUDNN_CHECK(cudnnRNNForwardInference(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, h.xDesc,
            x.block()->data(), h.hxDesc, hx.block()->data(), h.cxDesc,
            cx.block()->data(), h.wDesc, W.block()->data(), h.yDesc,
            y.block()->mutable_data(), h.hyDesc, hy.block()->mutable_data(),
            h.cyDesc, cy.block()->mutable_data(),
            h.workspace.block()->mutable_data(), h.workspace_size));
      },
      {x.block(), hx.block(), cx.block(), W.block()},
      {y.block(), hy.block(), cy.block()});
  return {y, hy, cy};
}

vector<Tensor> GpuRNNForwardTraining(const Tensor &x, const Tensor &hx,
                                     const Tensor &cx, const Tensor &W,
                                     CudnnRNNHandle &h) {
  Tensor y(Shape{h.seq_length, h.batch_size,
                 h.hidden_size * (h.bidirectional ? 2 : 1)},
           x.device());
  Tensor hy(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                  h.hidden_size},
            x.device());
  Tensor cy(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                  h.hidden_size},
            x.device());
  y.device()->Exec(
      [&y, &hy, &cy, &x, &hx, &cx, &W, &h](Context *ctx) {
        CUDNN_CHECK(cudnnRNNForwardTraining(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, h.xDesc,
            x.block()->data(), h.hxDesc, hx.block()->data(), h.cxDesc,
            cx.block()->data(), h.wDesc, W.block()->data(), h.yDesc,
            y.block()->mutable_data(), h.hyDesc, hy.block()->mutable_data(),
            h.cyDesc, cy.block()->mutable_data(),
            h.workspace.block()->mutable_data(), h.workspace_size,
            h.reserve_space.block()->mutable_data(), h.reserve_size));
      },
      {x.block(), hx.block(), cx.block(), W.block()},
      {y.block(), hy.block(), cy.block()});

  return {y, hy, cy};
}

vector<Tensor> GpuRNNBackwardx(const Tensor &y, const Tensor &dy,
                               const Tensor &dhy, const Tensor &dcy,
                               const Tensor &W, const Tensor &hx,
                               const Tensor &cx, CudnnRNNHandle &h) {
  Tensor dx(Shape{h.seq_length, h.batch_size, h.feature_size}, y.device());
  Tensor dhx(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                   h.hidden_size},
             y.device());
  Tensor dcx(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                   h.hidden_size},
             y.device());
  dx.device()->Exec(
      [&dx, &dhx, &dcx, &y, &dy, &dhy, &dcy, &W, &hx, &cx, &h](Context *ctx) {
        CUDNN_CHECK(cudnnRNNBackwardData(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, h.yDesc,
            y.block()->data(), h.dyDesc, dy.block()->data(), h.dhyDesc,
            dhy.block()->data(), h.dcyDesc, dcy.block()->data(), h.wDesc,
            W.block()->data(), h.hxDesc, hx.block()->data(), h.cxDesc,
            cx.block()->data(), h.dxDesc, dx.block()->mutable_data(), h.dhxDesc,
            dhx.block()->mutable_data(), h.dcxDesc, dcx.block()->mutable_data(),
            h.workspace.block()->mutable_data(), h.workspace_size,
            h.reserve_space.block()->mutable_data(), h.reserve_size));
      },
      {y.block(), dy.block(), dhy.block(), dcy.block(), hx.block(), cx.block(),
       W.block()},
      {dx.block(), dhx.block(), dcx.block()});
  return {dx, dhx, dcx};
}

Tensor GpuRNNBackwardW(const Tensor &x, const Tensor &hx, const Tensor &y,
                       CudnnRNNHandle &h) {
  Tensor dW(Shape{h.weights_size}, x.device());
  dW.device()->Exec(
      [&dW, &x, &hx, &y, &h](Context *ctx) {
        CUDNN_CHECK(cudnnRNNBackwardWeights(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, h.xDesc,
            x.block()->data(), h.hxDesc, hx.block()->data(), h.yDesc,
            y.block()->data(), h.workspace.block()->mutable_data(),
            h.workspace_size, h.dwDesc, dW.block()->mutable_data(),
            h.reserve_space.block()
                ->mutable_data(),  // from previous backward data
            h.reserve_size));
      },
      {x.block(), y.block(), hx.block()}, {dW.block()});
  return dW;
}

#endif  // USE_CUDNN
}  // namespace singa
