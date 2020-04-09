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
CudnnRNNHandle::CudnnRNNHandle(const vector<Tensor> &x,
  const size_t feature_size,
  const size_t hidden_size,
  const int mode,
  const size_t num_layers,
  const int  bias,
  const float dropout,
  const int bidirectional)
  : bias(bias),
    dropout(dropout),
    bidirectional(bidirectional),
    feature_size(feature_size),
    hidden_size(hidden_size),
    mode(mode),
    num_layers(num_layers)
{
  dev = x[0].device();
  ctx = x[0].device()->context(0);

  update_data_desc(x);
  init_dropout_desc();
  init_rnn_desc();
  init_parameters_desc();
  init_workspace();

std::cout<<"handle ok\n";
}

void CudnnRNNHandle::init_workspace(){
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

void CudnnRNNHandle::init_parameters_desc(){
  /* weights size
   *   depends on rnn desc */
  CUDNN_CHECK(cudnnGetRNNParamsSize(ctx->cudnn_handle, rnnDesc, xDesc[0],
                                    &weights_size, CUDNN_DATA_FLOAT));
  /* weights desc
   *   depends on weights size */
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&dwDesc));

  int dimW[3];
  dimW[0] = weights_size / sizeof(float);
  dimW[1] = 1;
  dimW[2] = 1;
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, 3, dimW));
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, 3, dimW));
}

void CudnnRNNHandle::init_rnn_desc(){
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
  algo = CUDNN_RNN_ALGO_STANDARD;  // TODO
  CUDNN_CHECK(cudnnSetRNNDescriptor(
      ctx->cudnn_handle, rnnDesc, hidden_size, num_layers, dropoutDesc,
      CUDNN_LINEAR_INPUT,
      CUDNN_UNIDIRECTIONAL, RNNMode,
      algo,  // CUDNN_RNN_ALGO_STANDARD,
      CUDNN_DATA_FLOAT));
}
void CudnnRNNHandle::init_dropout_desc(){
  /* drop out */
  size_t seed = 0x1234567;
  CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropoutDesc));
  size_t stateSize;
  CUDNN_CHECK(cudnnDropoutGetStatesSize(ctx->cudnn_handle, &stateSize));
  CUDA_CHECK(cudaMalloc(&states, stateSize));
  CUDNN_CHECK(cudnnSetDropoutDescriptor(dropoutDesc, ctx->cudnn_handle,
                                        dropout, states, stateSize, seed));
}

void CudnnRNNHandle::update_data_desc(const vector<Tensor> &x){
  /*handle x*/
  for (int i=0;i<x.size();i++){ CHECK_EQ(x[i].shape(1), feature_size); }
  seq_length = x.size();
  batch_size = x[0].shape(0);

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

  int dimA[3];
  int strideA[3];

  // init list of desc for x, y
  for (int i = 0; i < seq_length; i++) {
    dimA[0] = x[i].shape(0);
    dimA[1] = feature_size;
    dimA[2] = 1;
    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA,
                                           strideA));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(dxDesc[i], CUDNN_DATA_FLOAT, 3, dimA,
                                           strideA));
    dimA[0] = x[i].shape(0);
    dimA[1] = hidden_size;
    dimA[2] = 1;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimA,
                                           strideA));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(dyDesc[i], CUDNN_DATA_FLOAT, 3, dimA,
                                           strideA));
  }

  dimA[0] = num_layers;
  dimA[1] = batch_size;
  dimA[2] = hidden_size;

  strideA[0] = dimA[2] * dimA[1];
  strideA[1] = dimA[2];
  strideA[2] = 1;

  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(dhxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(dcxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(dhyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(dcyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
}

Tensor CudnnRNNHandle::merge_inputs(size_t num, const vector<Tensor> &in) {
  if (num == 1)
    return in.at(0);
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
  std::cout<<"destructing\n";
  free(xDesc);
  free(yDesc);
  free(dxDesc);
  free(dyDesc);
  std::cout<<"destructor ok\n";
}

vector<Tensor> GpuRNNForwardInference(const vector<Tensor> &x, Tensor &W, CudnnRNNHandle &rnn_handle){
  CHECK_EQ(x.size(), rnn_handle.seq_length);
  for (int i=0;i<x.size();i++){ CHECK_EQ(x[i].shape(1), rnn_handle.feature_size); }
  int update=0;
  for (int i=0;i<x.size();i++){
    if (x[i].shape(0) != rnn_handle.batch_size){
      update=1;
    }
  }
  if(update) rnn_handle.update_data_desc(x);

  Tensor contiguous_x = rnn_handle.merge_inputs(x.size(), x);

  Tensor contiguous_y(Shape{contiguous_x.Size() * rnn_handle.hidden_size / rnn_handle.feature_size}, x[0].device());

  contiguous_y.device()->Exec(
      [&contiguous_y, &contiguous_x, &W, &rnn_handle](Context *ctx) {
        void *hx = NULL;
        void *cx = NULL;
        void *hy = NULL;
        void *cy = NULL;
        CUDNN_CHECK(cudnnRNNForwardInference(
            ctx->cudnn_handle, rnn_handle.rnnDesc, rnn_handle.seq_length,
            rnn_handle.xDesc, contiguous_x.block()->data(), rnn_handle.hxDesc, hx,
            rnn_handle.cxDesc, cx, rnn_handle.wDesc, W.block()->data(),
            rnn_handle.yDesc, contiguous_y.block()->mutable_data(), rnn_handle.hyDesc, hy,
            rnn_handle.cyDesc, cy, rnn_handle.workspace.block()->mutable_data(),
            rnn_handle.workspace_size));
      },
      {contiguous_x.block(), W.block()}, {contiguous_y.block()});
  std::cout<<"inference ok\n";

  vector<Tensor> y = rnn_handle.split_output(x.size(), rnn_handle.hidden_size, x, contiguous_y);
  return y;
}


vector<Tensor> GpuRNNForwardTraining(const vector<Tensor> &x, Tensor &W, CudnnRNNHandle &rnn_handle){
  CHECK_EQ(x.size(), rnn_handle.seq_length);
  for (int i=0;i<x.size();i++){ CHECK_EQ(x[i].shape(1), rnn_handle.feature_size); }
  int update=0;
  for (int i=0;i<x.size();i++){
    if (x[i].shape(0) != rnn_handle.batch_size){
      update=1;
    }
  }
  if(update) rnn_handle.update_data_desc(x);

  Tensor contiguous_x = rnn_handle.merge_inputs(x.size(), x);

  Tensor contiguous_y(Shape{contiguous_x.Size() * rnn_handle.hidden_size / rnn_handle.feature_size}, x[0].device());

  contiguous_y.device()->Exec(
      [&contiguous_y, &contiguous_x, &W, &rnn_handle](Context *ctx) {
        void *hx = NULL;
        void *cx = NULL;
        void *hy = NULL;
        void *cy = NULL;

        CUDNN_CHECK(cudnnRNNForwardTraining(
            ctx->cudnn_handle, rnn_handle.rnnDesc, rnn_handle.seq_length,
            rnn_handle.xDesc, contiguous_x.block()->data(), rnn_handle.hxDesc, hx,
            rnn_handle.cxDesc, cx, rnn_handle.wDesc, W.block()->data(),
            rnn_handle.yDesc, contiguous_y.block()->mutable_data(), rnn_handle.hyDesc, hy,
            rnn_handle.cyDesc, cy, rnn_handle.workspace.block()->mutable_data(),
            rnn_handle.workspace_size,
            rnn_handle.reserve_space.block()->mutable_data(),
            rnn_handle.reserve_size));
      },
      {contiguous_x.block(), W.block()}, {contiguous_y.block()});

  std::cout<<"training ok\n";

  vector<Tensor> y = rnn_handle.split_output(x.size(), rnn_handle.hidden_size, x, contiguous_y);
  return y;
}

vector<Tensor> GpuRNNBackwardx(const vector<Tensor> &y, const vector<Tensor> &dy, const Tensor &W, CudnnRNNHandle &rnn_handle){

  Tensor contiguous_y = rnn_handle.merge_inputs(y.size(), y);
  Tensor contiguous_dy = rnn_handle.merge_inputs(dy.size(), dy);
  Tensor contiguous_dx(Shape{contiguous_y.Size() * rnn_handle.feature_size / rnn_handle.hidden_size}, y[0].device());

  contiguous_dx.device()->Exec(
      [&contiguous_dx, &contiguous_y, &contiguous_dy, &W, &rnn_handle](Context *ctx) {
        void *hx = NULL;
        void *cx = NULL;
        void *dhx = NULL;
        void *dcx = NULL;
        void *dhy = NULL;
        void *dcy = NULL;
        CUDNN_CHECK(cudnnRNNBackwardData(
            ctx->cudnn_handle, rnn_handle.rnnDesc, rnn_handle.seq_length,
            rnn_handle.yDesc, contiguous_y.block()->data(), rnn_handle.dyDesc,
            contiguous_dy.block()->data(), rnn_handle.dhyDesc, dhy, rnn_handle.dcyDesc,
            dcy, rnn_handle.wDesc, W.block()->data(), rnn_handle.hxDesc, hx,
            rnn_handle.cxDesc, cx, rnn_handle.dxDesc,
            contiguous_dx.block()->mutable_data(), rnn_handle.dhxDesc, dhx,
            rnn_handle.dcxDesc, dcx,
            rnn_handle.workspace.block()->mutable_data(),
            rnn_handle.workspace_size,
            rnn_handle.reserve_space.block()->mutable_data(),
            rnn_handle.reserve_size));
      },
      {contiguous_y.block(), contiguous_dy.block(), W.block()}, {contiguous_dx.block()});
  std::cout<<"back x ok\n";

  vector<Tensor> dx = rnn_handle.split_output(y.size(), rnn_handle.feature_size, y, contiguous_dx);
  return dx;
}

Tensor GpuRNNBackwardW(const vector<Tensor> &x, const vector<Tensor> &y, CudnnRNNHandle &rnn_handle){

  Tensor dW(Shape{rnn_handle.weights_size}, x[0].device());

  Tensor contiguous_x = rnn_handle.merge_inputs(x.size(), x);
  Tensor contiguous_y = rnn_handle.merge_inputs(y.size(), y);
  dW.device()->Exec(
      [&dW, &contiguous_x, &contiguous_y, &rnn_handle](Context *ctx) {
        void *hx = NULL;
        CUDNN_CHECK(cudnnRNNBackwardWeights(
            ctx->cudnn_handle, rnn_handle.rnnDesc, rnn_handle.seq_length,
            rnn_handle.xDesc, contiguous_x.block()->data(), rnn_handle.hxDesc, hx,
            rnn_handle.yDesc, contiguous_y.block()->data(),
            rnn_handle.workspace.block()->mutable_data(),
            rnn_handle.workspace_size, rnn_handle.dwDesc,
            dW.block()->mutable_data(),
            rnn_handle.reserve_space.block()
                ->mutable_data(),  // from previous backward data
            rnn_handle.reserve_size));
      },
      {contiguous_x.block(), contiguous_y.block()}, {dW.block()});
  std::cout<<"back w ok\n";
  return dW;
}

#endif  // USE_CUDNN
}  // namespace singa
