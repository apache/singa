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

  cudnnTensorDescriptor_t *xDesc = new cudnnTensorDescriptor_t[seq_length];
  init_xDesc(xDesc, *this);

  init_dropout_desc();
  init_rnn_desc();
  init_parameters_desc(xDesc);
  init_workspace(xDesc);
}

void CudnnRNNHandle::init_workspace(cudnnTensorDescriptor_t *xDesc) {
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

void CudnnRNNHandle::init_parameters_desc(cudnnTensorDescriptor_t *xDesc) {
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

void init_yDesc(cudnnTensorDescriptor_t *yDesc, CudnnRNNHandle &h) {
  int dimA[3];
  int strideA[3];
  dimA[0] = h.batch_size;
  dimA[1] = h.bidirectional ? h.hidden_size * 2 : h.hidden_size;
  dimA[2] = 1;
  strideA[0] = dimA[2] * dimA[1];
  strideA[1] = dimA[2];
  strideA[2] = 1;

  for (int i = 0; i < h.seq_length; i++) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc[i]));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(yDesc[i], h.cudnnDataType, 3, dimA,
                                           strideA));
  }
}

void init_xDesc(cudnnTensorDescriptor_t *xDesc, CudnnRNNHandle &h) {
  int dimA[3];
  int strideA[3];
  dimA[0] = h.batch_size;
  dimA[1] = h.feature_size;
  dimA[2] = 1;
  strideA[0] = dimA[2] * dimA[1];
  strideA[1] = dimA[2];
  strideA[2] = 1;

  for (int i = 0; i < h.seq_length; i++) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc[i]));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(xDesc[i], h.cudnnDataType, 3, dimA,
                                           strideA));
  }
}

void init_hc_Desc(cudnnTensorDescriptor_t &hxDesc, CudnnRNNHandle &h) {
  int dimA[3];
  int strideA[3];
  dimA[0] = h.num_layers * (h.bidirectional ? 2 : 1);
  dimA[1] = h.batch_size;
  dimA[2] = h.hidden_size;
  strideA[0] = dimA[2] * dimA[1];
  strideA[1] = dimA[2];
  strideA[2] = 1;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&hxDesc));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(hxDesc, h.cudnnDataType, 3, dimA, strideA));
}

vector<Tensor> GpuRNNForwardInference(const Tensor &x, const Tensor &hx,
                                      const Tensor &cx, const Tensor &W,
                                      CudnnRNNHandle &h) {
  CHECK_EQ(h.feature_size, x.shape(2)) << "feature size should not change";
  h.seq_length = x.shape(0);
  h.batch_size = x.shape(1);  // update batch size to accomodate bs change
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
      [y, hy, cy, x, hx, cx, &W, &h](Context *ctx) {
        // require desc, [x], hx, cx, w, y, hy, cy
        cudnnTensorDescriptor_t *xDesc =
            new cudnnTensorDescriptor_t[h.seq_length];
        cudnnTensorDescriptor_t *yDesc =
            new cudnnTensorDescriptor_t[h.seq_length];
        init_xDesc(xDesc, h);
        init_yDesc(yDesc, h);
        cudnnTensorDescriptor_t hxDesc;
        cudnnTensorDescriptor_t cxDesc;
        cudnnTensorDescriptor_t hyDesc;
        cudnnTensorDescriptor_t cyDesc;
        init_hc_Desc(hxDesc, h);
        init_hc_Desc(cxDesc, h);
        init_hc_Desc(hyDesc, h);
        init_hc_Desc(cyDesc, h);

        auto xptr = x.block()->data();
        auto hxptr = hx.block()->data();
        auto cxptr = cx.block()->data();
        auto Wptr = W.block()->data();
        auto yptr = y.block()->mutable_data();
        auto hyptr = hy.block()->mutable_data();
        auto cyptr = cy.block()->mutable_data();
        auto wsptr = h.workspace.block()->mutable_data();

        CUDNN_CHECK(cudnnRNNForwardInference(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, xDesc, xptr, hxDesc,
            hxptr, cxDesc, cxptr, h.wDesc, Wptr, yDesc, yptr, hyDesc, hyptr,
            cyDesc, cyptr, wsptr, h.workspace_size));

        delete[] xDesc;
        delete[] yDesc;
      },
      {x.block(), hx.block(), cx.block(), W.block()},
      {y.block(), hy.block(), cy.block(), h.workspace.block()},
      "cudnnRNNForwardInterface");
  return {y, hy, cy};
}

vector<Tensor> GpuRNNForwardTraining(const Tensor &x, const Tensor &hx,
                                     const Tensor &cx, const Tensor &W,
                                     CudnnRNNHandle &h) {
  CHECK_EQ(h.feature_size, x.shape(2)) << "feature size should not change";
  h.seq_length = x.shape(0);
  h.batch_size = x.shape(1);  // update batch size to accomodate bs change
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
      [y, hy, cy, x, hx, cx, &W, &h](Context *ctx) {
        // require desc, [x], hx, cx, w, y, hy, cy
        cudnnTensorDescriptor_t *xDesc =
            new cudnnTensorDescriptor_t[h.seq_length];
        cudnnTensorDescriptor_t *yDesc =
            new cudnnTensorDescriptor_t[h.seq_length];
        init_xDesc(xDesc, h);
        init_yDesc(yDesc, h);
        cudnnTensorDescriptor_t hxDesc;
        cudnnTensorDescriptor_t cxDesc;
        cudnnTensorDescriptor_t hyDesc;
        cudnnTensorDescriptor_t cyDesc;
        init_hc_Desc(hxDesc, h);
        init_hc_Desc(cxDesc, h);
        init_hc_Desc(hyDesc, h);
        init_hc_Desc(cyDesc, h);

        auto xptr = x.block()->data();
        auto hxptr = hx.block()->data();
        auto cxptr = cx.block()->data();
        auto Wptr = W.block()->data();
        auto yptr = y.block()->mutable_data();
        auto hyptr = hy.block()->mutable_data();
        auto cyptr = cy.block()->mutable_data();
        auto wsptr = h.workspace.block()->mutable_data();
        auto rsptr = h.reserve_space.block()->mutable_data();
        CUDNN_CHECK(cudnnRNNForwardTraining(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, xDesc, xptr, hxDesc,
            hxptr, cxDesc, cxptr, h.wDesc, Wptr, yDesc, yptr, hyDesc, hyptr,
            cyDesc, cyptr, wsptr, h.workspace_size, rsptr, h.reserve_size));
        delete[] xDesc;
        delete[] yDesc;
      },
      {x.block(), hx.block(), cx.block(), W.block()},
      {y.block(), hy.block(), cy.block(), h.workspace.block(),
       h.reserve_space.block()},
      "cudnnRNNForwardTraining");

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
      [dx, dhx, dcx, y, dy, dhy, dcy, &W, hx, cx, &h](Context *ctx) {
        // require desc:
        //      [dx], hx, dhx, cx, dcx, w,
        // [y], [dy],     dhy,     dcy
        cudnnTensorDescriptor_t *dxDesc =
            new cudnnTensorDescriptor_t[h.seq_length];
        cudnnTensorDescriptor_t *yDesc =
            new cudnnTensorDescriptor_t[h.seq_length];
        cudnnTensorDescriptor_t *dyDesc =
            new cudnnTensorDescriptor_t[h.seq_length];
        init_yDesc(yDesc, h);
        init_xDesc(dxDesc, h);
        init_yDesc(dyDesc, h);
        cudnnTensorDescriptor_t hxDesc;
        cudnnTensorDescriptor_t cxDesc;
        cudnnTensorDescriptor_t dhxDesc;
        cudnnTensorDescriptor_t dcxDesc;
        cudnnTensorDescriptor_t dhyDesc;
        cudnnTensorDescriptor_t dcyDesc;
        init_hc_Desc(hxDesc, h);
        init_hc_Desc(cxDesc, h);
        init_hc_Desc(dhxDesc, h);
        init_hc_Desc(dcxDesc, h);
        init_hc_Desc(dhyDesc, h);
        init_hc_Desc(dcyDesc, h);

        auto dxptr = dx.block()->mutable_data();
        auto hxptr = hx.block()->data();
        auto dhxptr = dhx.block()->mutable_data();
        auto cxptr = cx.block()->data();
        auto dcxptr = dcx.block()->mutable_data();
        auto Wptr = W.block()->data();
        auto yptr = y.block()->data();
        auto dyptr = dy.block()->data();
        auto dhyptr = dhy.block()->data();
        auto dcyptr = dcy.block()->data();
        auto wsptr = h.workspace.block()->mutable_data();
        auto rsptr = h.reserve_space.block()->mutable_data();

        CUDNN_CHECK(cudnnRNNBackwardData(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, yDesc, yptr, dyDesc,
            dyptr, dhyDesc, dhyptr, dcyDesc, dcyptr, h.wDesc, Wptr, hxDesc,
            hxptr, cxDesc, cxptr, dxDesc, dxptr, dhxDesc, dhxptr, dcxDesc,
            dcxptr, wsptr, h.workspace_size, rsptr, h.reserve_size));
        delete[] dxDesc;
        delete[] yDesc;
        delete[] dyDesc;
      },
      {y.block(), dy.block(), dhy.block(), dcy.block(), hx.block(), cx.block(),
       W.block()},
      {dx.block(), dhx.block(), dcx.block(), h.workspace.block(),
       h.reserve_space.block()},
      "cudnnRNNBackwardx");
  return {dx, dhx, dcx};
}

Tensor GpuRNNBackwardW(const Tensor &x, const Tensor &hx, const Tensor &y,
                       CudnnRNNHandle &h) {
  Tensor dW(Shape{h.weights_size}, x.device());
  dW.device()->Exec(
      [dW, x, hx, y, &h](Context *ctx) {
        cudnnTensorDescriptor_t *xDesc =
            new cudnnTensorDescriptor_t[h.seq_length];
        cudnnTensorDescriptor_t *yDesc =
            new cudnnTensorDescriptor_t[h.seq_length];
        init_xDesc(xDesc, h);
        init_yDesc(yDesc, h);
        cudnnTensorDescriptor_t hxDesc;
        init_hc_Desc(hxDesc, h);
        auto xptr = x.block()->data();
        auto hxptr = hx.block()->data();
        auto yptr = y.block()->data();
        auto dWptr = dW.block()->mutable_data();
        auto wsptr = h.workspace.block()->mutable_data();
        auto rsptr = h.reserve_space.block()->mutable_data();
        CUDNN_CHECK(cudnnRNNBackwardWeights(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, xDesc, xptr, hxDesc,
            hxptr, yDesc, yptr, wsptr, h.workspace_size, h.dwDesc, dWptr, rsptr,
            h.reserve_size));
        delete[] xDesc;
        delete[] yDesc;
      },
      {x.block(), y.block(), hx.block()},
      {dW.block(), h.workspace.block(), h.reserve_space.block()},
      "cudnnRnnBackwardW");
  return dW;
}

#endif  // USE_CUDNN
}  // namespace singa
