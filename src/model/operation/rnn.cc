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

#include <map>
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
  // cudnn rnn bias is not available in cudnn v7.4.5, not found in cudnn.h
  CHECK_EQ(bias, 1) << "Current implementation always include bias";
  CHECK(bidirectional == 0 || bidirectional == 1)
      << "bidirectional should be 0 or 1 not " << bidirectional;

  dev = x.device();
  ctx = x.device()->context(0);

  // TODO: batch first mode failed in cudnn
  batch_first = 0;

  // x shape {seq, bs, ..}
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
  init_param_mapping(xDesc);
  delete[] xDesc;
}

void CudnnRNNHandle::init_workspace(cudnnTensorDescriptor_t *xDesc) {
  /* workspace data */
  // Need for every pass
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(ctx->cudnn_handle, rnnDesc, seq_length,
                                       xDesc, &workspace_size_bytes));
  // Only needed in training, shouldn't be touched between passes.
  CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
      ctx->cudnn_handle, rnnDesc, seq_length, xDesc, &reserve_size_bytes));

  workspace_size = workspace_size_bytes / sizeof(float);
  reserve_size = reserve_size_bytes / sizeof(float);
  workspace = Tensor(Shape{workspace_size}, dev);
  reserve_space = Tensor(Shape{reserve_size}, dev);
}

void CudnnRNNHandle::init_parameters_desc(cudnnTensorDescriptor_t *xDesc) {
  /* weights size
   *   depends on rnn desc */
  CUDNN_CHECK(cudnnGetRNNParamsSize(ctx->cudnn_handle, rnnDesc, xDesc[0],
                                    &weights_size_bytes, cudnnDataType));
  /* weights desc
   *   depends on weights size */
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&dwDesc));

  weights_size = weights_size_bytes / sizeof(float);  // TODO different types
  int dimW[3];
  dimW[0] = weights_size;  // TODO different types
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

void init_yDesc(cudnnTensorDescriptor_t *yDesc, CudnnRNNHandle &h) {
  int dimA[] = {h.batch_size,
                h.bidirectional ? h.hidden_size * 2 : h.hidden_size, 1};
  int strideA[] = {dimA[1] * dimA[2], dimA[2], 1};
  for (int i = 0; i < h.seq_length; i++) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc[i]));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(yDesc[i], h.cudnnDataType, 3, dimA,
                                           strideA));
  }
}

void init_xDesc(cudnnTensorDescriptor_t *xDesc, CudnnRNNHandle &h) {
  int dimA[] = {h.batch_size, h.feature_size, 1};
  int strideA[] = {dimA[1] * dimA[2], dimA[2], 1};
  for (int i = 0; i < h.seq_length; i++) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc[i]));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(xDesc[i], h.cudnnDataType, 3, dimA,
                                           strideA));
  }
}

void init_hc_Desc(cudnnTensorDescriptor_t &hxDesc, CudnnRNNHandle &h) {
  /* If direction is CUDNN_BIDIRECTIONAL then the first dimension should match
  double the numLayers argument passed to cudnnSetRNNDescriptor(). */
  /* The second dimension must match the batchSize parameter in xDesc */
  /* the third dimension must match the hiddenSize argument passed to the
  cudnnSetRNNDescriptor() call used to initialize rnnDesc. */
  int dimA[] = {h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                h.hidden_size};
  int strideA[] = {dimA[2] * dimA[1], dimA[2], 1};
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&hxDesc));
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(hxDesc, h.cudnnDataType, 3, dimA, strideA));
}

/*
vector<Tensor> GpuRNNForwardTraining();
vector<Tensor> GpuRNNForwardInference();
vector<Tensor> GpuRNNBackwardx();
Tensor GpuRNNBackwardW();
*/

vector<Tensor> GpuRNNForwardInference(const Tensor &x, const Tensor &hx,
                                      const Tensor &cx, const Tensor &W,
                                      CudnnRNNHandle &h) {
  CHECK_EQ(h.feature_size, x.shape(2)) << "feature size should not change";

  // in
  // x in shape {seq, bs, ..}
  // out
  // y in shape {seq, bs, ..}

  h.batch_size = x.shape(1);  // update batch size to accomodate bs change
  h.seq_length = x.shape(0);

  Tensor y(Shape{h.seq_length, h.batch_size,
                 h.hidden_size * (h.bidirectional ? 2 : 1)},
           x.device());
  Tensor hy(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                  h.hidden_size},
            x.device());
  Tensor cy(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                  h.hidden_size},
            x.device());
  y.SetValue(0.0f);
  hy.SetValue(0.0f);
  cy.SetValue(0.0f);
  h.workspace.SetValue(0.0f);
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

        auto x_con = Contiguous(x);

        auto xptr = x_con.block()->data();
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
            cyDesc, cyptr, wsptr, h.workspace_size_bytes));

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

  // in
  // x in shape {seq, bs, ..}
  // out
  // y in shape {seq, bs, ..}

  // update batch size to accomodate bs change
  h.batch_size = x.shape(1);
  h.seq_length = x.shape(0);

  Tensor y(Shape{h.seq_length, h.batch_size,
                 h.hidden_size * (h.bidirectional ? 2 : 1)},
           x.device());
  Tensor hy(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                  h.hidden_size},
            x.device());
  Tensor cy(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                  h.hidden_size},
            x.device());
  y.SetValue(0.0f);
  hy.SetValue(0.0f);
  cy.SetValue(0.0f);
  h.workspace.SetValue(0.0f);
  h.reserve_space.SetValue(0.0f);

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

        auto x_con = Contiguous(x);

        auto xptr = x_con.block()->data();
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
            cyDesc, cyptr, wsptr, h.workspace_size_bytes, rsptr,
            h.reserve_size_bytes));
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
  // in
  // y shape {seq, bs}
  // dy shape {seq, bs}
  Tensor dx(Shape{h.seq_length, h.batch_size, h.feature_size}, y.device());
  Tensor dhx(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                   h.hidden_size},
             y.device());
  Tensor dcx(Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                   h.hidden_size},
             y.device());
  dx.SetValue(0.0f);
  dhx.SetValue(0.0f);
  dcx.SetValue(0.0f);
  h.workspace.SetValue(0.0f);
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

        auto y_con = Contiguous(y);
        auto dy_con = Contiguous(dy);

        auto dxptr = dx.block()->mutable_data();
        auto hxptr = hx.block()->data();
        auto dhxptr = dhx.block()->mutable_data();
        auto cxptr = cx.block()->data();
        auto dcxptr = dcx.block()->mutable_data();
        auto Wptr = W.block()->data();
        auto yptr = y_con.block()->data();
        auto dyptr = dy_con.block()->data();
        auto dhyptr = dhy.block()->data();
        auto dcyptr = dcy.block()->data();
        auto wsptr = h.workspace.block()->mutable_data();
        auto rsptr = h.reserve_space.block()->mutable_data();

        CUDNN_CHECK(cudnnRNNBackwardData(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, yDesc, yptr, dyDesc,
            dyptr, dhyDesc, dhyptr, dcyDesc, dcyptr, h.wDesc, Wptr, hxDesc,
            hxptr, cxDesc, cxptr, dxDesc, dxptr, dhxDesc, dhxptr, dcxDesc,
            dcxptr, wsptr, h.workspace_size_bytes, rsptr,
            h.reserve_size_bytes));
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
  // in
  // x shape {seq, bs}
  // y shape {seq, bs}
  dW.SetValue(0.0f);
  h.workspace.SetValue(0.0f);
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

        auto y_con = Contiguous(y);
        auto x_con = Contiguous(x);

        auto xptr = x_con.block()->data();
        auto hxptr = hx.block()->data();
        auto yptr = y_con.block()->data();
        auto dWptr = dW.block()->mutable_data();
        auto wsptr = h.workspace.block()->mutable_data();
        auto rsptr = h.reserve_space.block()->mutable_data();

        CUDNN_CHECK(cudnnRNNBackwardWeights(
            ctx->cudnn_handle, h.rnnDesc, h.seq_length, xDesc, xptr, hxDesc,
            hxptr, yDesc, yptr, wsptr, h.workspace_size_bytes, h.dwDesc, dWptr,
            rsptr, h.reserve_size_bytes));
        delete[] xDesc;
        delete[] yDesc;
      },
      {x.block(), y.block(), hx.block()},
      {dW.block(), h.workspace.block(), h.reserve_space.block()},
      "cudnnRnnBackwardW");
  return dW;
}

void CudnnRNNHandle::init_param_mapping(cudnnTensorDescriptor_t *xDesc) {
  int linLayerIDRange = 2;
  if (mode == 0 || mode == 1) {
    // vanilla relu/tanh
    linLayerIDRange = 2;
  } else if (mode == 2) {
    // lstm
    linLayerIDRange = 8;
  } else if (mode == 3) {
    // gru
    linLayerIDRange = 6;
  }
  int pseudoLayerRange = (bidirectional ? 2 : 1) * num_layers;

  // dummy weights for getting the offset
  Tensor weights(
      Shape{
          weights_size,
      },
      dev);
  weights.SetValue(0.0f);
  const void *W_ptr = weights.block()->data();

  void *param_ptr = nullptr;
  int dims[] = {1, 1, 1};
  cudnnDataType_t data_type;
  cudnnTensorFormat_t tensor_format;
  int n_dims;
  cudnnFilterDescriptor_t paramDesc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&paramDesc));

  vector<bool> paramTypes{false, true};
  for (int linLayerID = 0; linLayerID < linLayerIDRange; linLayerID++) {
    for (int pseudoLayer = 0; pseudoLayer < pseudoLayerRange; pseudoLayer++) {
      for (const bool &is_bias : paramTypes) {
        // get param ptr
        if (is_bias) {
          CUDNN_CHECK(cudnnGetRNNLinLayerBiasParams(
              ctx->cudnn_handle, rnnDesc, pseudoLayer, xDesc[0], wDesc, W_ptr,
              linLayerID, paramDesc, &param_ptr));
        } else {
          CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
              ctx->cudnn_handle, rnnDesc, pseudoLayer, xDesc[0], wDesc, W_ptr,
              linLayerID, paramDesc, &param_ptr));
        }

        // get param dims
        CUDNN_CHECK(cudnnGetFilterNdDescriptor(paramDesc, 3, &data_type,
                                               &tensor_format, &n_dims, dims));

        // get diff - offset
        size_t offset = (float *)param_ptr - (float *)W_ptr;

        // save in map
        weights_mapping[std::make_tuple(linLayerID, pseudoLayer, is_bias)] =
            std::make_tuple(offset, dims[0] * dims[1] * dims[2]);
      }
    }
  }
}

void GpuRNNSetParam(int linLayerID, int pseudoLayer, Tensor &weights,
                    Tensor &paramValues, bool is_bias, CudnnRNNHandle &h) {
  size_t offset, size;
  std::tie(offset, size) =
      h.weights_mapping[std::make_tuple(linLayerID, pseudoLayer, is_bias)];
  CHECK_EQ(size, paramValues.size()) << "param size is not expected";
  CopyDataToFrom(&weights, paramValues, size, offset, 0);
}

Tensor GpuRNNGetParamCopy(int linLayerID, int pseudoLayer, Tensor &weights,
                          bool is_bias, CudnnRNNHandle &h) {
  size_t offset, size;
  std::tie(offset, size) =
      h.weights_mapping[std::make_tuple(linLayerID, pseudoLayer, is_bias)];
  Tensor paramCopy(
      Shape{
          size,
      },
      weights.device());
  CopyDataToFrom(&paramCopy, weights, size, 0, offset);
  return paramCopy;
}

/*
vector<Tensor> GpuRNNForwardTrainingEx();
vector<Tensor> GpuRNNForwardInferenceEx();
vector<Tensor> GpuRNNBackwardxEx();
Tensor GpuRNNBackwardWEx();
*/

void init_data_desc(cudnnRNNDataDescriptor_t &desc, int data_size,
                    const Tensor seq_lengths, CudnnRNNHandle &h) {
  /* cudnnRNNDataDescriptor_t is a pointer to an opaque structure holding
  the description of an RNN data set. The function
  cudnnCreateRNNDataDescriptor() is used to create one instance, and
  cudnnSetRNNDataDescriptor() must be used to initialize this instance.
  */
  CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&desc));
  /* CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
    Data layout is padded, with outer stride from one time-step to the
  next. CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED The sequence length is
  sorted and packed as in basic RNN API.
  CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
    Data layout is padded, with outer stride from one batch to the next.
  */
  cudnnRNNDataLayout_t layout;
  if (h.batch_first) {
    layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
  } else {
    layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;
  }

  /* This is only effective when the descriptor is describing the RNN
  output, and the unpacked layout is specified.*/
  float paddingFill = 0.0f;

  /* Input. An integer array with batchSize number of elements.
  Describes the length (number of time-steps) of each sequence. Each
  element in seqLengthArray must be greater than 0 but less than or
  equal to maxSeqLength. */
  Tensor tmp = seq_lengths.Clone();
  tmp.ToHost();
  tmp = tmp.AsType(singa::kInt);
  const int *seq_lengths_ptr = static_cast<const int *>(tmp.block()->data());

  CUDNN_CHECK(cudnnSetRNNDataDescriptor(desc, h.cudnnDataType, layout,
                                        h.seq_length, h.batch_size, data_size,
                                        seq_lengths_ptr, (void *)&paddingFill));
}

vector<Tensor> GpuRNNForwardInferenceEx(const Tensor &x, const Tensor &hx,
                                        const Tensor &cx, const Tensor &W,
                                        const Tensor &seq_lengths,
                                        CudnnRNNHandle &h) {
  CHECK_EQ(h.feature_size, x.shape(2)) << "feature size should not change";

  Tensor y, hy, cy;
  Shape yshape, states_shape;

  if (h.batch_first) {
    LOG(FATAL) << "batch_first not implemented for GpuRNNForwardTrainingEx";
  } else {
    h.seq_length = x.shape(0);
    h.batch_size = x.shape(1);
    yshape = Shape{h.seq_length, h.batch_size,
                   h.hidden_size * (h.bidirectional ? 2 : 1)};
    states_shape = Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                         h.hidden_size};
  }

  y = Tensor(yshape, x.device());
  hy = Tensor(states_shape, x.device());
  cy = Tensor(states_shape, x.device());

  y.device()->Exec(
      [y, hy, cy, x, seq_lengths, hx, cx, &W, &h](Context *ctx) {
        // data descriptor
        cudnnRNNDataDescriptor_t xDesc, yDesc;
        init_data_desc(xDesc, h.feature_size, seq_lengths, h);
        init_data_desc(yDesc,
                       h.bidirectional ? h.hidden_size * 2 : h.hidden_size,
                       seq_lengths, h);

        // hidden cell states descriptor
        cudnnTensorDescriptor_t hxDesc, cxDesc, hyDesc, cyDesc;
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

        /* This routine is the extended version of the cudnnRNNForwardTraining()
        function. The cudnnRNNForwardTrainingEx() allows the user to use
        unpacked (padded) layout for input x and output y.
        */
        CUDNN_CHECK(cudnnRNNForwardInferenceEx(
            ctx->cudnn_handle, h.rnnDesc, xDesc, xptr, hxDesc, hxptr, cxDesc,
            cxptr, h.wDesc, Wptr, yDesc, yptr, hyDesc, hyptr, cyDesc, cyptr,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, wsptr,
            h.workspace_size_bytes));
      },
      {x.block(), hx.block(), cx.block(), W.block()},
      {y.block(), hy.block(), cy.block(), h.workspace.block(),
       h.reserve_space.block()});
  return {y, hy, cy};
}

vector<Tensor> GpuRNNForwardTrainingEx(const Tensor &x, const Tensor &hx,
                                       const Tensor &cx, const Tensor &W,
                                       const Tensor &seq_lengths,
                                       CudnnRNNHandle &h) {
  CHECK_EQ(h.feature_size, x.shape(2)) << "feature size should not change";

  Tensor y, hy, cy;
  Shape yshape, states_shape;

  if (h.batch_first) {
    LOG(FATAL) << "batch_first not implemented for GpuRNNForwardTrainingEx";
  } else {
    h.seq_length = x.shape(0);
    h.batch_size = x.shape(1);
    yshape = Shape{h.seq_length, h.batch_size,
                   h.hidden_size * (h.bidirectional ? 2 : 1)};
    states_shape = Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                         h.hidden_size};
  }

  y = Tensor(yshape, x.device());
  hy = Tensor(states_shape, x.device());
  cy = Tensor(states_shape, x.device());

  y.device()->Exec(
      [y, hy, cy, x, seq_lengths, hx, cx, &W, &h](Context *ctx) {
        // data descriptor
        cudnnRNNDataDescriptor_t xDesc, yDesc;
        init_data_desc(xDesc, h.feature_size, seq_lengths, h);
        init_data_desc(yDesc,
                       h.bidirectional ? h.hidden_size * 2 : h.hidden_size,
                       seq_lengths, h);

        // hidden cell states descriptor
        cudnnTensorDescriptor_t hxDesc, cxDesc, hyDesc, cyDesc;
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

        /* This routine is the extended version of the cudnnRNNForwardTraining()
        function. The cudnnRNNForwardTrainingEx() allows the user to use
        unpacked (padded) layout for input x and output y.
        */
        CUDNN_CHECK(cudnnRNNForwardTrainingEx(
            ctx->cudnn_handle, h.rnnDesc, xDesc, xptr, hxDesc, hxptr, cxDesc,
            cxptr, h.wDesc, Wptr, yDesc, yptr, hyDesc, hyptr, cyDesc, cyptr,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, wsptr,
            h.workspace_size_bytes, rsptr, h.reserve_size_bytes));
      },
      {x.block(), hx.block(), cx.block(), W.block()},
      {y.block(), hy.block(), cy.block(), h.workspace.block(),
       h.reserve_space.block()});
  return {y, hy, cy};
}

vector<Tensor> GpuRNNBackwardxEx(const Tensor &y, const Tensor &dy,
                                 const Tensor &dhy, const Tensor &dcy,
                                 const Tensor &W, const Tensor &hx,
                                 const Tensor &cx, const Tensor &seq_lengths,
                                 CudnnRNNHandle &h) {
  // y shape: {bs, seq}
  // dy shape: {bs, seq}
  // dx shape: {bs, seq}
  Shape xshape, states_shape;
  if (h.batch_first) {
    LOG(FATAL) << "batch_first not implemented for GpuRNNBackwardxEx";
  } else {
    xshape = Shape{h.batch_size, h.seq_length, h.feature_size};
    states_shape = Shape{h.num_layers * (h.bidirectional ? 2 : 1), h.batch_size,
                         h.hidden_size};
  }
  Tensor dx(xshape, y.device());
  Tensor dhx(states_shape, y.device());
  Tensor dcx(states_shape, y.device());

  dx.SetValue(0.0f);
  dhx.SetValue(0.0f);
  dcx.SetValue(0.0f);
  h.workspace.SetValue(0.0f);

  dx.device()->Exec(
      [dx, dhx, dcx, y, dy, dhy, dcy, &W, hx, cx, seq_lengths,
       &h](Context *ctx) {
        cudnnRNNDataDescriptor_t yDesc, dyDesc, dxDesc;
        init_data_desc(yDesc,
                       h.bidirectional ? h.hidden_size * 2 : h.hidden_size,
                       seq_lengths, h);
        init_data_desc(dyDesc,
                       h.bidirectional ? h.hidden_size * 2 : h.hidden_size,
                       seq_lengths, h);
        init_data_desc(dxDesc, h.feature_size, seq_lengths, h);

        /* other tensors desc*/
        cudnnTensorDescriptor_t hxDesc, cxDesc, dhxDesc, dcxDesc, dhyDesc,
            dcyDesc;
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

        CUDNN_CHECK(cudnnRNNBackwardDataEx(
            ctx->cudnn_handle, h.rnnDesc, yDesc, yptr, dyDesc, dyptr, NULL,
            NULL, dhyDesc, dhyptr, dcyDesc, dcyptr, h.wDesc, Wptr, hxDesc,
            hxptr, cxDesc, cxptr, dxDesc, dxptr, dhxDesc, dhxptr, dcxDesc,
            dcxptr, NULL, NULL, wsptr, h.workspace_size_bytes, rsptr,
            h.reserve_size_bytes));
      },
      {y.block(), dy.block(), dhy.block(), dcy.block(), hx.block(), cx.block(),
       W.block()},
      {dx.block(), dhx.block(), dcx.block(), h.workspace.block(),
       h.reserve_space.block()});
  return {dx, dhx, dcx};
}

Tensor GpuRNNBackwardWEx(const Tensor &x, const Tensor &hx, const Tensor &y,
                         const Tensor &seq_lengths, CudnnRNNHandle &h) {
  Tensor dW(Shape{h.weights_size}, x.device());
  dW.SetValue(0.0f);

  dW.device()->Exec(
      [dW, x, hx, y, seq_lengths, &h](Context *ctx) {
        cudnnRNNDataDescriptor_t xDesc, yDesc;
        init_data_desc(xDesc, h.feature_size, seq_lengths, h);
        init_data_desc(yDesc,
                       h.bidirectional ? h.hidden_size * 2 : h.hidden_size,
                       seq_lengths, h);

        /* other tensor desc */
        cudnnTensorDescriptor_t hxDesc;
        init_hc_Desc(hxDesc, h);

        auto xptr = x.block()->data();
        auto hxptr = hx.block()->data();
        auto yptr = y.block()->data();
        auto dWptr = dW.block()->mutable_data();
        auto wsptr = h.workspace.block()->mutable_data();
        auto rsptr = h.reserve_space.block()->mutable_data();

        CUDNN_CHECK(cudnnRNNBackwardWeightsEx(
            ctx->cudnn_handle, h.rnnDesc, xDesc, xptr, hxDesc, hxptr, yDesc,
            yptr, wsptr, h.workspace_size_bytes, h.dwDesc, dWptr, rsptr,
            h.reserve_size_bytes));
      },
      {x.block(), y.block(), hx.block()},
      {dW.block(), h.workspace.block(), h.reserve_space.block()});
  return dW;
}

#endif  // USE_CUDNN
}  // namespace singa
