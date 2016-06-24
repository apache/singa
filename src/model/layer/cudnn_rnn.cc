/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "./cudnn_rnn.h"
#ifdef USE_CUDNN
#include <cudnn.h>
#include <chrono>
#include "./cudnn_utils.h"
#include "singa/utils/logging.h"

namespace singa {
CudnnRNN::~CudnnRNN() {
  if (weight_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc_));
  if (dropout_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  if (rnn_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc_));
  if (x_descs_ != nullptr)
    for (size_t i = 0; i < seqLength_; i++) 
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_descs_[i]));
  if (y_descs_ != nullptr)
    for (size_t i = 0; i < seqLength_; i++) 
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_descs_[i]));
  if (hx_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hx_desc_));
  if (hy_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hy_desc_));
  if (cx_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cx_desc_));
  if (cy_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cy_desc_));
}

void CudnnRNN::Setup(const Shape& in_sample, const LayerConf &conf) {
  RNN::Setup(in_sample, conf);
  RNNConf rnn_conf = conf.rnn_conf();
  // convert MB to bytes
  workspace_byte_limit_ = rnn_conf.workspace_byte_limit() << 20;
  inputMode_ = ToLowerCase(rnn_conf.inputmode());
  direction_ = ToLowerCase(rnn_conf.direction());
  mode_ = ToLowerCase(rnn_conf.mode());
  CHECK(inputMode_ == "cudnn_linear_input" || inputMode_ == "cudnn_skip_input")
      << "CudnnRNN only supports two inputmodes: cudnn_linear_input, "
         "cudnn_skip_input";
  CHECK(direction_ == "cudnn_undirectional" || direction_ == "cudnn_bidirectional")
      << "CudnnRNN only supports two directions: cudnn_undirectional, "
         "cudnn_bidirectional";
  CHECK(mode_ == "cudnn_rnn_relu" || mode_ == "cudnn_rnn_tanh" ||
        mode_ == "cudnn_lstm" || mode_ == "cudnn_gru")
      << "CudnnRNN only supports four modes: cudnn_rnn_relu, "
         "cudnn_rnn_tanh, cudnn_lstm and cudnn_gru";
  // the first constant (4) is the size of float
  // the second constant (2, 8, 6) is the number of sets of params
  if (mode_ == "cudnn_rnn_relu" || mode_ == "cudnn_rnn_tanh")
    weightSize_ = 4 * 2 * (hiddenSize_ * in_sample[2] + hiddenSize_);
  else if (mode_ == "cudnn_lstm")
    weightSize_ = 4 * 8 * (hiddenSize_ * in_sample[2] + hiddenSize_);
  else if (mode_ == "cudnn_gru")
    weightSize_ = 4 * 6 * (hiddenSize_ * in_sample[2] + hiddenSize_);
  if (direction_ == "cudnn_bidirectional")
    weightSize_ = weightSize_ * 2;
}

void CudnnRNN::ToDevice(std::shared_ptr<Device> device) {
  weight_.ToDevice(device);
  workspace_.ToDevice(device);
}

void CudnnRNN::InitCudnn(const Tensor &input) {
  CHECK(!has_init_cudnn_);
  DataType dtype = input.data_type();
  auto dev = input.device();
  Context *ctx = dev->context(0);
  seqLength_ = input.shape(0);
  size_t batchsize = input.shape(1); /*(seqLength, minibatch, inputSize) !!! */
  size_t inputSize = input.shape(2);
  size_t numDirections;
  if (direction_ == "cudnn_undirectional")
    numDirections = 1;
  else 
    numDirections = 2;
  x_descs_ = new cudnnTensorDescriptor_t[seqLength_];
  y_descs_ = new cudnnTensorDescriptor_t[seqLength_];
  for (size_t i = 0; i < seqLength_; i++)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_descs_[i]));
  for (size_t i = 0; i < seqLength_; i++)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_descs_[i]));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&hx_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&cx_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&hy_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&cy_desc_));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&weight_desc_));
  CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
  CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc_));


  int dimA[3] = {batchsize, inputSize, 1};
  int strideA[3] = {dimA[2] * dimA[1], dimA[2], 1};
  for (size_t i = 0; i < seqLength_; i++){
    dimA[0] = batchsize;
    dimA[1] = inputSize;
    dimA[2] = 1;
    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_descs_[i], GetCudnnDataType(dtype), 3,
                                         dimA, strideA));
    dimA[0] = batchsize;
    dimA[1] = hiddenSize_ * numDirections;
    dimA[2] = 1;
    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(y_descs_[i], GetCudnnDataType(dtype), 3,
                                         dimA, strideA));
  }
  
  dimA[0] = numLayers_;
  dimA[1] = batchsize;
  dimA[2] = hiddenSize_ * numDirections;
  strideA[0] = dimA[2] * dimA[1];
  strideA[1] = dimA[2];
  strideA[2] = 1;
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(hx_desc_, GetCudnnDataType(dtype), 3,
                                         dimA, strideA));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(cx_desc_, GetCudnnDataType(dtype), 3,
                                         dimA, strideA));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(hy_desc_, GetCudnnDataType(dtype), 3,
                                         dimA, strideA));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(cy_desc_, GetCudnnDataType(dtype), 3,
                                         dimA, strideA));

  size_t dropoutStatesSize;
  CUDNN_CHECK(cudnnDropoutGetStatesSize(ctx->cudnn_handle, &dropoutStatesSize));
  dropoutStates_ = Tensor(Shape{dropoutStatesSize}, dev, dtype);
  CUDNN_CHECK(cudnnSetDropoutDescriptor(dropout_desc_, ctx->cudnn_handle, dropout_, this->dropoutStates_.block()->mutable_data(), dropoutStatesSize, 0x01234567));
  
  cudnnRNNInputMode_t inputMode;
  cudnnDirectionMode_t direction;
  cudnnRNNMode_t mode;
  
  if (inputMode_ == "cudnn_linear_input" || inputMode_ == "cudnn_skip_input"){
    if (inputMode_ == "cudnn_linear_input")
      inputMode = CUDNN_LINEAR_INPUT;
    else if (inputMode_ == "cudnn_skip_input")
      inputMode = CUDNN_SKIP_INPUT;
  }
  if (direction_ == "cudnn_undirectional" || direction_ == "cudnn_bidirectional"){
    if (direction_ == "cudnn_undirectional")
      direction = CUDNN_UNIDIRECTIONAL;
    else if (direction_ == "cudnn_bidirectional")
      direction = CUDNN_BIDIRECTIONAL;
  }
  if (mode_ == "cudnn_rnn_relu" || mode_ == "cudnn_rnn_tanh" ||
        mode_ == "cudnn_lstm" || mode_ == "cudnn_gru"){
    if (mode_ == "cudnn_rnn_relu")
      mode = CUDNN_RNN_RELU;
    else if (mode_ == "cudnn_rnn_tanh")
      mode = CUDNN_RNN_TANH;
    else if (mode_ == "cudnn_lstm")
      mode = CUDNN_LSTM;
    else if (mode_ == "cudnn_gru")
      mode = CUDNN_GRU;
  }
  CUDNN_CHECK(cudnnSetRNNDescriptor(rnn_desc_, hiddenSize_, numLayers_, dropout_desc_, inputMode, direction, mode, GetCudnnDataType(dtype)));

  size_t weightSize;
  CUDNN_CHECK(cudnnGetRNNParamsSize(ctx->cudnn_handle, rnn_desc_, x_descs_[0], &weightSize, GetCudnnDataType(dtype)));
  CHECK_EQ(weightSize, weightSize_);

  int filterDimA[3] = {weightSize_, 1, 1};
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(weight_desc_, GetCudnnDataType(dtype), CUDNN_TENSOR_NCHW, 3, filterDimA));

  
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(ctx->cudnn_handle, rnn_desc_, seqLength_, x_descs_, &workspace_count_));
  workspace_ = Tensor(Shape{workspace_count_}, dev, dtype);

  CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(ctx->cudnn_handle, rnn_desc_, seqLength_, x_descs_, &ReserveSize_));
  reserve_ = Tensor(Shape{ReserveSize_}, dev, dtype);
  has_init_cudnn_ = true;
}

const vector<Tensor> CudnnRNN::Forward(int flag, const vector<Tensor>& inputs) {
  /*(seqLength, minibatch, inputSize) !!! */
  singa::Tensor input = inputs[0];
  singa::Tensor hx = inputs[1];
  singa:: Tensor cx = inputs[2];
  CHECK_EQ(input.device()->lang(), kCuda);
  CHECK_EQ(input.device()->lang(), this->weight_.device()->lang());
  CHECK_EQ(input.nDim(), 3u);
  vector<Tensor> data_output;
  if (flag & kTrain) buf_.push(input);  // buffer the input for backward
  size_t batchsize = input.shape(1); /*(seqLength, minibatch, inputSize) !!! */
  DataType dtype = input.data_type();
  auto dev = input.device();
 
  if (!has_init_cudnn_) InitCudnn(input);
 
    
  size_t numDirections;
  if (direction_ == "cudnn_undirectional")
    numDirections = 1;
  else 
    numDirections = 2;
  
  Shape shape{seqLength_, batchsize, hiddenSize_ * numDirections};
  Tensor output(shape, dev, dtype);
  Shape shape1{numLayers_, batchsize, hiddenSize_ * numDirections};
  Tensor hy(shape1, dev, dtype);
  Tensor cy(shape1, dev, dtype);
  
  output.device()->Exec([input, output, hx, hy, cx, cy, this](Context *ctx) {
    Block *inblock = input.block(), *outblock = output.block(),
          *wblock = this->weight_.block(), *hxblock = hx.block(), 
          *hyblock = hy.block(), *cxblock = cx.block(), 
          *cyblock = cy.block();
    cudnnRNNForwardTraining(
        ctx->cudnn_handle, this->rnn_desc_, seqLength_, this->x_descs_, 
        inblock->data(), this->hx_desc_, hxblock->data(), this->cx_desc_, 
        cxblock->data(), this->weight_desc_, wblock->data(), this->y_descs_, 
        outblock->mutable_data(), this->hy_desc_, hyblock->mutable_data(), 
        cy_desc_, cyblock->mutable_data(), this->workspace_.block()->mutable_data(), 
        this->workspace_count_ * sizeof(float), this->reserve_.block()->mutable_data(), 
        this->ReserveSize_ * sizeof(float));
}, {input.block(), weight_.block(), hx.block(), cx.block()}, 
   {output.block(), hy.block(), cy.block()}, workspace_.block());
  buf_.push(output);
  buf_.push(hx);
  buf_.push(hy);  // in order to assign shape to dhy
  buf_.push(cx);
  buf_.push(cy);  // in order to assign shape to dcy
  data_output.push_back(output);
  data_output.push_back(hy);
  data_output.push_back(cy);
  return data_output;
}

const std::pair<vector<Tensor>, vector<Tensor>> CudnnRNN::Backward(
    int flag, const vector<Tensor>& grads) {
  CHECK(has_init_cudnn_);
  singa::Tensor grad = grads[0];
  singa::Tensor dhy = grads[1];
  singa::Tensor dcy = grads[2];
  CHECK_EQ(grad.device()->lang(), kCuda);
  CHECK_EQ(grad.nDim(), 3u);
  CHECK(!buf_.empty());
  Tensor cy = buf_.top();
  buf_.pop();
  CHECK(!buf_.empty());
  Tensor cx = buf_.top();
  buf_.pop();
  CHECK(!buf_.empty());
  Tensor hy = buf_.top();
  buf_.pop();
  CHECK(!buf_.empty());
  Tensor hx = buf_.top();
  buf_.pop();
  CHECK(!buf_.empty());
  Tensor src_output = buf_.top();
  buf_.pop();
  CHECK(!buf_.empty());
  Tensor src_data = buf_.top();
  buf_.pop();
  vector<Tensor> param_grad;
  vector<Tensor> data_grad;
  Tensor dx;
  dx.ResetLike(src_data);
  Tensor dw;
  dw.ResetLike(weight_);
  Tensor dhx;
  dhx.ResetLike(hx);
  Tensor dcx;
  dcx.ResetLike(cx);


  dx.device()->Exec([grad, dw, src_data, src_output, hx, this](Context *ctx) {
    Block *inblock = src_data.block(), *srcoutblock = src_output.block(), 
          *dwblock = dw.block(), *hxblock = hx.block();
    cudnnRNNBackwardWeights(
        ctx->cudnn_handle, this->rnn_desc_, seqLength_, this->x_descs_, 
        inblock->data(), this->hx_desc_, hxblock->data(), this->y_descs_, 
        srcoutblock->data(), this->workspace_.block()->mutable_data(), 
        this->workspace_count_ * sizeof(float), this->weight_desc_, 
        dwblock->mutable_data(), this->reserve_.block()->mutable_data(), 
        this->ReserveSize_ * sizeof(float));
  }, {src_data.block(), hx.block(), src_output.block()}, {dw.block(), workspace_.block()}); 
  
  // LOG(ERROR) << "backward src";
  dx.device()->Exec([grad, dw, src_output, dx, cy, cx, hy, hx, dhy, dcy, dhx, dcx, this](Context *ctx) {
    Block *srcoutblock = src_output.block(), *wblock = this->weight_.block(), *dxblock = dx.block(),
          *dyblock = grad.block(), *cxblock = cx.block(), *hxblock = hx.block(), *dhyblock = dhy.block(),
          *dcyblock = dcy.block(), *dhxblock = dhx.block(), *dcxblock = dcx.block();
    cudnnRNNBackwardData(
        ctx->cudnn_handle, this->rnn_desc_, seqLength_, this->y_descs_, srcoutblock->data(), 
        this->y_descs_, dyblock->data(), this->hy_desc_, dhyblock->data(), 
        this->cy_desc_, dcyblock->data(), this->weight_desc_, wblock->data(), 
        this->hx_desc_, hxblock->data(), this->cx_desc_, cxblock->data(), 
        this->x_descs_, dxblock->mutable_data(), this->hx_desc_, dhxblock->mutable_data(), 
        this->cx_desc_, dcxblock->mutable_data(), this->workspace_.block()->mutable_data(), 
        this->workspace_count_ * sizeof(float), this->reserve_.block()->mutable_data(), 
        this->ReserveSize_ * sizeof(float));
  }, {hx.block(), src_output.block(), grad.block(), grad.block(), dhy.block(), dcy.block(), 
      this->weight_.block(), hx.block(), cx.block()}, 
     {dx.block(), dhx.block(), dcx.block(), reserve_.block(), workspace_.block()}); 
  param_grad.push_back(dw);
  data_grad.push_back(dx);
  data_grad.push_back(dhx);
  data_grad.push_back(dcx);
  return std::make_pair(data_grad, param_grad);
}

}  // namespace singa
#endif  // USE_CUDNN
