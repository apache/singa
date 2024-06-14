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
#if CUDNN_VERSION >= 5005
#include <chrono>

#include "./cudnn_utils.h"
#include "singa/utils/logging.h"

namespace singa {
RegisterLayerClass(cudnn_rnn, CudnnRNN);
CudnnRNN::~CudnnRNN() {
  if (weight_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc_));
  if (dropout_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  if (rnn_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc_));
  if (hx_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(hx_desc_));
  if (hy_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(hy_desc_));
  if (cx_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(cx_desc_));
  if (cy_desc_ != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(cy_desc_));
  if (dhx_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dhx_desc_));
  if (dhy_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dhy_desc_));
  if (dcx_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dcx_desc_));
  if (dcy_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dcy_desc_));
  DestroyIODescriptors();
}

void CudnnRNN::ToDevice(std::shared_ptr<Device> device) {
  RNN::ToDevice(device);
  workspace_.ToDevice(device);
  reserve_space_.ToDevice(device);
  dropout_state_.ToDevice(device);
}

void CudnnRNN::DestroyIODescriptors() {
  if (x_descs_ != nullptr) {
    for (size_t i = 0; i < max_length_; i++) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_descs_[i]));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(dx_descs_[i]));
    }
    delete[] x_descs_;
    delete[] dx_descs_;
  }
  if (y_descs_ != nullptr) {
    for (size_t i = 0; i < max_length_; i++) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_descs_[i]));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy_descs_[i]));
    }
    delete[] y_descs_;
    delete[] dy_descs_;
  }
}

void CudnnRNN::UpdateIODescriptors(size_t len, const vector<Tensor> &inputs) {
  bool reset = false;
  if (max_length_ < len) {
    DestroyIODescriptors();
    max_length_ = len;
    x_descs_ = new cudnnTensorDescriptor_t[len];
    dx_descs_ = new cudnnTensorDescriptor_t[len];
    y_descs_ = new cudnnTensorDescriptor_t[len];
    dy_descs_ = new cudnnTensorDescriptor_t[len];
    for (size_t i = 0; i < len; i++) {
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_descs_[i]));
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_descs_[i]));
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_descs_[i]));
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_descs_[i]));
    }
    reset = true;
  }

  for (size_t i = 0; i < len; i++) {
    CHECK_EQ(inputs[i].shape(1), input_size_);
    if (inputs[i].shape(0) != batch_size_ || reset) {
      int d[3] = {1, 1, 1}, s[3] = {1, 1, 1};
      d[0] = static_cast<int>(inputs[i].shape(0));
      CHECK_GT(d[0], 0);
      d[1] = static_cast<int>(inputs[i].shape(1));
      s[0] = d[1] * d[2];
      s[1] = d[2];
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_descs_[i], dtype_, 3, d, s));
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(dx_descs_[i], dtype_, 3, d, s));

      d[0] = static_cast<int>(inputs[i].shape(0));
      d[1] = static_cast<int>(hidden_size_ * num_directions_);
      s[0] = d[1] * d[2];
      s[1] = d[2];
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(y_descs_[i], dtype_, 3, d, s));
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(dy_descs_[i], dtype_, 3, d, s));
    }
  }
}

// must be called after setting IO descriptors
void CudnnRNN::SetRNNDescriptor(shared_ptr<Device> dev) {
  auto ctx = dev->context(0);
  CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
  size_t state_size;
  CUDNN_CHECK(cudnnDropoutGetStatesSize(ctx->cudnn_handle, &state_size));
  dropout_state_ = Tensor(Shape{state_size}, dev, kChar);
  CUDNN_CHECK(cudnnSetDropoutDescriptor(
      dropout_desc_, ctx->cudnn_handle, 1 - dropout_,  // keep probability
      dropout_state_.block()->mutable_data(), state_size, seed_));

  CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc_));
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
  if (input_mode_ == "skip") input_mode = CUDNN_SKIP_INPUT;

  cudnnDirectionMode_t direction = CUDNN_UNIDIRECTIONAL;
  if (direction_ == "bidirectional") direction = CUDNN_BIDIRECTIONAL;

  cudnnRNNMode_t rnn_mode = CUDNN_LSTM;
  if (rnn_mode_ == "relu")
    rnn_mode = CUDNN_RNN_RELU;
  else if (rnn_mode_ == "tanh")
    rnn_mode = CUDNN_RNN_TANH;
  else if (rnn_mode_ == "gru")
    rnn_mode = CUDNN_GRU;
#if CUDNN_MAJOR <= 5
  CUDNN_CHECK(cudnnSetRNNDescriptor(rnn_desc_, hidden_size_, num_stacks_,
                                    dropout_desc_, input_mode, direction,
                                    rnn_mode, dtype_));
#else
  CUDNN_CHECK(cudnnSetRNNDescriptor(
      ctx->cudnn_handle, rnn_desc_, hidden_size_, num_stacks_, dropout_desc_,
      input_mode, direction, rnn_mode, CUDNN_RNN_ALGO_STANDARD, dtype_));
#endif
  size_t weight_size;
  CUDNN_CHECK(cudnnGetRNNParamsSize(ctx->cudnn_handle, rnn_desc_, x_descs_[0],
                                    &weight_size, dtype_));
  // check the size manually calculated
  CHECK_EQ(weight_size, weight_.Size() * sizeof(float));
  int filter_dim[3] = {static_cast<int>(weight_size), 1, 1};
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&weight_desc_));
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(weight_desc_, dtype_,
                                         CUDNN_TENSOR_NCHW, 3, filter_dim));
}

void CudnnRNN::ResetHiddenAndCellDescriptors(size_t batch_size) {
  if (batch_size_ == 0) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dcx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cy_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dcy_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dhx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hy_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dhy_desc_));
  }

  int dim[3] = {1, 1, 1};
  dim[0] = static_cast<int>(num_stacks_ * num_directions_);
  dim[1] = static_cast<int>(batch_size);
  dim[2] = static_cast<int>(hidden_size_);
  int stride[3] = {1, 1, 1};
  stride[0] = dim[1] * dim[2];
  stride[1] = dim[2];
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(hx_desc_, dtype_, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(dhx_desc_, dtype_, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(hy_desc_, dtype_, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(dhy_desc_, dtype_, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(cx_desc_, dtype_, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(dcx_desc_, dtype_, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(cy_desc_, dtype_, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(dcy_desc_, dtype_, 3, dim, stride));
}

void CudnnRNN::UpdateSpaces(size_t seq_length, shared_ptr<Device> dev) {
  size_t count;
  auto ctx = dev->context(0);
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(ctx->cudnn_handle, rnn_desc_, seq_length,
                                       x_descs_, &count));
  if (workspace_.Size() != count) {
    workspace_ = Tensor(Shape{count}, dev, kChar);
    // workspace_.SetValue(0);
  }

  CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(ctx->cudnn_handle, rnn_desc_,
                                             seq_length, x_descs_, &count));
  if (reserve_space_.Size() != count) {
    reserve_space_ = Tensor(Shape{count}, dev, kChar);
    // reserve_space_.SetValue(0);
  }
}

void CudnnRNN::UpdateStates(size_t num_x, const vector<Tensor> &inputs) {
  UpdateIODescriptors(num_x, inputs);
  size_t new_batch_size = inputs.at(0).shape(0);
  if (batch_size_ != new_batch_size)
    ResetHiddenAndCellDescriptors(new_batch_size);
  if (rnn_desc_ == nullptr) SetRNNDescriptor(inputs.at(0).device());
  UpdateSpaces(num_x, inputs.at(0).device());
  batch_size_ = new_batch_size;
  seq_length_ = num_x;
}

Tensor CudnnRNN::MergeInputs(size_t num, const vector<Tensor> &in) {
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

vector<Tensor> CudnnRNN::SplitOutput(size_t num, size_t dim,
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

const vector<Tensor> CudnnRNN::Forward(int flag, const vector<Tensor> &inputs) {
  DataType dtype = inputs.at(0).data_type();
  auto dev = inputs.at(0).device();

  // copy input data into a block of contiguous memory
  // hx (and cx) is at the end of inputs
  CHECK_GT(inputs.size(), 1u + has_cell_);
  size_t num_x = inputs.size() - has_cell_ - 1;
  Tensor input = MergeInputs(num_x, inputs);
  // LOG(INFO) << "input size " << input.Size() << " value " << input.L1();

  if (rnn_desc_ != nullptr)
    CHECK_EQ(dtype_, GetCudnnDataType(dtype))
        << "Cannot change cudnn data type during training from " << dtype_
        << " to " << GetCudnnDataType(dtype);
  else
    dtype_ = GetCudnnDataType(dtype);

  UpdateStates(num_x, inputs);
  // CheckFowardShapes();

  Shape outshape{input.Size() * hidden_size_ / input_size_ * num_directions_};
  Tensor output(outshape, dev, dtype);
  // LOG(INFO) << "output size " << output.Size();
  Tensor hx = inputs.at(num_x);
  Shape state_shape{num_stacks_ * num_directions_, batch_size_, hidden_size_};
  Tensor hy(state_shape, dev, dtype);
  Tensor cy, cx;
  if (has_cell_) {
    cx = inputs.at(num_x + 1);
    cy.ResetLike(hy);
  }

  int did = input.device()->id();
  CHECK_EQ(did, output.device()->id());
  if (hx.Size()) {
    CHECK_EQ(did, hx.device()->id());
    CHECK_EQ(hx.device()->lang(), kCuda);
  }
  if (cx.Size()) {
    CHECK_EQ(did, cx.device()->id());
    CHECK_EQ(cx.device()->lang(), kCuda);
  }
  CHECK_EQ(did, weight_.device()->id());
  CHECK_EQ(did, workspace_.device()->id());
  CHECK_EQ(input.device()->lang(), kCuda);
  CHECK_EQ(output.device()->lang(), kCuda);
  CHECK_EQ(weight_.device()->lang(), kCuda);
  CHECK_EQ(workspace_.device()->lang(), kCuda);

  // LOG(INFO) << "hidden size " << hy.Size();
  // LOG(INFO) << "weight size " << weight_.Size() << " value " << weight_.L1();
  Block *inb = input.block(), *outb = output.block(),
        *wb = this->weight_.block(), *hxb = hx.block(), *cxb = cx.block(),
        *hyb = hy.block(), *cyb = cy.block(),
        *wspace = this->workspace_.block(),
        *rspace = this->reserve_space_.block();
  if (flag & kTrain) {
    CHECK_EQ(reserve_space_.device()->lang(), kCuda);
    CHECK_EQ(did, reserve_space_.device()->id());
    dev->Exec(
        [inb, outb, wb, hxb, cxb, hyb, cyb, wspace, rspace,
         this](Context *ctx) {
          // clang-format off
      cudnnRNNForwardTraining(
        ctx->cudnn_handle,
        this->rnn_desc_,
        this->seq_length_,
        this->x_descs_, inb->data(),
        this->hx_desc_, hxb == nullptr ? nullptr : hxb->data(),
        this->cx_desc_, cxb == nullptr ? nullptr : cxb->data(),
        this->weight_desc_, wb->data(),
        this->y_descs_, outb->mutable_data(),
        this->hy_desc_, hyb->mutable_data(),
        this->cy_desc_, cyb == nullptr ? nullptr : cyb->mutable_data(),
        wspace->mutable_data(),
        this->workspace_.Size(), rspace->mutable_data(),
        this->reserve_space_.Size());
          // clang-format on
        },
        {inb, wb, hxb, cxb}, {outb, hyb, cyb, wspace, rspace});
    buf_.push(input);
    buf_.push(output);
    buf_.push(hx);
    buf_.push(cx);
  } else {
    dev->Exec(
        [inb, outb, wb, hxb, cxb, hyb, cyb, wspace, this](Context *ctx) {
          // clang-format off
      cudnnRNNForwardInference(
        ctx->cudnn_handle,
        this->rnn_desc_,
        this->seq_length_,
        this->x_descs_, inb->data(),
        this->hx_desc_, hxb == nullptr ? nullptr : hxb->data(),
        this->cx_desc_, cxb == nullptr ? nullptr : cxb->data(),
        this->weight_desc_, wb->data(),
        this->y_descs_, outb->mutable_data(),
        this->hy_desc_, hyb->mutable_data(),
        this->cy_desc_, cyb == nullptr ? nullptr : cyb->mutable_data(),
        wspace->mutable_data(), this->workspace_.Size());
          // clang-format on
        },
        {inb, wb, hxb, cxb}, {outb, hyb, cyb, wspace});
  }
  auto outputs =
      SplitOutput(num_x, hidden_size_ * num_directions_, inputs, output);
  outputs.push_back(hy);
  if (has_cell_) outputs.push_back(cy);
  return outputs;
}

// TODO(wangwei) check Tensor device to be on cuda?
const std::pair<vector<Tensor>, vector<Tensor>> CudnnRNN::Backward(
    int flag, const vector<Tensor> &grads) {
  // dhy (and dcy) is at last
  const Tensor cx = buf_.top();  // cannot use const Tensor& due to pop()
  buf_.pop();
  const Tensor hx = buf_.top();
  buf_.pop();
  const Tensor y = buf_.top();
  buf_.pop();
  const Tensor x = buf_.top();
  buf_.pop();

  auto dev = y.device();
  auto dtype = y.data_type();

  CHECK_GT(grads.size(), 1u + has_cell_);
  size_t num_dy = grads.size() - has_cell_ - 1;
  CHECK_EQ(num_dy, seq_length_);
  const Tensor dy = MergeInputs(num_dy, grads);
  CHECK_EQ(dy.Size(), y.Size());
  const Tensor dhy = grads.at(num_dy);
  Tensor dcy;
  if (has_cell_) dcy = grads.at(num_dy + 1);

  Shape xshape{y.Size() * input_size_ / hidden_size_ / num_directions_};
  Tensor dx(xshape, dev, dtype);
  Tensor dw(weight_.shape(), dev, dtype);
  Shape state_shape{num_stacks_ * num_directions_, batch_size_, hidden_size_};
  Tensor dhx(state_shape, dev, dtype);
  Tensor dcx;
  if (has_cell_) dcx.ResetLike(dhx);
  dw.SetValue(0.0f);
  Block *yb = y.block(), *dyb = dy.block(), *dhyb = dhy.block(),
        *dcyb = dcy.block(), *xb = x.block(), *cxb = cx.block(),
        *wb = weight_.block(), *dwb = dw.block(), *hxb = hx.block(),
        *dxb = dx.block(), *dhxb = dhx.block(), *dcxb = dcx.block(),
        *wspace = workspace_.block(), *rspace = reserve_space_.block();

  y.device()->Exec(
      [yb, dyb, dhyb, dcyb, xb, cxb, wb, dwb, hxb, dxb, dhxb, dcxb, wspace,
       rspace, this](Context *ctx) {
        // clang-format off
    cudnnRNNBackwardData(
      ctx->cudnn_handle,
      this->rnn_desc_,
      this->seq_length_,
      this->y_descs_, yb->data(),
      this->dy_descs_, dyb->data(),
      this->dhy_desc_, dhyb == nullptr ? nullptr : dhyb->data(),
      this->dcy_desc_, dcyb == nullptr ? nullptr : dcyb->data(),
      this->weight_desc_, wb->data(),
      this->hx_desc_, hxb == nullptr ? nullptr : hxb->data(),
      this->cx_desc_, cxb == nullptr ? nullptr : cxb->data(),
      this->dx_descs_, dxb->mutable_data(),
      this->dhx_desc_, dhxb->mutable_data(),
      this->dcx_desc_, dcxb == nullptr ? nullptr : dcxb->mutable_data(),
      wspace->mutable_data(), this->workspace_.Size(),
      rspace->mutable_data(), this->reserve_space_.Size());
    cudnnRNNBackwardWeights(
      ctx->cudnn_handle,
      this->rnn_desc_,
      this->seq_length_,
      this->x_descs_, xb->data(),
      this->hx_desc_, hxb == nullptr ? nullptr : hxb->data(),
      this->y_descs_, yb->data(),
      wspace->data(), this->workspace_.Size(),
      this->dweight_desc_, dwb->mutable_data(),
      rspace->data(), this->reserve_space_.Size());
        // clang-format on
      },
      {yb, dyb, dhyb, dcyb, xb, wb, wspace, rspace},
      {dxb, dwb, dhxb, dcxb, wspace, rspace});

  vector<Tensor> param_grad{dw};
  auto data_grads = SplitOutput(num_dy, input_size_, grads, dx);
  data_grads.push_back(dhx);
  if (has_cell_) data_grads.push_back(dcx);
  return std::make_pair(data_grads, param_grad);
}

}  // namespace singa
#endif  // CUDNN_VERSION >= 5005
#endif  // USE_CUDNN
