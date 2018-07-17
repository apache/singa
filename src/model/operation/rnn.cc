#include "./rnn.h"

namespace singa {

RNNHandle::RNNHandle(const size_t Input_size, const size_t Hidden_size, const size_t Num_stacks,
                     const std::string Rnn_mode, const float Dropout, const bool bidirectional) {

  input_size_ = Input_size;
  CHECK_GT(input_size_, 0u);
  hidden_size_ = Hidden_size;
  CHECK_GT(hidden_size_, 0u);
  num_stacks_ = Num_stacks;
  CHECK_GT(num_stacks_, 0u);
  dropout_ = Dropout;  // drop probability
  CHECK_GE(dropout_, 0);

  if (bidirectional)
    num_directions_ = 2;
  else
    num_directions_ = 1;

  rnn_mode_ = Rnn_mode;
  if (rnn_mode_ == "lstm") {
    has_cell_ = true;
  } else if (rnn_mode_ != "relu" && rnn_mode_ != "tanh" && rnn_mode_ != "gru") {
    LOG(FATAL) << "RNN memory unit (mode) of " << rnn_mode_
               << " is not supported Please use 'relu', 'tanh', 'lstm' and 'gru'";
  }
  // the first constant (4) is the size of float
  // the second constant (2, 8, 6) is the number of sets of params
  int mult = 1;
  if (rnn_mode_ == "relu" || rnn_mode_ == "tanh")
    mult *= 1;
  else if (rnn_mode_ == "lstm")
    mult *= 4;
  else if (rnn_mode_ == "gru")
    mult *= 3;
  if (bidirectional)
    mult *= 2;

  weight_size = 0;
  for (size_t i = 0; i < num_stacks_; i++) {
    size_t dim = hidden_size_ * (input_size_ +  hidden_size_ + 2);
    if (i > 0)
      dim = hidden_size_ * (hidden_size_ +  hidden_size_ + 2);
    weight_size += mult * dim;
  }
};

#ifdef USE_CUDNN

CudnnRNNHandle::CudnnRNNHandle(const vector<Tensor> &inputs, const size_t Input_size, const size_t Hidden_size, const size_t Num_stacks,
                               const std::string Rnn_mode, const float Dropout, const bool bidirectional):
  RNNHandle(Input_size, Hidden_size, Num_stacks, Rnn_mode, Dropout, bidirectional) {

  CHECK_GT(inputs.size(), 1u + has_cell_);
  size_t num_x = inputs.size() - has_cell_ - 1;

  DataType dtype = inputs.at(0).data_type();
  if (rnn_desc_ != nullptr)
    CHECK_EQ(dtype_, GetCudnnDataType(dtype))
        << "Cannot change cudnn data type during training from " << dtype_
        << " to " << GetCudnnDataType(dtype);
  else
    dtype_ = GetCudnnDataType(dtype);

  UpdateStates(num_x, inputs);
};

void CudnnRNNHandle::UpdateStates(size_t num_x, const vector<Tensor> &inputs) {
  UpdateIODescriptors(num_x, inputs);
  size_t new_batch_size = inputs.at(0).shape(0);
  if (batch_size_ != new_batch_size)
    ResetHiddenAndCellDescriptors(new_batch_size);
  if (rnn_desc_ == nullptr)
    SetRNNDescriptor(inputs.at(0).device());
  UpdateSpaces(num_x, inputs.at(0).device());
  batch_size_ = new_batch_size;
  seq_length_ = num_x;
};

void CudnnRNNHandle::DestroyIODescriptors() {
  if (x_descs_ != nullptr) {
    for (size_t i = 0; i < max_length_; i++) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_descs_[i]));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(dx_descs_[i]));
    }
    delete [] x_descs_;
    delete [] dx_descs_;
  }
  if (y_descs_ != nullptr) {
    for (size_t i = 0; i < max_length_; i++) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_descs_[i]));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy_descs_[i]));
    }
    delete [] y_descs_;
    delete [] dy_descs_;
  }
};

void CudnnRNNHandle::UpdateIODescriptors(size_t len, const vector<Tensor> &inputs) {
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
};

void CudnnRNNHandle::ResetHiddenAndCellDescriptors(size_t batch_size) {
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
};

void CudnnRNNHandle::SetRNNDescriptor(shared_ptr<Device> dev) {
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
  //if (input_mode_ == "skip")
  //input_mode = CUDNN_SKIP_INPUT;

  cudnnDirectionMode_t direction = CUDNN_UNIDIRECTIONAL;
  if (num_directions_ == 2)
    direction = CUDNN_BIDIRECTIONAL;

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
  CUDNN_CHECK(cudnnSetRNNDescriptor(ctx->cudnn_handle, rnn_desc_, hidden_size_, num_stacks_,
                                    dropout_desc_, input_mode, direction,
                                    rnn_mode, CUDNN_RNN_ALGO_STANDARD, dtype_));
#endif
  size_t weight_size_;
  CUDNN_CHECK(cudnnGetRNNParamsSize(ctx->cudnn_handle, rnn_desc_, x_descs_[0],
                                    &weight_size_, dtype_));
  // check the size manually calculated
  CHECK_EQ(weight_size_, weight_size * sizeof(float));
  int filter_dim[3] = {static_cast<int>(weight_size_), 1, 1};
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&weight_desc_));
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(weight_desc_, dtype_,
                                         CUDNN_TENSOR_NCHW, 3, filter_dim));
};

void CudnnRNNHandle::UpdateSpaces(size_t seq_length, shared_ptr<Device> dev) {
  size_t count;
  auto ctx = dev->context(0);
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(ctx->cudnn_handle, rnn_desc_,
                                       seq_length, x_descs_, &count));
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

Tensor MergeInputs(size_t num, const vector<Tensor> &in) {
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
};

vector<Tensor> SplitOutput(size_t num, size_t dim,
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
};

std::vector<std::vector<Tensor>> GpuRNNForwardTraining(const CudnnRNNHandle &crh, const vector<Tensor> &inputs, const Tensor &W) {
  DataType dtype = inputs.at(0).data_type();
  auto dev = inputs.at(0).device();

  CHECK_GT(inputs.size(), 1u + crh.has_cell_);
  size_t num_x = inputs.size() - crh.has_cell_ - 1;
  Tensor input = MergeInputs(num_x, inputs);

  Shape outshape{input.Size() * crh.hidden_size_ / crh.input_size_ * crh.num_directions_};
  Tensor output(outshape, dev, dtype);
  // LOG(INFO) << "output size " << output.Size();
  Tensor hx = inputs.at(num_x);
  Shape state_shape{crh.num_stacks_ * crh.num_directions_, crh.batch_size_, crh.hidden_size_};
  Tensor hy(state_shape, dev, dtype);

  Tensor cy, cx;
  if (crh.has_cell_) {
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
  CHECK_EQ(did, W.device()->id());
  CHECK_EQ(did, crh.workspace_.device()->id());
  CHECK_EQ(input.device()->lang(), kCuda);
  CHECK_EQ(output.device()->lang(), kCuda);
  CHECK_EQ(W.device()->lang(), kCuda);
  CHECK_EQ(crh.workspace_.device()->lang(), kCuda);
  CHECK_EQ(crh.reserve_space_.device()->lang(), kCuda);
  CHECK_EQ(did, crh.reserve_space_.device()->id());

  Block *inb = input.block(), *outb = output.block(),
         *wb = W.block(), *hxb = hx.block(), *cxb = cx.block(),
          *hyb = hy.block(), *cyb = cy.block(),
           *wspace = crh.workspace_.block(),
            *rspace = crh.reserve_space_.block();

  dev->Exec(
  [inb, outb, wb, hxb, cxb, hyb, cyb, wspace, rspace, &crh](Context * ctx) {
    // clang-format off
    cudnnRNNForwardTraining(
      ctx->cudnn_handle,
      crh.rnn_desc_,
      crh.seq_length_,
      crh.x_descs_, inb->data(),
      crh.hx_desc_, hxb == nullptr ? nullptr : hxb->data(),
      crh.cx_desc_, cxb == nullptr ? nullptr : cxb->data(),
      crh.weight_desc_, wb->data(),
      crh.y_descs_, outb->mutable_data(),
      crh.hy_desc_, hyb->mutable_data(),
      crh.cy_desc_, cyb == nullptr ? nullptr : cyb->mutable_data(),
      wspace->mutable_data(),
      crh.workspace_.Size(), rspace->mutable_data(),
      crh.reserve_space_.Size());
    // clang-format on
  },
  {inb, wb, hxb, cxb}, {outb, hyb, cyb, wspace, rspace});

  auto outputs =
    SplitOutput(num_x, crh.hidden_size_ * crh.num_directions_, inputs, output);
  outputs.push_back(hy);
  if (crh.has_cell_) outputs.push_back(cy);

  std::vector<Tensor> cache;
  cache.push_back(input);
  cache.push_back(output);
  cache.push_back(hx);
  cache.push_back(cx);
  cache.push_back(W);

  return {outputs, cache};
};

std::vector<Tensor> GpuRNNForwardInference(const CudnnRNNHandle &crh, const vector<Tensor> &inputs, const Tensor &W) {
  DataType dtype = inputs.at(0).data_type();
  auto dev = inputs.at(0).device();

  CHECK_GT(inputs.size(), 1u + crh.has_cell_);
  size_t num_x = inputs.size() - crh.has_cell_ - 1;
  Tensor input = MergeInputs(num_x, inputs);

  Shape outshape{input.Size() * crh.hidden_size_ / crh.input_size_ * crh.num_directions_};
  Tensor output(outshape, dev, dtype);
  // LOG(INFO) << "output size " << output.Size();
  Tensor hx = inputs.at(num_x);
  Shape state_shape{crh.num_stacks_ * crh.num_directions_, crh.batch_size_, crh.hidden_size_};
  Tensor hy(state_shape, dev, dtype);

  Tensor cy, cx;
  if (crh.has_cell_) {
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
  CHECK_EQ(did, W.device()->id());
  CHECK_EQ(did, crh.workspace_.device()->id());
  CHECK_EQ(input.device()->lang(), kCuda);
  CHECK_EQ(output.device()->lang(), kCuda);
  CHECK_EQ(W.device()->lang(), kCuda);
  CHECK_EQ(crh.workspace_.device()->lang(), kCuda);

  Block *inb = input.block(), *outb = output.block(),
         *wb = W.block(), *hxb = hx.block(), *cxb = cx.block(),
          *hyb = hy.block(), *cyb = cy.block(),
           *wspace = crh.workspace_.block();

  dev->Exec([inb, outb, wb, hxb, cxb, hyb, cyb, wspace, &crh](Context * ctx) {
    // clang-format off
    cudnnRNNForwardInference(
      ctx->cudnn_handle,
      crh.rnn_desc_,
      crh.seq_length_,
      crh.x_descs_, inb->data(),
      crh.hx_desc_, hxb == nullptr ? nullptr : hxb->data(),
      crh.cx_desc_, cxb == nullptr ? nullptr : cxb->data(),
      crh.weight_desc_, wb->data(),
      crh.y_descs_, outb->mutable_data(),
      crh.hy_desc_, hyb->mutable_data(),
      crh.cy_desc_, cyb == nullptr ? nullptr : cyb->mutable_data(),
      wspace->mutable_data(), crh.workspace_.Size());
    // clang-format on
  }, {inb, wb, hxb, cxb}, {outb, hyb, cyb, wspace});

  auto outputs =
    SplitOutput(num_x, crh.hidden_size_ * crh.num_directions_, inputs, output);
  outputs.push_back(hy);
  if (crh.has_cell_) outputs.push_back(cy);

  return outputs;
};

std::pair<vector<Tensor>, Tensor> GpuRNNBackward(const CudnnRNNHandle &crh, const vector<Tensor> &grads, const vector<Tensor> &cache) {
  const Tensor x = cache[0];
  const Tensor y = cache[1];
  const Tensor hx = cache[2];
  const Tensor cx = cache[3];
  const Tensor W = cache[4];

  auto dev = y.device();
  auto dtype = y.data_type();

  CHECK_GT(grads.size(), 1u + crh.has_cell_);
  size_t num_dy = grads.size() - crh.has_cell_ - 1;
  CHECK_EQ(num_dy, crh.seq_length_);
  const Tensor dy = MergeInputs(num_dy, grads);
  CHECK_EQ(dy.Size(), y.Size());
  const Tensor dhy = grads.at(num_dy);
  Tensor dcy;
  if (crh.has_cell_)
    dcy = grads.at(num_dy + 1);

  Shape xshape{y.Size() * crh.input_size_ / crh.hidden_size_ / crh.num_directions_};
  Tensor dx(xshape, dev, dtype);
  Tensor dw(W.shape(), dev, dtype);
  Shape state_shape{crh.num_stacks_ * crh.num_directions_, crh.batch_size_, crh.hidden_size_};
  Tensor dhx(state_shape, dev, dtype);
  Tensor dcx;
  if (crh.has_cell_)
    dcx.ResetLike(dhx);
  dw.SetValue(0.0f);
  Block *yb = y.block(), *dyb = dy.block(), *dhyb = dhy.block(),
         *dcyb = dcy.block(), *xb = x.block(), *cxb = cx.block(),
          *wb = W.block(), *dwb = dw.block(), *hxb = hx.block(),
           *dxb = dx.block(), *dhxb = dhx.block(), *dcxb = dcx.block(),
            *wspace = crh.workspace_.block(), *rspace = crh.reserve_space_.block();

  y.device()->Exec(
    [yb, dyb, dhyb, dcyb, xb, cxb, wb, dwb, hxb, dxb, dhxb, dcxb, wspace,
  rspace, &crh](Context * ctx) {
    // clang-format off
    cudnnRNNBackwardData(
      ctx->cudnn_handle,
      crh.rnn_desc_,
      crh.seq_length_,
      crh.y_descs_, yb->data(),
      crh.dy_descs_, dyb->data(),
      crh.dhy_desc_, dhyb == nullptr ? nullptr : dhyb->data(),
      crh.dcy_desc_, dcyb == nullptr ? nullptr : dcyb->data(),
      crh.weight_desc_, wb->data(),
      crh.hx_desc_, hxb == nullptr ? nullptr : hxb->data(),
      crh.cx_desc_, cxb == nullptr ? nullptr : cxb->data(),
      crh.dx_descs_, dxb->mutable_data(),
      crh.dhx_desc_, dhxb->mutable_data(),
      crh.dcx_desc_, dcxb == nullptr ? nullptr : dcxb->mutable_data(),
      wspace->mutable_data(), crh.workspace_.Size(),
      rspace->mutable_data(), crh.reserve_space_.Size());
    cudnnRNNBackwardWeights(
      ctx->cudnn_handle,
      crh.rnn_desc_,
      crh.seq_length_,
      crh.x_descs_, xb->data(),
      crh.hx_desc_, hxb == nullptr ? nullptr : hxb->data(),
      crh.y_descs_, yb->data(),
      wspace->data(), crh.workspace_.Size(),
      crh.dweight_desc_, dwb->mutable_data(),
      rspace->data(), crh.reserve_space_.Size());
    // clang-format on
  },
  {yb, dyb, dhyb, dcyb, xb, wb, wspace, rspace},
  {dxb, dwb, dhxb, dcxb, wspace, rspace});

  auto data_grads = SplitOutput(num_dy, crh.input_size_, grads, dx);
  data_grads.push_back(dhx);
  if (crh.has_cell_)
    data_grads.push_back(dcx);

  return std::make_pair(data_grads, dw);
};

#endif  // USE_CUDNN

}  // namespace singa


