#include "./rnn.h"

namespace singa {

RNNHandle::RNNHandle(const Tensor &input, const size_t Input_size, const size_t Hidden_size, const size_t Num_stacks,
                     const std::string Rnn_mode, const float Dropout, const bool bidirectional, const size_t Weight_size) {

  CHECK_EQ(input.shape(2), Input_size);
  batch_size_ = input.shape(1);
  seq_length_= input.shape(0);

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
  weight_size= Weight_size;

};

#ifdef USE_CUDNN

CudnnRNNHandle::CudnnRNNHandle(const Tensor &input, const size_t Input_size, const size_t Hidden_size, const size_t Num_stacks,
                               const std::string Rnn_mode, const float Dropout, const bool bidirectional, const size_t Weight_size):
  RNNHandle(input, Input_size, Hidden_size, Num_stacks, Rnn_mode, Dropout, bidirectional, Weight_size) {

  DataType dtype = input.data_type();
  dtype_ = GetCudnnDataType(dtype);

  UpdateIODescriptors(input);
  ResetHiddenAndCellDescriptors();
  SetRNNDescriptor(input.device());
  UpdateSpaces(seq_length_, input.device());
};

CudnnRNNHandle::~CudnnRNNHandle() {
  if (weight_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc_));
  if (dropout_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  if (rnn_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc_));
  if (hx_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hx_desc_));
  if (hy_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hy_desc_));
  if (cx_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cx_desc_));
  if (cy_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cy_desc_));
  if (dhx_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dhx_desc_));
  if (dhy_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dhy_desc_));
  if (dcx_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dcx_desc_));
  if (dcy_desc_ != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dcy_desc_));
  DestroyIODescriptors();
};

void CudnnRNNHandle::DestroyIODescriptors() {
  if (x_descs_ != nullptr) {
    for (size_t i = 0; i < seq_length_; i++) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_descs_[i]));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(dx_descs_[i]));
    }
    delete [] x_descs_;
    delete [] dx_descs_;
  }
  if (y_descs_ != nullptr) {
    for (size_t i = 0; i < seq_length_; i++) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_descs_[i]));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy_descs_[i]));
    }
    delete [] y_descs_;
    delete [] dy_descs_;
  }
};


void CudnnRNNHandle::UpdateIODescriptors(const Tensor &input) {
    x_descs_ = new cudnnTensorDescriptor_t[seq_length_];
    dx_descs_ = new cudnnTensorDescriptor_t[seq_length_];
    y_descs_ = new cudnnTensorDescriptor_t[seq_length_];
    dy_descs_ = new cudnnTensorDescriptor_t[seq_length_];
    for (size_t i = 0; i < seq_length_; i++) {
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_descs_[i]));
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_descs_[i]));
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_descs_[i]));
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_descs_[i]));
    }

    for (size_t i = 0; i < seq_length_; i++) {
    CHECK_EQ(input.shape(2), input_size_);
      int d[3] = {1, 1, 1}, s[3] = {1, 1, 1};
      d[0] = static_cast<int>(batch_size_);
      CHECK_GT(d[0], 0);
      d[1] = static_cast<int>(input_size_);
      s[0] = d[1] * d[2];
      s[1] = d[2];
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_descs_[i], dtype_, 3, d, s));
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(dx_descs_[i], dtype_, 3, d, s));

      d[0] = static_cast<int>(batch_size_);
      d[1] = static_cast<int>(hidden_size_ * num_directions_);
      s[0] = d[1] * d[2];
      s[1] = d[2];
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(y_descs_[i], dtype_, 3, d, s));
      CUDNN_CHECK(cudnnSetTensorNdDescriptor(dy_descs_[i], dtype_, 3, d, s));
    }
};

void CudnnRNNHandle::ResetHiddenAndCellDescriptors() {
  if (cx_desc_ == nullptr)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cx_desc_));
  if (dcx_desc_ == nullptr)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dcx_desc_));
  if (cy_desc_ == nullptr)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cy_desc_));
  if (dcy_desc_ == nullptr)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dcy_desc_));
  if (hx_desc_ == nullptr)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hx_desc_));
  if (dhx_desc_ == nullptr)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dhx_desc_));
  if (hy_desc_ == nullptr)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hy_desc_));
  if (dhy_desc_ == nullptr)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dhy_desc_));

  int dim[3] = {1, 1, 1};
  dim[0] = static_cast<int>(num_stacks_ * num_directions_);
  dim[1] = static_cast<int>(batch_size_);
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
};

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

std::vector<Tensor> GpuRNNForwardTraining(const CudnnRNNHandle &crh, const Tensor &input, const Tensor &hx, const Tensor &cx, const Tensor &W) {
  DataType dtype = input.data_type();
  auto dev = input.device();


  Shape outshape{input.Size() * crh.hidden_size_ / crh.input_size_ * crh.num_directions_};
  Tensor output(outshape, dev, dtype);
  // LOG(INFO) << "output size " << output.Size();

  Shape state_shape{crh.num_stacks_ * crh.num_directions_, crh.batch_size_, crh.hidden_size_};
  Tensor hy(state_shape, dev, dtype);

  Tensor cy;
  if (crh.has_cell_) {
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

  return {output, hy, cy};
};

std::vector<Tensor> GpuRNNForwardInference(const CudnnRNNHandle &crh, const Tensor &input, const Tensor &hx, const Tensor &cx, const Tensor &W) {
  DataType dtype = input.data_type();
  auto dev = input.device();

  Shape outshape{input.Size() * crh.hidden_size_ / crh.input_size_ * crh.num_directions_};
  Tensor output(outshape, dev, dtype);
  // LOG(INFO) << "output size " << output.Size();

  Shape state_shape{crh.num_stacks_ * crh.num_directions_, crh.batch_size_, crh.hidden_size_};
  Tensor hy(state_shape, dev, dtype);

  Tensor cy;
  if (crh.has_cell_) {
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

  return {output, hy, cy};
};

std::vector<Tensor> GpuRNNBackward(const CudnnRNNHandle &crh, const Tensor &dY, const Tensor &dhy, const Tensor &dcy, const std::vector<Tensor> &cache) {
  const Tensor x = cache[0];
  const Tensor y = cache[1];
  const Tensor hx = cache[2];
  const Tensor cx = cache[3];
  const Tensor W = cache[4];

  auto dev = y.device();
  auto dtype = y.data_type();

  CHECK_EQ(dY.Size(), y.Size());

  Shape xshape{y.Size() * crh.input_size_ / crh.hidden_size_ / crh.num_directions_};
  Tensor dx(xshape, dev, dtype);
  
  Tensor dw(W.shape(), dev, dtype);
  
  Shape state_shape{crh.num_stacks_ * crh.num_directions_, crh.batch_size_, crh.hidden_size_};
  Tensor dhx(state_shape, dev, dtype);
  
  Tensor dcx;
  if (crh.has_cell_)
    dcx.ResetLike(dhx);
  
  dw.SetValue(0.0f);
  Block *yb = y.block(), *dyb = dY.block(), *dhyb = dhy.block(),
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

  return {dx, dhx, dcx, dw};
};

#endif  // USE_CUDNN

}  // namespace singa


