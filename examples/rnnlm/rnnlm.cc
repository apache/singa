#include "rnnlm.h"
#include "rnnlm.pb.h"
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"

namespace singa {
using namespace mshadow;
using mshadow::cpu;

using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Tensor;


inline Tensor<cpu, 2> RTensor2(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}

inline Tensor<cpu, 1> RTensor1(Blob<float>* blob) {
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}

/*******EmbeddingLayer**************/
EmbeddingLayer::~EmbeddingLayer() {
  delete embed_;
}

void EmbeddingLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  int max_window = srclayers_[0]->data(this).shape()[0];
  word_dim_ = proto.GetExtension(embedding_conf).word_dim();
  data_.Reshape(vector<int>{max_window, word_dim_});
  grad_.ReshapeLike(data_);
  vocab_size_ = proto.GetExtension(embedding_conf).vocab_size();
  embed_ = Param::Create(proto.param(0));
  embed_->Setup(vector<int>{vocab_size_, word_dim_});
}

void EmbeddingLayer::ComputeFeature(int flag, Metric* perf) {
  window_ = static_cast<RNNLayer*>(srclayers_[0])->window();
  auto words = RTensor2(&data_);
  auto embed = RTensor2(embed_->mutable_data());
  auto word_idx = RTensor1(srclayers_[0]->mutable_data(this));

  for (int t = 0; t < window_; t++) {
    int idx = static_cast<int>(word_idx[t]);
    CHECK_GE(idx, 0);
    CHECK_LT(idx, vocab_size_);
    Copy(words[t], embed[idx]);
  }
}

void EmbeddingLayer::ComputeGradient(int flag, Metric* perf) {
  auto grad = RTensor2(&grad_);
  auto gembed = RTensor2(embed_->mutable_grad());
  auto word_idx = RTensor1(srclayers_[0]->mutable_data(this));
  gembed = 0;
  for (int t = 0; t < window_; t++) {
    int idx = static_cast<int>(word_idx[t]);
    Copy(gembed[idx], grad[t]);
  }
}
/***********HiddenLayer**********/
HiddenLayer::~HiddenLayer() {
  delete weight_;
}

void HiddenLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& innerproductData = srclayers_[0]->data(this);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(srclayers_[0]->grad(this));
  int word_dim = data_.shape()[1];
  weight_ = Param::Create(proto.param(0));
  weight_->Setup(std::vector<int>{word_dim, word_dim});
}

// hid[t] = sigmoid(hid[t-1] * W + src[t])
void HiddenLayer::ComputeFeature(int flag, Metric* perf) {
  window_ = static_cast<RNNLayer*>(srclayers_[0])->window();
  auto data = RTensor2(&data_);
  auto src = RTensor2(srclayers_[0]->mutable_data(this));
  auto weight = RTensor2(weight_->mutable_data());
  for (int t = 0; t < window_; t++) {  // Skip the 1st component
    if (t == 0) {
      data[t] = expr::F<op::sigmoid>(src[t]);
    } else {
      data[t] = dot(data[t - 1], weight);
      data[t] += src[t];
      data[t] = expr::F<op::sigmoid>(data[t]);
    }
  }
}

void HiddenLayer::ComputeGradient(int flag, Metric* perf) {
  auto data = RTensor2(&data_);
  auto grad = RTensor2(&grad_);
  auto weight = RTensor2(weight_->mutable_data());
  auto gweight = RTensor2(weight_->mutable_grad());
  auto gsrc = RTensor2(srclayers_[0]->mutable_grad(this));
  gweight = 0;
  TensorContainer<cpu, 1> tmp(Shape1(data_.shape()[1]));
  // Check!!
  for (int t = window_ - 1; t >= 0; t--) {
    if (t < window_ - 1) {
      tmp = dot(grad[t + 1], weight.T());
      grad[t] += tmp;
    }
    grad[t] = expr::F<op::sigmoid_grad>(data[t])* grad[t];
  }
  gweight = dot(data.Slice(0, window_-1).T(), grad.Slice(1, window_));
  Copy(gsrc, grad);
}

/*********** 1-Implementation for OutputLayer **********/
OutputLayer::~OutputLayer() {
  delete word_weight_;
  delete class_weight_;
}

void OutputLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 2);
  const auto& src = srclayers_[0]->data(this);
  int max_window = src.shape()[0];
  int vdim = src.count() / max_window;   // Dimension of input
  int vocab_size = proto.GetExtension(output_conf).vocab_size();
  int nclass = proto.GetExtension(output_conf).nclass();
  word_weight_ = Param::Create(proto.param(0));
  word_weight_->Setup(vector<int>{vocab_size, vdim});
  class_weight_ = Param::Create(proto.param(0));
  class_weight_->Setup(vector<int>{nclass, vdim});

  pword_.resize(max_window);
  pclass_.Reshape(vector<int>{max_window, nclass});
}

void OutputLayer::ComputeFeature(int flag, Metric* perf) {
  window_ = static_cast<RNNLayer*>(srclayers_[0])->window();
  auto pclass = RTensor2(&pclass_);
  auto src = RTensor2(srclayers_[0]->mutable_data(this));
  auto word_weight = RTensor2(word_weight_->mutable_data());
  auto class_weight = RTensor2(class_weight_->mutable_data());
  const float * label = srclayers_[1]->data(this).cpu_data();

  float loss = 0.f, ppl =0.f;
  for (int t = 0; t < window_; t++) {
    int start = static_cast<int>(label[t * 4 + 0]);
    int end = static_cast<int>(label[t * 4 + 1]);

    auto wordWeight = word_weight.Slice(start, end);
    pword_[t].Reshape(vector<int>{end-start});
    auto pword = RTensor1(&pword_[t]);
    pword = dot(src[t], wordWeight.T());
    Softmax(pword, pword);

    pclass[t] = dot(src[t], class_weight.T());
    Softmax(pclass[t], pclass[t]);

    int wid = static_cast<int>(label[t * 4 + 2]);
    int cid = static_cast<int>(label[t * 4 + 3]);
    loss += -log(std::max(pword[wid - start] * pclass[t][cid], FLT_MIN));
    ppl += log10(std::max(pword[wid - start] * pclass[t][cid], FLT_MIN));
  }

  perf->Add("loss", loss, window_);
  perf->Add("ppl before exp", ppl, window_);
}

void OutputLayer::ComputeGradient(int flag, Metric* perf) {
  auto pclass = RTensor2(&pclass_);
  auto src = RTensor2(srclayers_[0]->mutable_data(this));
  auto gsrc = RTensor2(srclayers_[0]->mutable_grad(this));
  auto word_weight = RTensor2(word_weight_->mutable_data());
  auto gword_weight = RTensor2(word_weight_->mutable_grad());
  auto class_weight = RTensor2(class_weight_->mutable_data());
  auto gclass_weight = RTensor2(class_weight_->mutable_grad());
  const float * label = srclayers_[1]->data(this).cpu_data();
  gclass_weight = 0;
  gword_weight = 0;
  for (int t = 0; t < window_; t++) {
    int start = static_cast<int>(label[t * 4 + 0]);
    int end = static_cast<int>(label[t * 4 + 1]);
    int wid = static_cast<int>(label[t * 4 + 2]);
    int cid = static_cast<int>(label[t * 4 + 3]);
    auto pword = RTensor1(&pword_[t]);

    // gL/gclass_act
    pclass[t][cid] -= 1.0;
    // gL/gword_act
    pword[wid] -= 1.0;

    // gL/gword_weight
    gword_weight.Slice(start, end) += dot(pword.FlatTo2D().T(), src[t].FlatTo2D());
    // gL/gclass_weight
    gclass_weight += dot(pclass[t].FlatTo2D().T(), src[t].FlatTo2D());

    gsrc[t] = dot(pword, word_weight.Slice(start, end));
    gsrc[t] += dot(pclass[t], class_weight);
  }
}
}
