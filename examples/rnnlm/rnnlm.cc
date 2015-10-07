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
#include "./rnnlm.h"

#include <string>
#include <algorithm>
#include "mshadow/tensor.h"
#include "mshadow/tensor_expr.h"
#include "mshadow/cxxnet_op.h"
#include "./rnnlm.pb.h"

namespace rnnlm {
using std::vector;
using std::string;

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


/*******DataLayer**************/
DataLayer::~DataLayer() {
  if (store_ != nullptr)
    delete store_;
}

void DataLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
  RNNLayer::Setup(conf, srclayers);
  string key;
  max_window_ = conf.GetExtension(data_conf).max_window();
  data_.Reshape(vector<int>{max_window_ + 1, 4});
  window_ = 0;
}

void SetInst(int k, const WordRecord& word, Blob<float>* to) {
  float* dptr = to->mutable_cpu_data() + k * 4;
  dptr[0] = static_cast<float>(word.word_index());
  dptr[1] = static_cast<float>(word.class_index());
  dptr[2] = static_cast<float>(word.class_start());
  dptr[3] = static_cast<float>(word.class_end());
}

void ShiftInst(int from, int to,  Blob<float>* data) {
  const float* f = data->cpu_data() + from * 4;
  float* t = data->mutable_cpu_data() + to * 4;
  // hard code the feature dim to be 4;
  t[0] = f[0]; t[1] = f[1]; t[2] = f[2]; t[3] = f[3];
}

void DataLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  string key, value;
  WordRecord word;
  if (store_ == nullptr) {
    store_ = singa::io::OpenStore(
        layer_conf_.GetExtension(data_conf).backend(),
        layer_conf_.GetExtension(data_conf).path(),
        singa::io::kRead);
    store_->Read(&key, &value);
    word.ParseFromString(value);
    SetInst(0, word, &data_);
  }
  ShiftInst(window_, 0, &data_);
  window_ = max_window_;
  for (int i = 1; i <= max_window_; i++) {
    if (!store_->Read(&key, &value)) {
      store_->SeekToFirst();
      CHECK(store_->Read(&key, &value));
    }
    word.ParseFromString(value);
    SetInst(i, word, &data_);
    if (word.word_index() == 0) {
      window_ = i;
      break;
    }
  }
}

/*******LabelLayer**************
void LabelLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  RNNLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  int max_window = dynamic_cast<DataLayer*>(srclayers[0])->max_window();
  data_.Reshape(vector<int>{max_window, 4});
}

void LabelLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  const auto& records = dynamic_cast<DataLayer*>(srclayers[0])->records();
  float *label = data_.mutable_cpu_data();
  window_ = dynamic_cast<RNNLayer*>(srclayers[0])->window();
  for (int i = 0; i < window_; i++) {
    WordRecord wordrecord = records[i + 1].GetExtension(word);
    label[4 * i + 0] = wordrecord.class_start();
    label[4 * i + 1] = wordrecord.class_end();
    label[4 * i + 2] = wordrecord.word_index();
    label[4 * i + 3] = wordrecord.class_index();
  }
}
*/

/*******EmbeddingLayer**************/
EmbeddingLayer::~EmbeddingLayer() {
  delete embed_;
}

void EmbeddingLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  RNNLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  int max_window = srclayers[0]->data(this).shape()[0];
  word_dim_ = conf.GetExtension(embedding_conf).word_dim();
  data_.Reshape(vector<int>{max_window, word_dim_});
  grad_.ReshapeLike(data_);
  vocab_size_ = conf.GetExtension(embedding_conf).vocab_size();
  embed_ = Param::Create(conf.param(0));
  embed_->Setup(vector<int>{vocab_size_, word_dim_});
}

void EmbeddingLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[0]);
  window_ = datalayer->window();
  auto words = RTensor2(&data_);
  auto embed = RTensor2(embed_->mutable_data());

  const float* idxptr = datalayer->data(this).cpu_data();
  for (int t = 0; t < window_; t++) {
    int idx = static_cast<int>(idxptr[t * 4]);
    CHECK_GE(idx, 0);
    CHECK_LT(idx, vocab_size_);
    Copy(words[t], embed[idx]);
  }
}

void EmbeddingLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  auto grad = RTensor2(&grad_);
  auto gembed = RTensor2(embed_->mutable_grad());
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[0]);
  gembed = 0;
  const float* idxptr = datalayer->data(this).cpu_data();
  for (int t = 0; t < window_; t++) {
    int idx = static_cast<int>(idxptr[t * 4]);
    Copy(gembed[idx], grad[t]);
  }
}
/***********HiddenLayer**********/
HiddenLayer::~HiddenLayer() {
  delete weight_;
}

void HiddenLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  RNNLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  const auto& innerproductData = srclayers[0]->data(this);
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(srclayers[0]->grad(this));
  int word_dim = data_.shape()[1];
  weight_ = Param::Create(conf.param(0));
  weight_->Setup(std::vector<int>{word_dim, word_dim});
}

// hid[t] = sigmoid(hid[t-1] * W + src[t])
void HiddenLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  window_ = dynamic_cast<RNNLayer*>(srclayers[0])->window();
  auto data = RTensor2(&data_);
  auto src = RTensor2(srclayers[0]->mutable_data(this));
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

void HiddenLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  auto data = RTensor2(&data_);
  auto grad = RTensor2(&grad_);
  auto weight = RTensor2(weight_->mutable_data());
  auto gweight = RTensor2(weight_->mutable_grad());
  auto gsrc = RTensor2(srclayers[0]->mutable_grad(this));
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

/*********** Implementation for LossLayer **********/
LossLayer::~LossLayer() {
  delete word_weight_;
  delete class_weight_;
}

void LossLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
  RNNLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 2);
  const auto& src = srclayers[0]->data(this);
  int max_window = src.shape()[0];
  int vdim = src.count() / max_window;   // Dimension of input
  int vocab_size = conf.GetExtension(loss_conf).vocab_size();
  int nclass = conf.GetExtension(loss_conf).nclass();
  word_weight_ = Param::Create(conf.param(0));
  word_weight_->Setup(vector<int>{vocab_size, vdim});
  class_weight_ = Param::Create(conf.param(1));
  class_weight_->Setup(vector<int>{nclass, vdim});

  pword_.resize(max_window);
  pclass_.Reshape(vector<int>{max_window, nclass});
}

void LossLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  window_ = dynamic_cast<RNNLayer*>(srclayers[0])->window();
  auto pclass = RTensor2(&pclass_);
  auto src = RTensor2(srclayers[0]->mutable_data(this));
  auto word_weight = RTensor2(word_weight_->mutable_data());
  auto class_weight = RTensor2(class_weight_->mutable_data());
  const float * label = srclayers[1]->data(this).cpu_data();

  float loss = 0.f, ppl = 0.f;
  for (int t = 0; t < window_; t++) {
    // label is the next word
    int start = static_cast<int>(label[(t + 1) * 4 + 2]);
    int end = static_cast<int>(label[(t + 1) * 4 + 3]);

    auto wordWeight = word_weight.Slice(start, end);
    CHECK_GT(end, start);
    pword_[t].Reshape(std::vector<int>{end-start});
    auto pword = RTensor1(&pword_[t]);
    pword = dot(src[t], wordWeight.T());
    Softmax(pword, pword);

    pclass[t] = dot(src[t], class_weight.T());
    Softmax(pclass[t], pclass[t]);

    int wid = static_cast<int>(label[(t + 1) * 4 + 0]);
    int cid = static_cast<int>(label[(t + 1) * 4 + 1]);
    CHECK_GT(end, wid);
    CHECK_GE(wid, start);
    loss_ += -log(std::max(pword[wid - start] * pclass[t][cid], FLT_MIN));
    ppl_ += log10(std::max(pword[wid - start] * pclass[t][cid], FLT_MIN));
  }
  num_ += window_;
}

void LossLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  auto pclass = RTensor2(&pclass_);
  auto src = RTensor2(srclayers[0]->mutable_data(this));
  auto gsrc = RTensor2(srclayers[0]->mutable_grad(this));
  auto word_weight = RTensor2(word_weight_->mutable_data());
  auto gword_weight = RTensor2(word_weight_->mutable_grad());
  auto class_weight = RTensor2(class_weight_->mutable_data());
  auto gclass_weight = RTensor2(class_weight_->mutable_grad());
  const float * label = srclayers[1]->data(this).cpu_data();
  gclass_weight = 0;
  gword_weight = 0;
  for (int t = 0; t < window_; t++) {
    int start = static_cast<int>(label[(t + 1) * 4 + 2]);
    int end = static_cast<int>(label[(t + 1) * 4 + 3]);
    int wid = static_cast<int>(label[(t + 1) * 4 + 0]);
    int cid = static_cast<int>(label[(t + 1) * 4 + 1]);
    auto pword = RTensor1(&pword_[t]);
    CHECK_GT(end, wid);
    CHECK_GE(wid, start);

    // gL/gclass_act
    pclass[t][cid] -= 1.0;
    // gL/gword_act
    pword[wid - start] -= 1.0;

    // gL/gword_weight
    gword_weight.Slice(start, end) += dot(pword.FlatTo2D().T(),
                                          src[t].FlatTo2D());
    // gL/gclass_weight
    gclass_weight += dot(pclass[t].FlatTo2D().T(),
                         src[t].FlatTo2D());

    gsrc[t] = dot(pword, word_weight.Slice(start, end));
    gsrc[t] += dot(pclass[t], class_weight);
  }
}

const std::string LossLayer::ToString(bool debug, int flag) {
  float loss = loss_ / num_;
  float ppl = exp10(- ppl_ / num_);
  loss_ = 0;
  num_ = 0;
  ppl_ = 0;
  return "loss = " + std::to_string(loss) + ", ppl = " + std::to_string(ppl);
}
}   // end of namespace rnnlm
