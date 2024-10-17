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

#include "singa/model/feed_forward_net.h"

#include "singa/model/initializer.h"
#include "singa/utils/channel.h"
#include "singa/utils/logging.h"
namespace singa {

FeedForwardNet::~FeedForwardNet() {}

std::shared_ptr<Layer> FeedForwardNet::Add(std::shared_ptr<Layer> layer) {
  layers_.push_back(layer);
  return layer;
}

std::shared_ptr<Layer> FeedForwardNet::Add(const LayerConf& conf,
                                           const Shape* sample_shape) {
  std::shared_ptr<Layer> layer(CreateLayer(conf.type()));
  CHECK(conf.has_name()) << "Must set layer name";
  if (sample_shape == nullptr)
    layer->Setup(layers_.back()->GetOutputSampleShape(), conf);
  else
    layer->Setup(*sample_shape, conf);
  Add(layer);
  LOG(INFO) << layer->name() << VecToStr(layer->GetOutputSampleShape());
  return layer;
}

const vector<string> FeedForwardNet::GetParamNames() const {
  vector<string> names;
  for (auto layer : layers_)
    for (const auto name : layer->param_names()) names.push_back(name);
  return names;
}
const vector<Tensor> FeedForwardNet::GetParamValues() const {
  vector<Tensor> values;
  for (auto layer : layers_)
    for (const auto value : layer->param_values()) values.push_back(value);
  return values;
}

const vector<ParamSpec> FeedForwardNet::GetParamSpecs() const {
  vector<ParamSpec> specs;
  for (auto layer : layers_)
    for (const auto spec : layer->param_specs()) specs.push_back(spec);
  return specs;
}

void FeedForwardNet::Compile(bool shuffle, Optimizer* opt, Loss* loss,
                             Metric* metric) {
  std::shared_ptr<Updater> updater = std::make_shared<Updater>(opt);
  Compile(shuffle, true, updater, loss, metric);
}

void FeedForwardNet::Compile(bool shuffle, bool to_register,
                             std::shared_ptr<Updater> updater, Loss* loss,
                             Metric* metric) {
  shuffle_ = shuffle;
  bool train = (updater != nullptr) && (loss != nullptr);
  bool test = metric != nullptr;
  CHECK(train || test) << "Must set updater and loss, or set metric";
  updater_ = updater;
  loss_ = loss;
  metric_ = metric;
  const auto specs = GetParamSpecs();
  auto params = GetParamValues();
  CHECK_EQ(specs.size(), params.size());
  for (size_t k = 0; k < specs.size(); k++) {
    if (to_register) {
      updater_->Register(specs[k].name(), specs[k]);
    }
    auto init = CreateInitializer(specs[k].filler());
    init->Fill(params[k]);
    LOG(INFO) << specs[k].name() << " : " << params[k].L1();
  }
}

void FeedForwardNet::ToDevice(std::shared_ptr<Device> device) {
  for (auto layer : layers_) layer->ToDevice(device);
  /*
  opt_->ToDevice(device);
  loss_->ToDevice(device);
  metric_->ToDevice(device);
  */
}

FeedForwardNet FeedForwardNet::Clone(std::shared_ptr<Device> device) {
  FeedForwardNet net;
  /*
  for (auto layer: layers_)
    net.layers_.push_back(layer->CloneTo(device));
  if (opt_ != nullptr)
    net.opt_ = opt_->CloneTo(device);
  if (loss_ != nullptr)
    net.loss_ = loss_.CloneTo(device);
  if (metric_ != nullptr)
    net.metric_ = metric_->CloneTo(device);
  net.shuffle_ = shuffle_;
  net.device_ = device;
  net.dtype_ = dtype;
  */
  return net;
}

void FeedForwardNet::AsType(DataType dtype) {
  LOG(FATAL) << "FeedForwardNet::AsType not implemented";
}

void FeedForwardNet::Train(size_t batchsize, int nb_epoch, const Tensor& x,
                           const Tensor& y, float val_split) {
  CHECK_EQ(x.shape(0), y.shape(0)) << "Diff num of sampels in x and y";
  size_t num_train = (size_t)(x.shape(0) * val_split);
  if (val_split == 0.0f) {
    Tensor dummy;
    Train(batchsize, nb_epoch, x, y, dummy, dummy);
  } else {
    const Tensor train_x = CopyRows(x, 0, num_train);
    const Tensor train_y = CopyRows(y, 0, num_train);
    const Tensor test_x = CopyRows(x, num_train, x.shape(0));
    const Tensor test_y = CopyRows(y, num_train, y.shape(0));
    Train(batchsize, nb_epoch, train_x, train_y, test_x, test_y);
  }
}

void FeedForwardNet::Train(size_t batchsize, int nb_epoch, const Tensor& x,
                           const Tensor& y, const Tensor& val_x,
                           const Tensor& val_y) {
  CHECK_EQ(x.shape(0), y.shape(0)) << "Diff num of sampels in x and y";
  int num_extra_samples = (int)x.shape(0) % batchsize;
  if (num_extra_samples != 0)
    LOG(WARNING) << "Pls set batchsize to make num_total_samples "
                 << "% batchsize == 0. Otherwise, the last "
                 << num_extra_samples << " samples would not be used";
  Channel* train_ch = GetChannel("train_perf");
  train_ch->EnableDestStderr(true);
  Channel* val_ch = GetChannel("val_perf");
  val_ch->EnableDestStderr(true);
  std::vector<size_t> index;
  for (size_t i = 0; i < x.shape(0) / batchsize; i++) index.push_back(i);
  for (int epoch = 0; epoch < nb_epoch; epoch++) {
    if (shuffle_) std::random_shuffle(index.begin(), index.end());
    float loss = 0.0f, metric = 0.0f;
    size_t b = 0;
    for (; b < x.shape(0) / batchsize; b++) {
      size_t idx = index[b];
      const Tensor bx = CopyRows(x, idx * batchsize, (idx + 1) * batchsize);
      const Tensor by = CopyRows(y, idx * batchsize, (idx + 1) * batchsize);
      const auto ret = TrainOnBatch(epoch, bx, by);
      loss += ret.first;
      metric += ret.second;
    }
    if (val_x.Size() == 0) continue;
    loss /= b;
    metric /= b;
    train_ch->Send(
        "Epoch " + std::to_string(epoch) +
        ", training loss = " + std::to_string(loss) +
        ", accuracy = " + std::to_string(metric) + ", lr = " +
        std::to_string(updater_->GetOptimizer()->GetLearningRate(epoch)));
    if (val_x.Size() && val_y.Size()) {
      const auto val_perf = Evaluate(val_x, val_y, batchsize);
      val_ch->Send(
          "Epoch " + std::to_string(epoch) +
          ", val loss = " + std::to_string(Sum(val_perf.first) / val_y.Size()) +
          ", metric = " + std::to_string(Sum(val_perf.second) / val_y.Size()));
    }
  }
}

const std::pair<float, float> FeedForwardNet::TrainOnBatch(int epoch,
                                                           const Tensor& x,
                                                           const Tensor& y) {
  int flag = kTrain;
  const Tensor fea = Forward(flag, x);
  float loss = loss_->Evaluate(flag, fea, y);
  float metric = metric_->Evaluate(fea, y);
  const Tensor grad = loss_->Backward();
  auto grads = Backward(kTrain, grad / static_cast<float>(x.shape(0)));
  auto names = GetParamNames();
  auto values = GetParamValues();
  for (size_t k = 0; k < grads.size(); k++) {
    updater_->Apply(epoch, names[k], grads[k], values.at(k));
  }
  return std::make_pair(loss, metric);
}

const Tensor FeedForwardNet::Forward(int flag, const Tensor& data) {
  Tensor input = data, output;
  // LOG(INFO) << data.L1();
  for (auto layer : layers_) {
    output = layer->Forward(flag, input);
    // LOG(INFO) << layer->name() << ": " << output.L2();
    input = output;
  }
  return output;
}

const vector<Tensor> FeedForwardNet::Backward(int flag, const Tensor& grad) {
  vector<Tensor> param_grads;
  std::stack<Tensor> buf;
  Tensor tmp = grad;
  for (int i = (int)layers_.size() - 1; i >= 0; i--) {
    // LOG(INFO) << layers_.at(i)->name() << " : " << tmp.L1();
    auto ret = layers_.at(i)->Backward(flag, tmp);
    tmp = ret.first;
    if (ret.second.size()) {
      for (int k = (int)ret.second.size() - 1; k >= 0; k--) {
        buf.push(ret.second[k]);
        // LOG(INFO) <<  "      " << buf.top().L1();
      }
    }
  }
  while (!buf.empty()) {
    param_grads.push_back(buf.top());
    buf.pop();
  }
  return param_grads;
}

std::pair<Tensor, Tensor> FeedForwardNet::Evaluate(const Tensor& x,
                                                   const Tensor& y,
                                                   size_t batchsize) {
  CHECK_EQ(x.shape(0), y.shape(0)) << "Diff num of sampels in x and y";
  CHECK_GE(x.shape(0), batchsize);
  int num_extra_samples = (int)x.shape(0) % batchsize;
  Tensor loss(Shape{x.shape(0)}), metric(Shape{x.shape(0)});
  for (size_t b = 0; b < x.shape(0) / batchsize; b++) {
    int start = (int)(b * batchsize), end = (int)(start + batchsize);
    const Tensor bx = CopyRows(x, start, end);
    const Tensor by = CopyRows(y, start, end);
    const auto ret = EvaluateOnBatch(bx, by);
    CopyDataToFrom(&loss, ret.first, batchsize, start, 0);
    CopyDataToFrom(&metric, ret.second, batchsize, start, 0);
  }
  {
    int start = (int)(x.shape(0) - batchsize), end = (int)x.shape(0);
    const Tensor bx = CopyRows(x, start, end);
    const Tensor by = CopyRows(y, start, end);
    const auto ret = EvaluateOnBatch(bx, by);
    int dst_offset = (int)(x.shape(0) - num_extra_samples);
    int src_offset = (int)(batchsize - num_extra_samples);
    CopyDataToFrom(&loss, ret.first, num_extra_samples, dst_offset, src_offset);
    CopyDataToFrom(&metric, ret.second, num_extra_samples, dst_offset,
                   src_offset);
  }
  return std::make_pair(loss, metric);
}

std::pair<Tensor, Tensor> FeedForwardNet::EvaluateOnBatch(const Tensor& x,
                                                          const Tensor& y) {
  int flag = kEval;
  const Tensor fea = Forward(flag, x);
  const Tensor l = loss_->Forward(flag, fea, y);
  const Tensor m = metric_->Forward(fea, y);
  return std::make_pair(l, m);
}

const Tensor FeedForwardNet::Predict(const Tensor& x, size_t batchsize) {
  CHECK_GE(x.shape(0), batchsize);
  int num_extra_samples = (int)(x.shape(0) % batchsize);
  const auto outshape = layers_.back()->GetOutputSampleShape();
  Tensor y(Shape{x.shape(0), Product(outshape)}, x.device());
  for (size_t b = 0; b < x.shape(0) / batchsize; b++) {
    int start = (int)(b * batchsize), end = (int)(start + batchsize);
    const Tensor bx = CopyRows(x, start, end);
    CopyDataToFrom(&y, PredictOnBatch(bx), batchsize * y.shape(1),
                   start * y.shape(1), 0);
  }
  if (num_extra_samples > 0) {
    int start = (int)(x.shape(0) - batchsize), end = (int)(x.shape(0));
    const Tensor bx = CopyRows(x, start, end);
    CopyDataToFrom(&y, PredictOnBatch(bx), num_extra_samples * y.shape(1),
                   (x.shape(0) - num_extra_samples) * y.shape(1),
                   (batchsize - num_extra_samples) * y.shape(1));
  }
  return y;
}

const Tensor FeedForwardNet::PredictOnBatch(const Tensor& x) {
  return Forward(kEval, x);
}
}  // namespace singa
