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

#include "singa/neuralnet/input_layer.h"
#include "singa/utils/context.h"
#include "singa/utils/singleton.h"

namespace singa {

StoreInputLayer::~StoreInputLayer() {
  if (store_ != nullptr) {
    delete store_;
  }
}

void StoreInputLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(conf, srclayers);
  batchsize_ = conf.store_conf().batchsize();
  if (conf.partition_dim() == 0) {
    batchsize_ /= conf.num_partitions();
  }
}

void StoreInputLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  string key, val;
  if (store_ == nullptr) {
    store_ = io::OpenStore(layer_conf_.store_conf().backend(),
        layer_conf_.store_conf().path(),
        io::kRead);
    if (layer_conf_.store_conf().random_skip() > 0) {
      std::uniform_int_distribution<int>
        distribution(0, layer_conf_.store_conf().random_skip());
      auto generator = Singleton<Context>::Instance()->rand_generator(
          std::this_thread::get_id());
      random_skip_ = distribution(*generator);
    }

    while (random_skip_ > 0) {
      if (!store_->Read(&key, &val)) {
        store_->SeekToFirst();
        CHECK(store_->Read(&key, &val));
      }
      random_skip_--;
    }
  }
  for (int k = 0; k < batchsize_; k++) {
    if (!store_->Read(&key, &val)) {
      store_->SeekToFirst();
      CHECK(store_->Read(&key, &val));
    }
    // TODO(wangwei) random skip and shuffle among this mini-batch
    Parse(k, flag, key, val);
  }
}

void SingleLabelRecordLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  StoreInputLayer::Setup(conf, srclayers);

  vector<int> shape {batchsize_};
  for (int s : conf.store_conf().shape())
    shape.push_back(s);
  data_.Reshape(shape);
  aux_data_.resize(batchsize_);
}
void SingleLabelRecordLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  StoreInputLayer::ComputeFeature(flag, srclayers);

  auto& store_conf = layer_conf_.store_conf();
  if (store_conf.has_mean_file() && mean_.count() == 0) {
    mean_.Reshape(vector<int>{data_.count() / batchsize_});
    LoadRecord(store_conf.backend(), store_conf.mean_file(), &mean_);
  } else if (store_conf.has_mean_value() && mean_.count() == 0) {
    mean_.Reshape(vector<int>{data_.count() / batchsize_});
    for (int i = 0; i < data_.count() / batchsize_; i++)
      mean_.mutable_cpu_data()[i] = store_conf.mean_value();
  }
  if (store_conf.has_std_file() && std_.count() == 0) {
    std_.Reshape(vector<int>{data_.count() / batchsize_});
    LoadRecord(store_conf.backend(), store_conf.std_file(), &std_);
    // TODO(wangwei) check std[i] != 0
  } else if (store_conf.has_std_value() && std_.count() == 0) {
    std_.Reshape(vector<int>{data_.count() / batchsize_});
    CHECK_NE(store_conf.std_value(), 0);
    for (int i = 0; i < data_.count() / batchsize_; i++)
      std_.mutable_cpu_data()[i] = store_conf.std_value();
  }

  if (mean_.count()) {
    const float* mean = mean_.cpu_data();
    for (int k = 0; k < batchsize_; k++) {
      float* dptr = data_.mutable_cpu_data() + k * mean_.count();
      for (int i = 0; i < mean_.count(); i++) {
        dptr[i] -= mean[i];
      }
    }
  }
  if (std_.count()) {
    const float* std = std_.cpu_data();
    for (int k = 0; k < batchsize_; k++) {
      float* dptr = data_.mutable_cpu_data() + k * std_.count();
      for (int i = 0; i < std_.count(); i++) {
        dptr[i] /= std[i];
      }
    }
  }
}


}  // namespace singa
