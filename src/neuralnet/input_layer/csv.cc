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
#include "singa/utils/tokenizer.h"

namespace singa {

void CSVInputLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  SingleLabelRecordLayer::Setup(conf, srclayers);
  sep_ = conf.store_conf().separator();
}

void CSVInputLayer::LoadRecord(const string& backend,
    const string&path, Blob<float>* to) {
  io::Store* store = io::OpenStore(backend, path, io::kRead);
  string key, val;
  CHECK(store->Read(&key, &val));
  float* ptr = to->mutable_cpu_data();
  Tokenizer t(val, sep_);
  string x;
  for (int i = 0; i< to->count(); i++) {
    t >> x;
    ptr[i] = stof(x);
  }
  CHECK(!t.Valid());
  delete store;
}

bool CSVInputLayer::Parse(int k, int flag, const string& key,
    const string& value) {
  float* ptr = data_.mutable_cpu_data() + k * data_.count() / batchsize_;
  Tokenizer t(value, sep_);
  string x;
  // parse label if not deploy phase and has_label is set.
  if ((flag & kDeploy) == 0 && layer_conf_.store_conf().has_label()) {
    t >> x;
    aux_data_[k] = stoi(x);
  }
  for (int i = 0; i< data_.count() / batchsize_; i++) {
    t >> x;
    ptr[i] = stof(x);
  }
  CHECK(!t.Valid());
  return true;
}

}  // namespace singa
