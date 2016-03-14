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
namespace singa {

using std::string;
using std::vector;

void RecordInputLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  SingleLabelRecordLayer::Setup(conf, srclayers);
  encoded_ = conf.store_conf().encoded();
}

void RecordInputLayer::LoadRecord(const string& backend,
    const string& path, Blob<float>* to) {
  io::Store* store = io::OpenStore(backend, path, io::kRead);
  string key, val;
  CHECK(store->Read(&key, &val));
  RecordProto image;
  image.ParseFromString(val);
  CHECK_EQ(to->count(), image.data_size());
  float* ptr = to->mutable_cpu_data();
  for (int i = 0; i< to->count(); i++)
    ptr[i] = image.data(i);
  delete store;
}

bool RecordInputLayer::Parse(int k, int flag, const string& key,
    const string& value) {
  RecordProto image;
  image.ParseFromString(value);
  int size = data_.count() / batchsize_;
  if (image.data_size()) {
    CHECK_EQ(size, image.data_size());
    float* ptr = data_.mutable_cpu_data() + k * size;
    for (int i = 0; i< size; i++)
      ptr[i] = image.data(i);
  } else if (image.pixel().size()) {
    CHECK_EQ(size, image.pixel().size());
    float* ptr = data_.mutable_cpu_data() + k * size;
    string pixel = image.pixel();
    for (int i = 0; i < size; i++)
      ptr[i] =  static_cast<float>(static_cast<uint8_t>(pixel[i]));
  } else {
    LOG(ERROR) << "not pixel nor pixel";
  }
  if ((flag & kDeploy) == 0) {  // deploy mode does not have label
    aux_data_.at(k) = image.label();
  }
  return true;
}

}  // namespace singa
