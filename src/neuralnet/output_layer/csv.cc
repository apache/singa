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
#include "singa/neuralnet/output_layer.h"

namespace singa {

void CSVOutputLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  OutputLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
}

void CSVOutputLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if (store_ == nullptr) {
    string backend = "textfile";
    const auto& conf = layer_conf_.store_conf();
    if (conf.has_backend())
      backend = conf.has_backend();
    store_ = io::OpenStore(backend, conf.path(), io::kCreate);
  }
  const auto& data = srclayers.at(0)->data(this);
  const auto& label = srclayers.at(0)->aux_data();
  int batchsize = data.shape()[0];
  CHECK_GT(batchsize, 0);
  int dim = data.count() / batchsize;
  if (label.size())
    CHECK_EQ(label.size(), batchsize);
  CHECK_GT(dim, 0);
  for (int k = 0; k < batchsize; k++) {
    std::ostringstream record;
    if (label.size())
      record << std::to_string(label[k]) << ",";
    auto* dptr = data.cpu_data() + k * dim;
    for (int i = 0; i < dim - 1; i++)
      record << std::to_string(dptr[i]) << ",";
    record << std::to_string(dptr[dim - 1]);
    store_->Write(std::to_string(inst_++), record.str());
  }
  store_->Flush();
}
}  // namespace singa
