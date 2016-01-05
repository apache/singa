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

#include <algorithm>
#include <iostream>
#include <fstream>
#include "singa/neuralnet/output_layer.h"

namespace singa {

void CharRNNOutputLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  OutputLayer::Setup(proto, srclayers);
  std::ifstream fin;
  const string path = proto.char_rnn_conf().vocab_path();
  fin.open(path);
  CHECK(fin.is_open()) << "Can't open vocab_path = " << path;
  std::stringstream stream;
  stream << fin.rdbuf();
  vocab_ = stream.str();
  fin.close();
}

void CharRNNOutputLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  const float* dptr =  srclayers[0]->data(this).cpu_data();
  for (int i = 0; i < srclayers[0]->data(this).shape(0); i++) {
    std::cout<<vocab_[static_cast<int>(dptr[i])];
  }
}

}  // namespace singa;
