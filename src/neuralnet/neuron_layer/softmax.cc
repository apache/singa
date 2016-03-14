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

#include "singa/neuralnet/neuron_layer.h"

namespace singa {

using namespace mshadow;
using mshadow::cpu;

using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Tensor;

using std::vector;

void SoftmaxLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  NeuronLayer::Setup(proto, srclayers);
  const auto& srcdata = srclayers[0]->data(this);
  batchsize_ = srcdata.shape()[0];
  dim_ = srcdata.count() / batchsize_;
  /*
  num_softmax_per_instance_ = proto.softmax_conf().num_softmax_per_instance();
  count_per_softmax_ = srcdata.count() / batchsize_ / num_softmax_per_instance_;
  */
  data_.Reshape(batchsize_, dim_);
  grad_.ReshapeLike(data_);
}

void SoftmaxLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  int dim = data_.count() / batchsize_;
  Shape<2> s = Shape2(batchsize_, dim);
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);
}

void SoftmaxLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  int batchsize = data_.shape()[0];
  LOG(FATAL) << "not implemented";
  for (int n = 0; n < batchsize; n++) {
    // TODO(wangwei) finish the code using new math API
    // gxi=[(gyi+gyi*yi)-\sum_k(gyk*yk)]*yi
  }
}

}  // namespace singa
