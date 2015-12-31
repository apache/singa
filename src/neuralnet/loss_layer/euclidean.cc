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

#include <glog/logging.h>
#include "singa/neuralnet/loss_layer.h"
#include "mshadow/tensor.h"

namespace singa {

using namespace mshadow;
using mshadow::cpu;

using mshadow::Shape1;
using mshadow::Tensor;

using std::vector;

void EuclideanLossLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 2);
  Layer::Setup(conf, srclayers);
}

void EuclideanLossLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  int count = srclayers[0]->data(this).count();
  CHECK_EQ(count, srclayers[1]->data(this).count());
  const float* reconstruct_dptr = srclayers[0]->data(this).cpu_data();
  const float* input_dptr = srclayers[1]->data(this).cpu_data();
  float loss = 0;
  for (int i = 0; i < count; i++) {
      loss += (input_dptr[i] - reconstruct_dptr[i]) *
        (input_dptr[i] - reconstruct_dptr[i]);
  }
  loss_ += loss / srclayers[0]->data(this).shape()[0];
  counter_++;
}

void EuclideanLossLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  int count = srclayers[0]->data(this).count();
  CHECK_EQ(count, srclayers[1]->data(this).count());
  const float* reconstruct_dptr = srclayers[0]->data(this).cpu_data();
  const float* input_dptr = srclayers[1]->data(this).cpu_data();
  Blob<float>* gsrcblob = srclayers[0]->mutable_grad(this);
  float* gsrcptr = gsrcblob->mutable_cpu_data();
  for (int i = 0; i < count; i++) {
    gsrcptr[i] = reconstruct_dptr[i]-input_dptr[i];
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc /= srclayers[0]->data(this).shape()[0];
}
const std::string EuclideanLossLayer::ToString(bool debug, int flag) {
  if (debug)
    return Layer::ToString(debug, flag);

  string disp = "Loss = " + std::to_string(loss_ / counter_);
  counter_ = 0;
  loss_ = 0;
  return disp;
}
}  // namespace singa
