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

#include "singa/neuralnet/loss_layer/softmax.h"

#include <algorithm>
#include <glog/logging.h>
#include "mshadow/tensor.h"

namespace singa {

using namespace mshadow;
using mshadow::cpu;

using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Tensor;

using std::vector;

void SoftmaxLossLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 2);
  LossLayer::Setup(proto, srclayers);
  data_.resize(1);
  data_.at(0).Reshape(srclayers[0]->data(this).shape());
  batchsize_ = data_.at(0).shape()[0];
  dim_ = data_.at(0).count() / batchsize_;
  topk_ = proto.softmaxloss_conf().topk();
  scale_ = proto.softmaxloss_conf().scale();
}

void SoftmaxLossLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  Shape<2> s = Shape2(batchsize_, dim_);
  Tensor<cpu, 2> prob(data_.at(0).mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);
  const auto& label = srclayers[1]->aux_data(this);
  const float* probptr = prob.dptr;
  float loss = 0, precision = 0;
  for (int n = 0; n < batchsize_; n++) {
    int ilabel = static_cast<int>(label[n]);
    //  CHECK_LT(ilabel,10);
    CHECK_GE(ilabel, 0);
    float prob_of_truth = probptr[ilabel];
    loss -= log(std::max(prob_of_truth, FLT_MIN));
    vector<std::pair<float, int> > probvec;
    for (int j = 0; j < dim_; ++j) {
      probvec.push_back(std::make_pair(probptr[j], j));
    }
    std::partial_sort(probvec.begin(), probvec.begin() + topk_, probvec.end(),
                      std::greater<std::pair<float, int> >());
    // check if true label is in top k predictions
    for (int k = 0; k < topk_; k++) {
      if (probvec[k].second == static_cast<int>(label[n])) {
        precision++;
        break;
      }
    }
    probptr += dim_;
  }
  CHECK_EQ(probptr, prob.dptr + prob.shape.Size());
  metric_.Add("loss", loss * scale_ / (1.0f * batchsize_));
  metric_.Add("accuracy", precision * scale_ / (1.0f * batchsize_));
}

void SoftmaxLossLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  const auto& label = srclayers[1]->aux_data();
  Blob<float>* gsrcblob = srclayers[0]->mutable_grad(this);
  gsrcblob->CopyFrom(data_.at(0));
  float* gsrcptr = gsrcblob->mutable_cpu_data();
  for (int n = 0; n < batchsize_; n++) {
    gsrcptr[n*dim_ + static_cast<int>(label[n])] -= 1.0f;
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc *= scale_ / (1.0f * batchsize_);
}

}  // namespace singa
