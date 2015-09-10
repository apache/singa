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
#include "neuralnet/layer.h"
#include "mshadow/tensor.h"


namespace singa {
using namespace mshadow;
using mshadow::cpu;

using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Shape3;
using mshadow::Shape4;
using mshadow::Tensor;


/********** * Implementation for EuclideanLossLayer*************************/
void EuclideanLossLayer::ComputeFeature(int flag, Metric* perf) {
  int count = srclayers_[0]->data(this).count();
  CHECK_EQ(count, srclayers_[1]->data(this).count());
  const float* reconstruct_dptr = srclayers_[0]->data(this).cpu_data();
  const float* input_dptr = srclayers_[1]->data(this).cpu_data();
  float loss = 0;
  for (int i = 0; i < count; i++) {
      loss += (input_dptr[i] - reconstruct_dptr[i]) *
        (input_dptr[i] - reconstruct_dptr[i]);
  }
  perf->Add("loss", loss / srclayers_[0]->data(this).shape()[0]);
}
void EuclideanLossLayer::ComputeGradient(int flag, Metric* perf) {
  int count = srclayers_[0]->data(this).count();
  CHECK_EQ(count, srclayers_[1]->data(this).count());
  const float* reconstruct_dptr = srclayers_[0]->data(this).cpu_data();
  const float* input_dptr = srclayers_[1]->data(this).cpu_data();
  Blob<float>* gsrcblob = srclayers_[0]->mutable_grad(this);
  float* gsrcptr = gsrcblob->mutable_cpu_data();
  for (int i = 0; i < count; i++) {
    gsrcptr[i] = reconstruct_dptr[i]-input_dptr[i];
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc /= srclayers_[0]->data(this).shape()[0];
}


/********** * Implementation for SoftmaxLossLayer*************************/
void SoftmaxLossLayer::Setup(const LayerProto& proto, int npartitions) {
  LossLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 2);
  data_.Reshape(srclayers_[0]->data(this).shape());
  batchsize_ = data_.shape()[0];
  dim_ = data_.count() / batchsize_;
  topk_ = proto.softmaxloss_conf().topk();
  metric_.Reshape(vector<int>{2});
  scale_ = proto.softmaxloss_conf().scale();
}
void SoftmaxLossLayer::ComputeFeature(int flag, Metric* perf) {
  Shape<2> s = Shape2(batchsize_, dim_);
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers_[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);
  const float* label = srclayers_[1]->data(this).cpu_data();
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
  perf->Add("loss", loss * scale_ / (1.0f * batchsize_));
  perf->Add("accuracy", precision * scale_ / (1.0f * batchsize_));
}

void SoftmaxLossLayer::ComputeGradient(int flag, Metric* perf) {
  const float* label = srclayers_[1]->data(this).cpu_data();
  Blob<float>* gsrcblob = srclayers_[0]->mutable_grad(this);
  gsrcblob->CopyFrom(data_);
  float* gsrcptr = gsrcblob->mutable_cpu_data();
  for (int n = 0; n < batchsize_; n++) {
    gsrcptr[n*dim_ + static_cast<int>(label[n])] -= 1.0f;
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc *= scale_ / (1.0f * batchsize_);
}

}  // namespace singa
