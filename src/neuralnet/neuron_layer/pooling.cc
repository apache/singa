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

#include "singa/neuralnet/neuron_layer/pooling.h"

#include <glog/logging.h>
#include "singa/utils/singleton.h"

namespace singa {

using std::vector;

/******************** Implementation for PoolingLayer******************/
void PoolingLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  PoolingProto pool_conf = conf.pooling_conf();
  kernel_ = pool_conf.kernel();
  stride_ = pool_conf.stride();
  CHECK_LT(pad_, kernel_);
  pool_ = conf.pooling_conf().pool();
  CHECK(pool_ == PoolingProto_PoolMethod_AVG
        || pool_ == PoolingProto_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
  const auto& srcshape = srclayers[0]->data(this).shape();
  int dim = srcshape.size();
  CHECK_GT(dim, 2);
  width_ = srcshape[dim - 1];
  height_ = srcshape[dim - 2];
  if (dim > 3)
    channels_ = srcshape[dim-3];
  else
    channels_ = 1;
  batchsize_ = srcshape[0];
  pooled_height_ = static_cast<int>((height_ - kernel_) / stride_) + 1;
  pooled_width_ = static_cast<int>((width_ - kernel_) / stride_) + 1;
  data_.resize(1);
  data_.at(0).Reshape(vector<int>{batchsize_, channels_, pooled_height_,
                            pooled_width_});
  grad_.ReshapeLike(data_.at(0));
}

void PoolingLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  auto src = Tensor4(srclayers[0]->mutable_data(this));
  auto data = Tensor4(&data_.at(0));
  if (pool_ == PoolingProto_PoolMethod_MAX)
    data = expr::pool<red::maximum>(src, kernel_, stride_);
  else if (pool_ == PoolingProto_PoolMethod_AVG)
    data = expr::pool<red::sum>(src, kernel_, stride_)
      * (1.0f / (kernel_ * kernel_));
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  auto src = Tensor4(srclayers[0]->mutable_data(this));
  auto gsrc = Tensor4(srclayers[0]->mutable_grad(this));
  auto data = Tensor4(&data_.at(0));
  auto grad = Tensor4(&grad_);
  if (pool_ == PoolingProto_PoolMethod_MAX)
    gsrc = expr::unpool<red::maximum>(src, data, grad, kernel_, stride_);
  else if (pool_ == PoolingProto_PoolMethod_AVG)
    gsrc = expr::unpool<red::sum>(src, data, grad, kernel_, stride_)
           * (1.0f / (kernel_ * kernel_));
}

/***************** Implementation of CPoolingLayer ***************/

void CPoolingLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  PoolingLayer::Setup(conf, srclayers);
  if (pool_ == PoolingProto_PoolMethod_MAX)
      mask_.ReshapeLike(data_.at(0));
}
void CPoolingLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if (pool_ == PoolingProto_PoolMethod_MAX)
    ForwardMaxPooling(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
        batchsize_, channels_, height_, width_, kernel_, kernel_, pad_, pad_,
        stride_, stride_, data_.at(0).mutable_cpu_data(), mask_.mutable_cpu_data());
  else if (pool_ == PoolingProto_PoolMethod_AVG)
    ForwardAvgPooling(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
        batchsize_, channels_, height_, width_, kernel_, kernel_, pad_, pad_,
        stride_, stride_, data_.at(0).mutable_cpu_data());
  else
    LOG(FATAL) << "unknow pooling method";
}

void CPoolingLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  if (pool_ == PoolingProto_PoolMethod_MAX)
    BackwardMaxPooling(grad_.cpu_data(), mask_.cpu_data(), batchsize_,
        channels_, height_, width_, kernel_, kernel_, pad_, pad_,
        stride_, stride_, srclayers[0]->mutable_grad(this)->mutable_cpu_data());
  else if (pool_ == PoolingProto_PoolMethod_AVG)
    BackwardAvgPooling(grad_.cpu_data(), batchsize_,
        channels_, height_, width_, kernel_, kernel_, pad_, pad_,
        stride_, stride_, srclayers[0]->mutable_grad(this)->mutable_cpu_data());
  else
    LOG(FATAL) << "unknow pooling method";
}

}  //  namespace singa
