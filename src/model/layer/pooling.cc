/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "./pooling.h"
#include "singa/model/layer.h"
namespace singa {

RegisterLayerClass(Pooling);
void Pooling::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  PoolingConf pool_conf = conf.pooling_conf();
  if (pool_conf.has_kernel_size()) {
    kernel_w_ = kernel_h_ = pool_conf.kernel_size();
  } else {
    kernel_w_ = pool_conf.kernel_w();
    kernel_h_ = pool_conf.kernel_h();
  }
  CHECK_GT(kernel_w_, 0u);
  CHECK_GT(kernel_h_, 0u);

  if (pool_conf.has_pad()) {
    pad_w_ = pad_h_ = pool_conf.pad();
  } else {
    pad_w_ = pool_conf.pad_w();
    pad_h_ = pool_conf.pad_h();
  }
  CHECK_GE(pad_w_, 0u);
  CHECK_GE(pad_h_, 0u);

  if (pool_conf.has_stride()) {
    stride_w_ = stride_h_ = pool_conf.stride();
  } else {
    stride_w_ = pool_conf.stride_w();
    stride_h_ = pool_conf.stride_h();
  }
  CHECK_GT(stride_w_, 0u);
  CHECK_GE(stride_h_, 0u);  // 0 for 1D pooling

  pool_ = pool_conf.pool();
  CHECK(pool_ == PoolingConf_PoolMethod_AVE ||
        pool_ == PoolingConf_PoolMethod_MAX ||
        pool_ == PoolingConf_PoolMethod_STOCHASTIC)
      << "Padding implemented only for average and max pooling.";

  CHECK_EQ(in_sample.size(), 3u);
  channels_ = in_sample.at(0);
  height_ = in_sample.at(1);
  width_ = in_sample.at(2);
  pooled_height_ = 1;
  if (stride_h_ > 0)
    pooled_height_ =
      static_cast<size_t>((height_ + 2 * pad_h_ - kernel_h_) / stride_h_) + 1;
  pooled_width_ =
    static_cast<size_t>((width_ + 2 * pad_w_ - kernel_w_) / stride_w_) + 1;
  out_sample_shape_ = vector<size_t>{channels_, pooled_height_, pooled_width_};
}

const Tensor Pooling::Forward(int flag, const Tensor& input) {
  Tensor out;

  return out;
}

const std::pair<Tensor, vector<Tensor>> Pooling::Backward(int flag,
                                                          const Tensor& grad) {
  vector<Tensor> param_grad;
  Tensor input_grad;

  return std::make_pair(input_grad, param_grad);
}

}  // namespace singa
