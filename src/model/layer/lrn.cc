/*********************************************************
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
************************************************************/
#include "lrn.h"

namespace singa{
RegisterLayerClass(LRN);
void LRN::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  out_sample_shape_ = in_sample;
  local_size_ = conf.lrn_conf().local_size();
  CHECK_EQ(local_size_ % 2, 1) << "LRN only supports odd values for Localvol";
  k_ = conf.lrn_conf().k();
  alpha_ = conf.lrn_conf().alpha();
  beta_ = conf.lrn_conf().beta();
}

const Tensor LRN::Forward(int flag, const Tensor& input) {
  //Tensor output;
  //const float salpha = alpha_ / local_size_;
  LOG(FATAL) << "Not implemented";
  /* Tensor API may be need
   * 1. set
   * template <typename Dtype>
   * void Set(Dtype val);
   *
   * 2. axpy
   * 3. padding
   *
   *
   */
  Tensor output;
  return output;
}

const std::pair<Tensor, vector<Tensor>> LRN::Backward(
    int flag, const Tensor& grad) {
  LOG(FATAL) << "Not implemented";
  Tensor dx;
  vector<Tensor> param_grad;
  return std::make_pair(dx, param_grad);
}

}  // namespace
