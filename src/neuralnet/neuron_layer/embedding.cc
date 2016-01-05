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
#include "singa/utils/math_addr.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/singleton.h"
#include "singa/utils/context.h"

namespace singa {

void EmbeddingLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  NeuronLayer::Setup(conf, srclayers);
  vocab_size_ = conf.embedding_conf().vocab_size();
  feature_dim_ = conf.embedding_conf().feature_dim();
  vocab_ = Param::Create(conf.param(0));
  vocab_->Setup(vector<int>{vocab_size_, feature_dim_});
  batchsize_ = srclayers.at(0)->data(unroll_index()).shape(0);
  data_.Reshape(batchsize_, feature_dim_);
  grad_.ReshapeLike(data_);
}

void EmbeddingLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  const float* word_idx = srclayers.at(0)->data(unroll_index()).cpu_data();
  int device = Singleton<Context>::Instance()->device_id();
  if (device == -1) {
    const float* src = vocab_->data().cpu_data();
    float* dst = data_.mutable_cpu_data();
    for (int i = 0; i < batchsize_; i++) {
      memcpy(dst + i * feature_dim_,
          src + static_cast<int>(word_idx[i]) * feature_dim_,
          feature_dim_ * sizeof(float));
    }
  } else {
#ifdef USE_GPU
    const float* src = vocab_->data().gpu_data();
    float* dst = data_.mutable_gpu_data();
    for (int i = 0; i < batchsize_; i++) {
      cudaMemcpy(dst + i * feature_dim_,
          src + static_cast<int>(word_idx[i]) * feature_dim_,
          feature_dim_ * sizeof(float), cudaMemcpyDefault);
    }
#else
    LOG(FATAL) << "Not implemented";
#endif
  }
}

void EmbeddingLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  const float* word_idx = srclayers.at(0)->data(unroll_index()).cpu_data();
  auto context = Singleton<Context>::Instance();
  if ((flag & kAggGrad) == 0)
    Zero(vocab_->mutable_grad());

  if (context->device_id() == -1) {
    const float* src = grad_.cpu_data();
    float* dst = vocab_->mutable_grad()->mutable_cpu_data();
    memset(dst, 0 , sizeof(float) * grad_.count());
    for (int i = 0; i < batchsize_; i++) {
      cpu_axpy(feature_dim_, 1.0f, src + i * feature_dim_,
          dst + static_cast<int>(word_idx[i]) * feature_dim_);
    }
  } else {
#ifdef USE_GPU
    const float* src = grad_.gpu_data();
    float* dst = vocab_->mutable_grad()->mutable_gpu_data();
    for (int i = 0; i < batchsize_; i++) {
      gpu_axpy(context->cublas_handle(), grad_.count(), 1.0f,
          src + i * feature_dim_,
          dst + static_cast<int>(word_idx[i]) * feature_dim_);
    }
#else
    LOG(FATAL) << "Not implemented";
#endif
  }
}

}  // namespace singa
