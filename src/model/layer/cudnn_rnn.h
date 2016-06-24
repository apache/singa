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

#ifndef SRC_MODEL_LAYER_CUDNN_RNN_H_
#define SRC_MODEL_LAYER_CUDNN_RNN_H_
#include "singa/singa_config.h"
#ifdef USE_CUDNN
#include <string>
#include <utility>
#include <vector>
#include "./rnn.h"
#include "singa/core/common.h"
#include "singa/model/layer.h"
#include "singa/proto/core.pb.h"
#include "singa/utils/string.h"
#include <cudnn.h>
#include <chrono>
#include "./cudnn_utils.h"
#include "singa/utils/logging.h"

namespace singa {
class CudnnRNN : public RNN {
 public:
  ~CudnnRNN();
  /// \copydoc Layer::layer_type()
  const std::string layer_type() const override { return "CudnnRNN"; }

  const vector<Tensor> Forward(int flag, const vector<Tensor>& inputs) override;
  const std::pair<vector<Tensor>, vector<Tensor>> Backward(int flag, const vector<Tensor>& grads) override;

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const Shape& in_sample, const LayerConf &conf) override;

  void ToDevice(std::shared_ptr<Device> device) override;

  size_t workspace_byte_limit() { return workspace_byte_limit_; }
  // string prefer() { return prefer_; }
  string inputMode() const { return inputMode_; }
  string direction() const { return direction_; }
  string mode() const { return mode_; }

 protected:
  /// Init cudnn related data structures.
  void InitCudnn(const Tensor& input);

 protected:
  bool has_init_cudnn_ = false;
  cudnnTensorDescriptor_t* x_descs_ = nullptr;
  cudnnTensorDescriptor_t* y_descs_ = nullptr;
  cudnnTensorDescriptor_t hx_desc_ = nullptr;
  cudnnTensorDescriptor_t cx_desc_ = nullptr;
  cudnnTensorDescriptor_t hy_desc_ = nullptr;
  cudnnTensorDescriptor_t cy_desc_ = nullptr;
  cudnnFilterDescriptor_t weight_desc_ = nullptr;
  cudnnRNNDescriptor_t rnn_desc_ = nullptr;
  cudnnDropoutDescriptor_t dropout_desc_ = nullptr;
  size_t workspace_byte_limit_, workspace_count_;
  size_t ReserveSize_;
  Tensor workspace_;
  string inputMode_;
  string direction_;
  string mode_;
  Tensor reserve_;
  Tensor dropoutStates_;
};

}  // namespace singa

#endif  // USE_CUDNN
#endif  // SRC_MODEL_LAYER_CUDNN_RNN_H_
