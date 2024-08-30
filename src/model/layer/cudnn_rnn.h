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
#include <cudnn.h>
#if CUDNN_VERSION >= 5005
#include <cudnn.h>

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "./cudnn_utils.h"
#include "./rnn.h"
#include "singa/core/common.h"
#include "singa/model/layer.h"
#include "singa/proto/core.pb.h"
#include "singa/utils/logging.h"
#include "singa/utils/string.h"

namespace singa {
class CudnnRNN : public RNN {
 public:
  ~CudnnRNN();
  /// \copydoc Layer::layer_type()
  // const std::string layer_type() const override { return "CudnnRNN"; }

  const vector<Tensor> Forward(int flag, const vector<Tensor>& inputs) override;
  const std::pair<vector<Tensor>, vector<Tensor>> Backward(
      int flag, const vector<Tensor>& grads) override;

  void ToDevice(std::shared_ptr<Device> device) override;

  void SetRNNDescriptor(shared_ptr<Device> dev);
  void ResetHiddenAndCellDescriptors(size_t batch_size);
  void DestroyIODescriptors();
  void UpdateIODescriptors(size_t num, const vector<Tensor>& inputs);
  void UpdateSpaces(size_t num, shared_ptr<Device> dev);
  void UpdateStates(size_t num, const vector<Tensor>& inputs);
  Tensor MergeInputs(size_t num, const vector<Tensor>& in);
  vector<Tensor> SplitOutput(size_t num, size_t dim, const vector<Tensor>& in,
                             const Tensor output);

 protected:
  cudnnTensorDescriptor_t* x_descs_ = nullptr;
  cudnnTensorDescriptor_t* dx_descs_ = nullptr;
  cudnnTensorDescriptor_t* y_descs_ = nullptr;
  cudnnTensorDescriptor_t* dy_descs_ = nullptr;
  cudnnTensorDescriptor_t hx_desc_ = nullptr;
  cudnnTensorDescriptor_t dhx_desc_ = nullptr;
  cudnnTensorDescriptor_t cx_desc_ = nullptr;
  cudnnTensorDescriptor_t dcx_desc_ = nullptr;
  cudnnTensorDescriptor_t hy_desc_ = nullptr;
  cudnnTensorDescriptor_t dhy_desc_ = nullptr;
  cudnnTensorDescriptor_t cy_desc_ = nullptr;
  cudnnTensorDescriptor_t dcy_desc_ = nullptr;
  cudnnFilterDescriptor_t weight_desc_ = nullptr;
  cudnnFilterDescriptor_t dweight_desc_ = nullptr;
  cudnnRNNDescriptor_t rnn_desc_ = nullptr;
  cudnnDropoutDescriptor_t dropout_desc_ = nullptr;
  cudnnDataType_t dtype_ = CUDNN_DATA_FLOAT;
  Tensor workspace_;
  Tensor reserve_space_;
  Tensor dropout_state_;
};

}  // namespace singa

#endif  // CUDNN_VERSION >= 5005
#endif  // USE_CUDNN
#endif  // SRC_MODEL_LAYER_CUDNN_RNN_H_
