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

#ifndef SRC_MODEL_LAYER_CUDNN_DROPOUT_H_
#define SRC_MODEL_LAYER_CUDNN_DROPOUT_H_
#include "singa/singa_config.h"
#ifdef USE_CUDNN
#include <cudnn.h>
// cudnn dropout is added in cudnn 5
#if CUDNN_MAJOR >= 5
#include <cudnn.h>

#include <string>
#include <utility>
#include <vector>

#include "./dropout.h"
#include "singa/model/layer.h"

namespace singa {
class CudnnDropout : public Dropout {
 public:
  ~CudnnDropout();
  /// \copydoc Layer::layer_type()
  // const std::string layer_type() const override { return "CudnnDropout"; }

  const Tensor Forward(int flag, const Tensor& input) override;
  const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                   const Tensor& grad) override;

  void ToDevice(std::shared_ptr<Device> device) override;

 private:
  /// Init cudnn related data structures.
  void InitCudnn(int size, DataType dtype, std::shared_ptr<Device> dev,
                 Context* ctx);

 private:
  bool has_init_cudnn_ = false;
  cudnnDropoutDescriptor_t drop_desc_ = nullptr;
  cudnnTensorDescriptor_t x_desc_ = nullptr, y_desc_ = nullptr;
  size_t state_size_, reserve_size_;
  Tensor state_;
};
}  // namespace singa
#endif  // CUDNN_MAJOR>=5
#endif  // USE_CUDNN
#endif  // SRC_MODEL_LAYER_CUDNN_DROPOUT_H_
