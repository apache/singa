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
#ifndef SINGA_MODEL_LAYER_CUDNN_LRN_H_
#define SINGA_MODEL_LAYER_CUDNN_LRN_H_
#include "singa/singa_config.h"
#ifdef USE_CUDNN

#include "cudnn_utils.h"
#include "lrn.h"

namespace singa {
class CudnnLRN : public LRN {
 public:
  ~CudnnLRN();
  /// \copy doc Layer::layer_type()
  // const std::string layer_type() const override { return "CudnnLRN"; }

  const Tensor Forward(int flag, const Tensor& input) override;
  const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                   const Tensor& grad) override;

 private:
  /// Init cudnn related data structures.
  void InitCudnn(const Shape& shape, DataType dtype);

 private:
  bool has_init_cudnn_ = false;
  cudnnLRNMode_t mode_;
  cudnnLRNDescriptor_t lrn_desc_ = nullptr;
  cudnnTensorDescriptor_t shape_desc_ = nullptr;

};  // class CudnnLRN
}  // namespace singa

#endif  // USE_CUDNN
#endif  // SINGA_MODEL_LAYER_CUDNN_LRN_H_H
