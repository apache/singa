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

#ifndef SRC_MODEL_LAYER_OPENCL_POOLING_H_
#define SRC_MODEL_LAYER_OPENCL_POOLING_H_

#include "pooling.h"
#include "singa/core/common.h"
#include "singa/model/layer.h"
#include "singa/proto/core.pb.h"
#include "singa/utils/opencl_utils.h"

#ifdef USE_OPENCL

namespace singa {

class OpenclPooling : public Pooling {
 public:
  /// \copydoc Layer::layer_type()
  const std::string layer_type() const override { return "OpenclPooling"; }

  const Tensor Forward(int flag, const Tensor& input) override;

  const std::pair<Tensor, std::vector<Tensor>> Backward(
      int flag, const Tensor& grad) override;

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const Shape& in_sample, const LayerConf& conf) override;

 private:
  void Pooling_Forward_Max(const int num, Block* src, Block* mask,
                           const int height, const int width,
                           const int pooled_h, const int pooled_w,
                           const int kernel_h, const int kernel_w,
                           const int stride_h, const int stride_w,
                           const int pad_h, const int pad_w, Block* dst,
                           const int channels, Context* ctx);

  void Pooling_Forward_Ave(const int num, Block* src, Block* dst,
                           const int height, const int width,
                           const int pooled_h, const int pooled_w,
                           const int kernel_h, const int kernel_w,
                           const int stride_h, const int stride_w,
                           const int pad_h, const int pad_w, const int channels,
                           Context* ctx);

  void Pooling_Forward_Sto_Train(Block* src, Block* rand, const int height,
                                 const int width, const int pooled_h,
                                 const int pooled_w, const int kernel_h,
                                 const int kernel_w, const int stride_h,
                                 const int stride_w, const int channels,
                                 Block* dst, Context* ctx);

  void Pooling_Forward_Sto_Test(Block* src, Block* dst, const int height,
                                const int width, const int pooled_h,
                                const int pooled_w, const int kernel_h,
                                const int kernel_w, const int stride_h,
                                const int stride_w, const int channels,
                                Context* ctx);

  void Pooling_Backward_Max(Block* top, Block* mask, const int num,
                            const int channels, const int height,
                            const int width, const int pooled_h,
                            const int pooled_w, const int kernel_h,
                            const int kernel_w, const int pad_h,
                            const int pad_w, const int stride_h,
                            const int stride_w, Block* bottom, Context* ctx);

  void Pooling_Backward_Ave(Block* bottom, const int num, const int channels,
                            const int height, const int width,
                            const int pooled_h, const int pooled_w,
                            const int kernel_h, const int kernel_w,
                            const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w, Block* top,
                            Context* ctx);

  void Pooling_Backward_Sto(Block* src, Block* rand, Block* dst,
                            const int height, const int width,
                            const int pooled_h, const int pooled_w,
                            const int kernel_h, const int kernel_w,
                            const int stride_h, const int stride_w,
                            const int channels, Context* ctx);
};

}  // namespace singa

#endif  // USE_OPENCL

#endif  // SRC_MODEL_LAYER_OPENCL_POOLING_H_
