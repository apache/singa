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

#ifndef SRC_MODEL_LAYER_OPENCL_CONVOLUTION_H_
#define SRC_MODEL_LAYER_OPENCL_CONVOLUTION_H_

#include "convolution.h"
#include "singa/core/common.h"
#include "singa/model/layer.h"
#include "singa/proto/core.pb.h"
#include "singa/singa_config.h"
#include "singa/utils/opencl_utils.h"

#ifdef USE_OPENCL

namespace singa {

class OpenclConvolution : public Convolution {
 public:
  /// \copydoc Layer::layer_type()
  const std::string layer_type() const override { return "OpenclConvolution"; }

  const Tensor Forward(int flag, const Tensor& input) override;

  const std::pair<Tensor, std::vector<Tensor>> Backward(
      int flag, const Tensor& grad) override;

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const Shape& in_sample, const LayerConf& conf) override;

  void ToDevice(std::shared_ptr<Device> device) override;

 private:
  void Im2Col(Block* src, int data_im_off, const int height, const int width,
              const int kernel_h, const int kernel_w, const int pad_h,
              const int pad_w, const int stride_h, const int stride_w,
              const int conv_h, const int conv_w, const int data_col_off,
              const int channels, Block* dst, Context* ctx);

  void Col2Im(Block* src, const int data_col_off, const int height,
              const int width, const int kernel_h, const int kernel_w,
              const int pad_h, const int pad_w, const int stride_h,
              const int stride_w, const int conv_h, const int conv_w,
              const int data_im_off, const int channels, Block* dst,
              Context* ctx);
};

}  // namespace singa

#endif  // USE_OPENCL

#endif  // SRC_MODEL_LAYER_OPENCL_CONVOLUTION_H_
