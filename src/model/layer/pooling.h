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
#ifndef SRC_MODEL_LAYER_POOLING_H_
#define SRC_MODEL_LAYER_POOLING_H_
#include <cfloat>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "singa/model/layer.h"

namespace singa {
class Pooling : public Layer {
 public:
  /// \copydoc Layer::layer_type()
  // const std::string layer_type() const override { return "Pooling"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const Shape& in_sample, const LayerConf& conf) override;
  const Shape GetOutputSampleShape() const override {
    CHECK(out_sample_shape_.size()) << "You may haven't call Setup()";
    return out_sample_shape_;
  }
  /// \copydoc Layer::Forward(int flag, const Tensor&)
  const Tensor Forward(int flag, const Tensor& input) override;

  /// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
  const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                   const Tensor& grad) override;

  void ForwardMaxPooling(const float* bottom, const int num, const int channels,
                         const int height, const int width, const int pooled_h,
                         const int pooled_w, const int kernel_h,
                         const int kernel_w, const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w, float* top,
                         float* mask);

  void BackwardMaxPooling(const float* top, const float* mask, const int num,
                          const int channels, const int height, const int width,
                          const int pooled_h, const int pooled_w,
                          const int kernel_h, const int kernel_w,
                          const int pad_h, const int pad_w, const int stride_h,
                          const int stride_w, float* bottom);

  void ForwardAvgPooling(const float* bottom, const int num, const int channels,
                         const int height, const int width, const int pooled_h,
                         const int pooled_w, const int kernel_h,
                         const int kernel_w, const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w, float* top);

  void BackwardAvgPooling(const float* top, const int num, const int channels,
                          const int height, const int width, const int pooled_h,
                          const int pooled_w, const int kernel_h,
                          const int kernel_w, const int pad_h, const int pad_w,
                          const int stride_h, const int stride_w,
                          float* bottom);

  size_t kernel_w() const { return kernel_w_; }
  size_t kernel_h() const { return kernel_h_; }
  size_t pad_w() const { return pad_w_; }
  size_t pad_h() const { return pad_h_; }
  size_t stride_w() const { return stride_w_; }
  size_t stride_h() const { return stride_h_; }
  PoolingConf_PoolMethod pool_method() const { return pool_; }
  size_t channels() const { return channels_; }
  size_t height() const { return height_; }
  size_t width() const { return width_; }

 protected:
  size_t kernel_w_, pad_w_, stride_w_;
  size_t kernel_h_, pad_h_, stride_h_;
  size_t channels_, height_, width_, pooled_height_, pooled_width_;
  PoolingConf_PoolMethod pool_;
  // To store the input and output(of forward) tensors
  std::stack<Tensor> buf_;
  Shape out_sample_shape_;
};
}  // namespace singa
#endif  // SRC_MODEL_LAYER_POOLING_H_
