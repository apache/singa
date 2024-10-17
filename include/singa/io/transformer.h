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

#ifndef SINGA_IO_TRANSFORMER_H_
#define SINGA_IO_TRANSFORMER_H_

#include <string>
#include <vector>

#include "singa/core/tensor.h"
#include "singa/proto/io.pb.h"
#include "singa/proto/model.pb.h"

namespace singa {

/// Base apply class that does data transformations in pre-processing stage.
class Transformer {
 public:
  Transformer() {}
  virtual ~Transformer() {}

  virtual void Setup(const TransformerConf& conf) {}

  virtual Tensor Apply(int flag, Tensor& input) = 0;
};

class ImageTransformer : public Transformer {
 public:
  void Setup(const TransformerConf& conf) override {
    featurewise_center_ = conf.featurewise_center();
    featurewise_std_norm_ = conf.featurewise_std_norm();
    resize_height_ = conf.resize_height();
    resize_width_ = conf.resize_width();
    rescale_ = conf.rescale();
    horizontal_mirror_ = conf.horizontal_mirror();
    image_dim_order_ = conf.image_dim_order();

    /// if crop_shape not contain 2 elements, ignore crop option.
    if (conf.crop_shape_size() == 2)
      crop_shape_ = {conf.crop_shape(0), conf.crop_shape(1)};
  }

  Tensor Apply(int flag, Tensor& input) override;

  bool featurewise_center() const { return featurewise_center_; }
  bool featurewise_std_norm() const { return featurewise_std_norm_; }
  bool horizontal_mirror() const { return horizontal_mirror_; }
  int resize_height() const { return resize_height_; }
  int resize_width() const { return resize_width_; }
  float rescale() const { return rescale_; }
  const Shape crop_shape() const { return crop_shape_; }
  const string image_dim_order() const { return image_dim_order_; }

 private:
  bool featurewise_center_ = false;
  bool featurewise_std_norm_ = false;
  bool horizontal_mirror_ = false;
  int resize_height_ = 0;
  int resize_width_ = 0;
  float rescale_ = 0.f;
  Shape crop_shape_ = {};
  std::string image_dim_order_ = "CHW";
};

#ifdef USE_OPENCV
Tensor resize(Tensor& input, const size_t resize_height,
              const size_t resize_width, const string& image_dim_order);
#endif
Tensor crop(Tensor& input, const size_t crop_height, const size_t crop_width,
            const size_t crop_h_offset, const size_t crop_w_offset,
            const string& image_dim_order);
Tensor mirror(Tensor& input, const bool horizontal_mirror,
              const bool vertical_mirror, const string& image_dim_order);
}  // namespace singa
#endif  // SINGA_IO_TRANSFORMER_H_
