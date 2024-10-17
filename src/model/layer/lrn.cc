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

#include <vector>

namespace singa {
RegisterLayerClass(singa_lrn, LRN);
RegisterLayerClass(singacpp_lrn, LRN);
RegisterLayerClass(singacuda_lrn, LRN);
RegisterLayerClass(singacl_lrn, LRN);
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
  Tensor x = input.Clone();
  x.Reshape(Shape{input.shape(0), input.Size() / input.shape(0)});
  vector<Tensor> channels, images;
  // for each image
  for (size_t i = 0; i < input.shape(0); ++i) {
    Tensor image = CopyRows(x, i, i + 1);
    image.Reshape(Shape{input.shape(1), input.shape(2) * input.shape(3)});
    // for each channel of the image
    channels.clear();
    for (size_t c = 0; c < input.shape(1); ++c) {
      Tensor window =
          CopyRows(image, std::max(0, static_cast<int>(c) - local_size_ / 2),
                   std::min(input.shape(1), c + local_size_ / 2 + 1));
      window = Square(window);

      Tensor ch, tmp(Shape{input.shape(2) * input.shape(3)});
      SumRows(window, &tmp);

      tmp *= alpha_;
      tmp += k_;
      tmp = Pow(tmp, beta_);

      ch = CopyRows(image, c, c + 1);
      ch.Reshape(tmp.shape());
      ch = ch / tmp;
      ch.Reshape(Shape{input.shape(2), input.shape(3)});
      channels.push_back(ch);
    }
    Tensor normalized_image = ConcatenateRows(channels);
    normalized_image.Reshape(
        Shape{input.shape(1), input.shape(2) * input.shape(3)});
    images.push_back(normalized_image);
  }
  Tensor output = ConcatenateRows(images);
  output.Reshape(input.shape());
  buf_.push(input);

  return output;
}

const std::pair<Tensor, vector<Tensor>> LRN::Backward(int flag,
                                                      const Tensor& grad) {
  Tensor dx;
  if ((flag & kTrain) == kTrain) {
    Tensor dy = grad.Clone();
    dy.Reshape(Shape{grad.shape(0), grad.Size() / grad.shape(0)});
    Tensor x = buf_.top();
    buf_.pop();
    x.Reshape(dy.shape());
    vector<Tensor> channels, images;
    // for each image
    for (size_t i = 0; i < grad.shape(0); ++i) {
      Tensor image = CopyRows(x, i, i + 1);
      image.Reshape(Shape{grad.shape(1), grad.shape(2) * grad.shape(3)});
      // for each channel of the image
      channels.clear();
      for (size_t c = 0; c < grad.shape(1); ++c) {
        Tensor window =
            CopyRows(image, std::max(0, static_cast<int>(c) - local_size_ / 2),
                     std::min(grad.shape(1), c + local_size_ / 2 + 1));
        Tensor tmp(Shape{grad.shape(2) * grad.shape(3)});
        window = Square(window);
        SumRows(window, &tmp);
        tmp *= alpha_;
        tmp += k_;
        tmp.Reshape(Shape{grad.shape(2), grad.shape(3)});
        channels.push_back(tmp);
      }
      Tensor norm_image = ConcatenateRows(channels);
      norm_image.Reshape(Shape{grad.shape(1), grad.shape(2) * grad.shape(3)});
      images.push_back(norm_image);
    }
    Tensor norm = ConcatenateRows(images);
    norm.Reshape(dy.shape());
    dx = Pow(norm, -beta_);
    dx = dx * dy;
    Tensor tmp = dx * x;
    tmp = tmp / norm;
    images.clear();
    for (size_t i = 0; i < grad.shape(0); ++i) {
      Tensor image = CopyRows(tmp, i, i + 1);
      image.Reshape(Shape{grad.shape(1), grad.shape(2) * grad.shape(3)});
      // for each channel of the image
      channels.clear();
      for (size_t c = 0; c < grad.shape(1); ++c) {
        Tensor window =
            CopyRows(image, std::max(0, static_cast<int>(c) - local_size_ / 2),
                     std::min(grad.shape(1), c + local_size_ / 2 + 1));
        Tensor tmpr(Shape{grad.shape(2) * grad.shape(3)});
        SumRows(window, &tmpr);
        tmpr.Reshape(Shape{grad.shape(2), grad.shape(3)});
        channels.push_back(tmpr);
      }
      Tensor pooled_image = ConcatenateRows(channels);
      pooled_image.Reshape(Shape{grad.shape(1), grad.shape(2) * grad.shape(3)});
      images.push_back(pooled_image);
    }
    Tensor tmp2 = ConcatenateRows(images);
    tmp2 *= (-2.0f * beta_ * alpha_);
    tmp2.Reshape(x.shape());
    tmp2 = tmp2 * x;
    dx = dx + tmp2;
    dx.Reshape(grad.shape());
  } else {
    LOG(ERROR) << "Do not call backward for evaluation phase";
  }
  vector<Tensor> param_grad;
  return std::make_pair(dx, param_grad);
}

}  // namespace singa
