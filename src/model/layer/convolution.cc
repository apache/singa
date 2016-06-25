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

#include "./convolution.h"
#include <vector>
#include "singa/model/layer.h"

namespace singa {
using std::vector;

void Convolution::Setup(const Shape& in_sample, const LayerConf &conf) {
  Layer::Setup(in_sample, conf);
  ConvolutionConf conv_conf = conf.convolution_conf();
  // kernel_size, pad, and stride are repeated fields.
  if (conv_conf.kernel_size_size() > 0) {
    if (conv_conf.kernel_size_size() == 1) {
      kernel_w_ = kernel_h_ = conv_conf.kernel_size(0);
    } else {
      kernel_w_ = conv_conf.kernel_size(0);
      kernel_h_ = conv_conf.kernel_size(1);
    }
  } else {
    kernel_w_ = conv_conf.kernel_w();
    kernel_h_ = conv_conf.kernel_h();
  }
  CHECK_GT(kernel_w_, 0u);
  CHECK_GT(kernel_h_, 0u);

  if (conv_conf.pad_size() > 0) {
    if (conv_conf.pad_size() == 1) {
      pad_w_ = pad_h_ = conv_conf.pad(0);
    } else {
      pad_w_ = conv_conf.pad(0);
      pad_h_ = conv_conf.pad(1);
    }
  } else {
    pad_w_ = conv_conf.pad_w();
    pad_h_ = conv_conf.pad_h();
  }
  CHECK_GE(pad_w_, 0u);
  CHECK_GE(pad_h_, 0u);

  if (conv_conf.stride_size() > 0) {
    if (conv_conf.stride_size() == 1) {
      stride_w_ = stride_h_ = conv_conf.stride(0);
    } else {
      stride_w_ = conv_conf.stride(0);
      stride_h_ = conv_conf.stride(1);
    }
  } else {
    stride_w_ = conv_conf.stride_w();
    stride_h_ = conv_conf.stride_h();
  }
  CHECK_GT(stride_w_, 0u);
  CHECK_GT(stride_h_, 0u);

  num_filters_ = conv_conf.num_output();
  bias_term_ = conv_conf.bias_term();

  // Shape of input image
  CHECK_EQ(in_sample.size(), 3u);
  channels_ = in_sample.at(0);
  height_ = in_sample.at(1);
  width_ = in_sample.at(2);

  conv_height_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  conv_width_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  out_sample_shape_ = vector<size_t>{num_filters_, conv_height_, conv_width_};

  col_height_ = channels_ * kernel_w_ * kernel_h_;
  col_width_ = conv_height_ * conv_width_;

  // Setup shape of weight_ and bias_
  weight_.Reshape(Shape{num_filters_, col_height_});
  bias_.Reshape(Shape{num_filters_});
  // Push back params into param_values_
  // Assume the order of param is: weight, bias
  for (const auto &spec : conf.param()) param_specs_.push_back(spec);
  param_values_.push_back(&weight_);
  param_values_.push_back(&bias_);
}

/// \copydoc Layer::Forward(int flag, const Tensor&)
const Tensor Convolution::Forward(int flag, const Tensor &input) {
  Tensor output;
  // will be used in cpp version later
  Tensor col_data(Shape{col_height_, col_width_});
  Tensor col_grad(Shape{col_height_, col_width_});
  return output;
}

/// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
const std::pair<Tensor, vector<Tensor>> Convolution::Backward(
    int flag, const Tensor &grad) {
  vector<Tensor> param_grad;
  Tensor input_grad;

  return std::make_pair(input_grad, param_grad);
}
void Convolution::ToDevice(std::shared_ptr<Device> device) {
  Layer::ToDevice(device);
  weight_.ToDevice(device);
  bias_.ToDevice(device);
}
}  // namespace singa
