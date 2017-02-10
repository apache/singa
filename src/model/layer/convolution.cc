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

RegisterLayerClass(singacpp_convolution, Convolution);
void Convolution::Setup(const Shape &in_sample, const LayerConf &conf) {
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

  const int kStrideDefault = 1;
  if (conv_conf.stride_size() > 0) {
    if (conv_conf.stride_size() == 1) {
      stride_w_ = stride_h_ = conv_conf.stride(0);
    } else {
      stride_w_ = conv_conf.stride(0);
      stride_h_ = conv_conf.stride(1);
    }
  } else {
    stride_w_ = kStrideDefault;
    stride_h_ = kStrideDefault;
    if (conv_conf.has_stride_w()) {
        stride_w_ = conv_conf.stride_w();
    }
    if (conv_conf.has_stride_h()) {
        stride_h_ = conv_conf.stride_h();
    }
  }
  CHECK_GT(stride_w_, 0u);
  CHECK_GE(stride_h_, 0u);  // 0 for 1D conv

  num_filters_ = conv_conf.num_output();
  bias_term_ = conv_conf.bias_term();

  // Shape of input image
  CHECK_EQ(in_sample.size(), 3u);
  channels_ = in_sample.at(0);
  height_ = in_sample.at(1);
  width_ = in_sample.at(2);

  conv_height_ = 1;
  if (stride_h_ > 0)
    conv_height_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  conv_width_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  out_sample_shape_ = vector<size_t>{num_filters_, conv_height_, conv_width_};

  col_height_ = channels_ * kernel_w_ * kernel_h_;
  col_width_ = conv_height_ * conv_width_;

  // Setup shape of weight_ and bias_
  weight_.Reshape(Shape{num_filters_, col_height_});
  if (bias_term_)
    bias_.Reshape(Shape{num_filters_});
  // Assume the order of param is: weight, bias
  for (const auto &spec : conf.param()) param_specs_.push_back(spec);
}

/// \copydoc Layer::Forward(int flag, const Tensor&)
const Tensor Convolution::Forward(int flag, const Tensor &input) {
  CHECK(buf_.empty());
  CHECK_EQ(input.device()->lang(), kCpp);
  CHECK_EQ(input.nDim(), 4u);
  if (flag & kTrain) buf_.push(input);
  size_t batchsize = input.shape(0);
  size_t imagesize = input.Size() / batchsize;
  DataType dtype = input.data_type();
  auto dev = input.device();
  Shape shape{batchsize, num_filters_, conv_height_, conv_width_};
  Tensor output(shape, dev, dtype);
  Tensor col_data(Shape{col_height_, col_width_});
  float *data_col = new float[col_height_ * col_width_];
  auto in_data = input.data<float>();
  for (size_t b = 0; b < batchsize; b++) {
    Im2col(in_data + b * imagesize, channels_, height_, width_, kernel_h_,
        kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, data_col);
    col_data.CopyDataFromHostPtr(data_col, col_height_ * col_width_);
    Tensor each = Mult(weight_, col_data);
    if (bias_term_) {
      AddColumn(bias_, &each);
    }
    CopyDataToFrom(&output, each, each.Size(), b * each.Size());
  }
  delete[] data_col;
  return output;
}

/// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
const std::pair<Tensor, vector<Tensor>> Convolution::Backward(
    int flag, const Tensor &grad) {
  CHECK_EQ(grad.device()->lang(), kCpp);
  CHECK_EQ(grad.nDim(), 4u);
  CHECK(!buf_.empty());
  Tensor src_data = buf_.top();
  buf_.pop();
  vector<Tensor> param_grad;
  Tensor dx;
  Tensor db, dw;
  dx.ResetLike(src_data);
  dw.ResetLike(weight_);
  dw.SetValue(0.0f);
  size_t batchsize = grad.shape(0);
  size_t imagesize = src_data.Size() / batchsize;
  if (bias_term_) {
    auto tmpshp = Shape{batchsize * num_filters_, grad.Size() / (batchsize * num_filters_)};
    Tensor tmp1 = Reshape(grad, tmpshp);

    Tensor tmp2(Shape{batchsize * num_filters_});
    SumColumns(tmp1, &tmp2);
    Tensor tmp3 = Reshape(tmp2, Shape{batchsize, num_filters_});

    db.ResetLike(bias_);
    SumRows(tmp3, &db);
  }

  auto in_data = src_data.data<float>();
  Tensor col_data(Shape{col_height_, col_width_});
  float *data_col = new float[col_height_ * col_width_];
  float *dx_b = new float[imagesize];
  for (size_t b = 0; b < batchsize; b++) {
    Im2col(in_data + b * imagesize, channels_, height_, width_, kernel_h_,
           kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, data_col);

    col_data.CopyDataFromHostPtr(data_col, col_height_ * col_width_);
    Tensor grad_b(Shape{num_filters_, conv_height_ * conv_width_});
    CopyDataToFrom(&grad_b, grad, grad_b.Size(), 0, b * grad_b.Size());
    dw += Mult(grad_b, col_data.T());
    Tensor dcol_b = Mult(weight_.T(), grad_b);
    auto dcol_data = dcol_b.data<float>();
    Col2im(dcol_data, channels_, height_, width_, kernel_h_, kernel_w_, pad_h_,
           pad_w_, stride_h_, stride_w_, dx_b);
    dx.CopyDataFromHostPtr(dx_b, imagesize, b * imagesize);
  }
  param_grad.push_back(dw);
  if (bias_term_)
    param_grad.push_back(db);
  delete[] data_col;
  delete[] dx_b;
  return std::make_pair(dx, param_grad);
}
void Convolution::ToDevice(std::shared_ptr<Device> device) {
  Layer::ToDevice(device);
  weight_.ToDevice(device);
  bias_.ToDevice(device);
}

void Convolution::Im2col(const float *data_im, const int channels,
                         const int height, const int width,
                         const int kernel_h, const int kernel_w,
                         const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         float *data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col  = ( width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
              data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

void Convolution::Col2im(const float *data_col, const int channels,
                         const int height, const int width,
                         const int kernel_h, const int kernel_w,
                         const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         float *data_im) {
  memset(data_im, 0, height * width * channels * sizeof(float));
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col  = ( width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}
}  // namespace singa
