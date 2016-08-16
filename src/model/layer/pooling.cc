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

#include "./pooling.h"
#include "singa/model/layer.h"
namespace singa {

RegisterLayerClass(singacpp_pooling, Pooling);
void Pooling::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  PoolingConf pool_conf = conf.pooling_conf();
  if (pool_conf.has_kernel_size()) {
    kernel_w_ = kernel_h_ = pool_conf.kernel_size();
  } else {
    kernel_w_ = pool_conf.kernel_w();
    kernel_h_ = pool_conf.kernel_h();
  }
  CHECK_GT(kernel_w_, 0u);
  CHECK_GT(kernel_h_, 0u);

  if (pool_conf.has_pad()) {
    pad_w_ = pad_h_ = pool_conf.pad();
  } else {
    pad_w_ = pool_conf.pad_w();
    pad_h_ = pool_conf.pad_h();
  }
  CHECK_GE(pad_w_, 0u);
  CHECK_GE(pad_h_, 0u);

  if (pool_conf.has_stride()) {
    stride_w_ = stride_h_ = pool_conf.stride();
  } else {
    stride_w_ = pool_conf.stride_w();
    stride_h_ = pool_conf.stride_h();
  }
  CHECK_GT(stride_w_, 0u);
  CHECK_GE(stride_h_, 0u);  // 0 for 1D pooling

  pool_ = pool_conf.pool();
  CHECK(pool_ == PoolingConf_PoolMethod_AVE ||
        pool_ == PoolingConf_PoolMethod_MAX ||
        pool_ == PoolingConf_PoolMethod_STOCHASTIC)
      << "Padding implemented only for average and max pooling.";

  CHECK_EQ(in_sample.size(), 3u);
  channels_ = in_sample.at(0);
  height_ = in_sample.at(1);
  width_ = in_sample.at(2);
  pooled_height_ = 1;
  if (stride_h_ > 0)
    pooled_height_ =
        static_cast<size_t>((height_ + 2 * pad_h_ - kernel_h_) / stride_h_) + 1;
  pooled_width_ =
      static_cast<size_t>((width_ + 2 * pad_w_ - kernel_w_) / stride_w_) + 1;
  out_sample_shape_ = vector<size_t>{channels_, pooled_height_, pooled_width_};
}

const Tensor Pooling::Forward(int flag, const Tensor& input) {
  CHECK(buf_.empty());
  CHECK_EQ(input.device()->lang(), kCpp);
  CHECK_EQ(input.nDim(), 4u);
  size_t batchsize = input.shape(0);
  DataType dtype = input.data_type();
  auto dev = input.device();
  Shape shape{batchsize, channels_, pooled_height_, pooled_width_};
  Tensor output(shape, dev, dtype);
  float* outptr = new float[output.Size()];
  auto inptr = input.data<float>();
  if (pool_ == PoolingConf_PoolMethod_MAX) {
    Tensor mask;
    mask.ResetLike(output);
    float* maskptr = new float[mask.Size()];
    ForwardMaxPooling(inptr, batchsize, channels_, height_, width_, kernel_h_,
                      kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, outptr,
                      maskptr);
    mask.CopyDataFromHostPtr(maskptr, mask.Size());
    if (flag & kTrain) buf_.push(mask);
    delete[] maskptr;
  } else if (pool_ == PoolingConf_PoolMethod_AVE)
    ForwardAvgPooling(inptr, batchsize, channels_, height_, width_, kernel_h_,
                      kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, outptr);
  else
    LOG(FATAL) << "Unknow pooling method";

  output.CopyDataFromHostPtr(outptr, output.Size());
  delete[] outptr;
  return output;
}

const std::pair<Tensor, vector<Tensor>> Pooling::Backward(int flag,
                                                          const Tensor& grad) {
  CHECK_EQ(grad.device()->lang(), kCpp);
  CHECK_EQ(grad.nDim(), 4u);
  vector<Tensor> param_grad;
    size_t batchsize = grad.shape(0);
  Shape shape{batchsize, channels_, height_, width_};
  auto dev = grad.device();
  DataType dtype = grad.data_type();
  Tensor dx(shape, dev, dtype);
  auto gradptr = grad.data<float>();
  float* dxptr = new float[dx.Size()];
  if (pool_ == PoolingConf_PoolMethod_MAX) {
    CHECK(!buf_.empty());
    Tensor mask = buf_.top();
    buf_.pop();
    auto maskptr = mask.data<float>();
    BackwardMaxPooling(gradptr, maskptr, batchsize, channels_, height_, width_,
                       kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_,
                       stride_w_, dxptr);
  } else if (pool_ == PoolingConf_PoolMethod_AVE) {
    BackwardAvgPooling(gradptr, batchsize, channels_, height_, width_,
                       kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_,
                       stride_w_, dxptr);
  } else {
    LOG(FATAL) << "Unknow pooling method";
  }

  dx.CopyDataFromHostPtr(dxptr, dx.Size());
  delete[] dxptr;
  return std::make_pair(dx, param_grad);
}

void Pooling::ForwardMaxPooling(const float* bottom, const int num,
                                const int channels, const int height,
                                const int width, const int kernel_h,
                                const int kernel_w, const int pad_h,
                                const int pad_w, const int stride_h,
                                const int stride_w, float* top, float* mask) {
  int top_height = (height + pad_h * 2 - kernel_h) / stride_h + 1;
  int top_width = (width + pad_w * 2 - kernel_w) / stride_w + 1;
  int top_count = num * top_height * top_width * channels;
  for (int i = 0; i < top_count; i++) {
    mask[i] = -1;
    top[i] = -FLT_MAX;
  }
  const int bottom_offset = height * width;
  const int top_offset = top_height * top_width;
  // The main loop
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height);
          int wend = std::min(wstart + kernel_w, width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int top_index = ph * top_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              if (bottom[index] > top[top_index]) {
                top[top_index] = bottom[index];
                mask[top_index] = index;
              }
            }
          }
        }
      }
      // compute offset
      bottom += bottom_offset;
      top += top_offset;
      mask += top_offset;
    }
  }
}

void Pooling::BackwardMaxPooling(const float* top, const float* mask,
                                 const int num, const int channels,
                                 const int height, const int width,
                                 const int kernel_h, const int kernel_w,
                                 const int pad_h, const int pad_w,
                                 const int stride_h, const int stride_w,
                                 float* bottom) {
  int top_height = (height + pad_h * 2 - kernel_h) / stride_h + 1;
  int top_width = (width + pad_w * 2 - kernel_w) / stride_w + 1;
  const int top_offset = top_height * top_width;
  const int bottom_offset = height * width;
  memset(bottom, 0, sizeof(float) * num * channels * bottom_offset);
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          const int top_idx = ph * top_width + pw;
          const int bottom_idx = static_cast<int>(mask[top_idx]);
          bottom[bottom_idx] += top[top_idx];
        }
      }
      top += top_offset;
      mask += top_offset;
      bottom += bottom_offset;
    }
  }
}

void Pooling::ForwardAvgPooling(const float* bottom, const int num,
                                const int channels, const int height,
                                const int width, const int kernel_h,
                                const int kernel_w, const int pad_h,
                                const int pad_w, const int stride_h,
                                const int stride_w, float* top) {
  int top_height = (height + pad_h * 2 - kernel_h) / stride_h + 1;
  int top_width = (width + pad_w * 2 - kernel_w) / stride_w + 1;
  int top_count = num * top_height * top_width * channels;
  for (int i = 0; i < top_count; i++) {
    top[i] = 0;
  }
  const int bottom_offset = height * width;
  const int top_offset = top_height * top_width;
  // The main loop
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height + pad_h);
          int wend = std::min(wstart + kernel_w, width + pad_w);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, height);
          wend = std::min(wend, width);
          const int top_index = ph * top_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              top[top_index] += bottom[index];
            }
          }
          top[top_index] /= pool_size;
        }
      }
      // compute offset
      bottom += bottom_offset;
      top += top_offset;
    }
  }
}

void Pooling::BackwardAvgPooling(const float* top, const int num,
                                 const int channels, const int height,
                                 const int width, const int kernel_h,
                                 const int kernel_w, const int pad_h,
                                 const int pad_w, const int stride_h,
                                 const int stride_w, float* bottom) {
  int top_height = (height + pad_h * 2 - kernel_h) / stride_h + 1;
  int top_width = (width + pad_w * 2 - kernel_w) / stride_w + 1;
  const int top_offset = top_height * top_width;
  const int bottom_offset = height * width;
  memset(bottom, 0, sizeof(float) * num * channels * bottom_offset);
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height + pad_h);
          int wend = std::min(wstart + kernel_w, width + pad_w);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, height);
          wend = std::min(wend, width);
          const int top_index = ph * top_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              bottom[index] += top[top_index] / pool_size;
            }
          }
        }
      }
      top += top_offset;
      bottom += bottom_offset;
    }
  }
}
}  // namespace singa
