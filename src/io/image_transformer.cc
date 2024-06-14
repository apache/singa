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

#include <time.h>

#include "singa/io/transformer.h"

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

namespace singa {

Tensor ImageTransformer::Apply(int flag, Tensor& input) {
  CHECK_LE(input.nDim(), 4u);
  CHECK_GE(input.nDim(), 2u);
  CHECK_EQ(input.data_type(), kFloat32)
      << "Data type " << input.data_type() << " is invalid for an raw image";
  srand((unsigned int)time(NULL));
  /// TODO
  /// currently only consider one sample each time

  /// resize image using opencv resize
  Tensor temp1;
#ifdef USE_OPENCV
  temp1 = resize(input, resize_height_, resize_width_, image_dim_order_);
#else
  temp1 = input;
#endif

  /// crop
  Tensor temp2;
  size_t height = 0, width = 0;
  if (input.nDim() >= 3u) {
    if (image_dim_order_ == "CHW")
      height = temp1.shape(input.nDim() - 2),
      width = temp1.shape(input.nDim() - 1);
    else if (image_dim_order_ == "HWC")
      height = temp1.shape(input.nDim() - 3),
      width = temp1.shape(input.nDim() - 2);
    else
      LOG(FATAL) << "Unknow dimension order for images " << image_dim_order_
                 << " Only support 'HWC' and 'CHW'";
  } else  /// input is 2D gray image
    height = temp1.shape(0), width = temp1.shape(1);

  if (crop_shape_.size() == 2) {
    if (flag == kTrain) {
      /// random crop
      if (crop_shape_[0] > height || crop_shape_[0] > width)
        LOG(FATAL) << "Crop size larger than the size of raw image";
      size_t crop_h_offset = rand() % ((height - crop_shape_[0]) / 2),
             crop_w_offset = rand() % ((width - crop_shape_[1]) / 2);
      temp2 = crop(temp1, crop_shape_[0], crop_shape_[1], crop_h_offset,
                   crop_w_offset, image_dim_order_);
    } else if (flag == kEval) {
      /// central crop
      size_t crop_h_offset = (height - crop_shape_[0]) / 2,
             crop_w_offset = (width - crop_shape_[1]) / 2;
      temp2 = crop(temp1, crop_shape_[0], crop_shape_[1], crop_h_offset,
                   crop_w_offset, image_dim_order_);
    }
  } else
    temp2 = temp1;

  /// mirror
  Tensor output;
  if ((flag == kTrain) && (rand() % 2))
    output = mirror(temp2, true, false, image_dim_order_);
  else
    output = temp2;
  return output;
}

#ifdef USE_OPENCV
Tensor resize(Tensor& input, const size_t resize_height,
              const size_t resize_width, const string& image_dim_order) {
  CHECK_LE(input.nDim(), 4u);
  CHECK_GE(input.nDim(), 2u);
  if (!resize_height || !resize_width) return input;
  Tensor output;
  cv::Mat mat;
  const auto* in = input.data<float>();
  if (input.nDim() == 4u) {
    /// TODO
    /// batch based resize
    LOG(FATAL) << "Not implemented";
  } else if (input.nDim() == 3u) {
    if (image_dim_order == "CHW") {
      size_t height = input.shape(1), width = input.shape(2),
             channel = input.shape(0);
      if (channel == 3u) {
        mat = cv::Mat(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
        for (size_t i = 0; i < height; i++)
          for (size_t j = 0; j < width; j++)
            for (size_t k = 0; k < channel; k++)
              mat.at<cv::Vec3f>(i, j)[k] =
                  in[k * height * width + i * width + j];
      } else if (channel == 1u) {
        mat = cv::Mat(height, width, CV_32FC1);
        for (size_t i = 0; i < height; i++)
          for (size_t j = 0; j < width; j++)
            mat.at<cv::Vec<float, 1>>(i, j)[0] = in[i * width + j];
      } else
        LOG(FATAL) << "Invalid channel size: " << channel;
    } else if (image_dim_order == "HWC") {
      size_t height = input.shape(0), width = input.shape(1),
             channel = input.shape(2);
      if (channel == 3u) {
        mat = cv::Mat(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
        for (size_t i = 0; i < height; i++)
          for (size_t j = 0; j < width; j++)
            for (size_t k = 0; k < channel; k++)
              mat.at<cv::Vec3f>(i, j)[k] =
                  in[i * width * channel + j * channel + k];
      } else if (channel == 1u) {  /// 2D gray image
        mat = cv::Mat(height, width, CV_32FC1);
        for (size_t i = 0; i < height; i++)
          for (size_t j = 0; j < width; j++)
            mat.at<cv::Vec<float, 1>>(i, j)[0] = in[i * width + j];
      } else
        LOG(FATAL) << "Invalid channel size: " << channel;
    } else {
      LOG(FATAL) << "Unknow dimension order for images " << image_dim_order
                 << " Only support 'HWC' and 'CHW'";
    }
  } else {  /// 2D gray image
    size_t height = input.shape(0), width = input.shape(1);
    mat = cv::Mat(height, width, CV_32FC1);
    for (size_t i = 0; i < height; i++)
      for (size_t j = 0; j < width; j++)
        mat.at<cv::Vec<float, 1>>(i, j)[0] = in[i * width + j];
  }
  cv::Size size(resize_width, resize_height);
  cv::Mat resized;
  cv::resize(mat, resized, size);
  CHECK_EQ(resized.size().height, resize_height);
  CHECK_EQ(resized.size().width, resize_width);
  size_t new_size = resize_height * resize_width * resized.channels();
  float* out = new float[new_size];
  if (input.nDim() == 4u) {
    /// TODO
    /// batch based resize
    LOG(FATAL) << "Not implemented";
  } else if (input.nDim() == 3u) {
    if (image_dim_order == "CHW") {
      size_t height = resize_height, width = resize_width,
             channel = input.shape(0);
      if (channel == 3u) {
        for (size_t i = 0; i < height; i++)
          for (size_t j = 0; j < width; j++)
            for (size_t k = 0; k < channel; k++)
              out[k * height * width + i * width + j] =
                  resized.at<cv::Vec3f>(i, j)[k];
      } else {  /// 2D gray image
        for (size_t i = 0; i < height; i++)
          for (size_t j = 0; j < width; j++)
            out[i * width + j] = resized.at<cv::Vec<float, 1>>(i, j)[0];
      }
      Tensor temp(Shape{channel, height, width});
      temp.CopyDataFromHostPtr<float>(out, new_size);
      output = temp;
    } else {
      size_t height = resize_height, width = resize_width,
             channel = input.shape(2);
      if (channel == 3u) {
        for (size_t i = 0; i < height; i++)
          for (size_t j = 0; j < width; j++)
            for (size_t k = 0; k < channel; k++)
              out[i * width * channel + j * channel + k] =
                  resized.at<cv::Vec3f>(i, j)[k];
      } else {  /// 1 channel
        for (size_t i = 0; i < height; i++)
          for (size_t j = 0; j < width; j++)
            out[i * width + j] = resized.at<cv::Vec<float, 1>>(i, j)[0];
      }
      Tensor temp(Shape{height, width, channel});
      temp.CopyDataFromHostPtr<float>(out, new_size);
      output = temp;
    }
  } else {  /// 2D gray image
    size_t height = resize_height, width = resize_width;
    for (size_t i = 0; i < height; i++)
      for (size_t j = 0; j < width; j++)
        out[i * width + j] = resized.at<cv::Vec<float, 1>>(i, j)[0];
    Tensor temp(Shape{height, width});
    temp.CopyDataFromHostPtr<float>(out, new_size);
    output = temp;
  }
  delete[] out;
  return output;
}
#endif

Tensor crop(Tensor& input, const size_t crop_height, const size_t crop_width,
            const size_t crop_h_offset, const size_t crop_w_offset,
            const string& image_dim_order) {
  CHECK_LE(input.nDim(), 4u);
  CHECK_GE(input.nDim(), 2u);

  Tensor output;
  const float* in = input.data<float>();
  size_t out_idx = 0, in_idx = 0;
  if (input.nDim() == 4u) {
    /// TODO
    LOG(FATAL) << "Not implemented";
  } else if (input.nDim() == 3u) {
    if (image_dim_order == "CHW") {
      size_t height = input.shape(1), width = input.shape(2),
             channel = input.shape(0);
      CHECK_LE(crop_height + crop_h_offset, height);
      CHECK_LE(crop_width + crop_w_offset, width);
      float* out = new float[crop_height * crop_width * channel];
      for (size_t c = 0; c < channel; c++) {
        for (size_t h = 0; h < crop_height; h++) {
          for (size_t w = 0; w < crop_width; w++) {
            in_idx =
                (c * height + crop_h_offset + h) * width + crop_w_offset + w;
            out_idx = (c * crop_height + h) * crop_width + w;
            out[out_idx] = in[in_idx];
          }
        }
      }
      output.Resize(Shape{channel, crop_height, crop_width});
      output.CopyDataFromHostPtr<float>(out,
                                        crop_height * crop_width * channel);
      delete[] out;
    } else if (image_dim_order == "HWC") {
      size_t height = input.shape(0), width = input.shape(1),
             channel = input.shape(2);
      CHECK_LE(crop_height + crop_h_offset, height);
      CHECK_LE(crop_width + crop_w_offset, width);
      float* out = new float[crop_height * crop_width * channel];
      for (size_t c = 0; c < channel; c++) {
        for (size_t h = 0; h < crop_height; h++) {
          for (size_t w = 0; w < crop_width; w++) {
            in_idx =
                ((crop_h_offset + h) * width + crop_w_offset + w) * channel + c;
            out_idx = (h * crop_width + w) * channel + c;
            out[out_idx] = in[in_idx];
          }
        }
      }
      output.Resize(Shape{crop_height, crop_width, channel});
      output.CopyDataFromHostPtr<float>(out,
                                        crop_height * crop_width * channel);
      delete[] out;
    } else {
      LOG(FATAL) << "Unknow dimension order for images " << image_dim_order
                 << " Only support 'HWC' and 'CHW'";
    }
  } else {  /// 2D gray image
    size_t height = input.shape(0), width = input.shape(1);
    CHECK_LE(crop_height + crop_h_offset, height);
    CHECK_LE(crop_width + crop_w_offset, width);
    float* out = new float[crop_height * crop_width];
    for (size_t h = 0; h < crop_height; h++) {
      for (size_t w = 0; w < crop_width; w++) {
        in_idx = (crop_h_offset + h) * width + crop_w_offset + w;
        out_idx = h * crop_width + w;
        out[out_idx] = in[in_idx];
      }
    }
    output.Resize(Shape{crop_height, crop_width});
    output.CopyDataFromHostPtr<float>(out, crop_height * crop_width);
    delete[] out;
  }
  return output;
}

Tensor mirror(Tensor& input, const bool horizontal_mirror,
              const bool vertical_mirror, const string& image_dim_order) {
  CHECK_LE(input.nDim(), 4u);
  CHECK_GE(input.nDim(), 2u);
  if (!horizontal_mirror && !vertical_mirror) return input;

  Tensor output;
  const float* in = input.data<float>();
  size_t out_idx = 0, in_idx = 0;
  if (input.nDim() == 4u) {
    /// TODO
    LOG(FATAL) << "Not implemented";
  } else if (input.nDim() == 3u) {
    if (image_dim_order == "CHW") {
      size_t height = input.shape(1), width = input.shape(2),
             channel = input.shape(0);
      float* out = new float[height * width * channel];
      for (size_t c = 0; c < channel; c++) {
        for (size_t h = 0; h < height; h++) {
          for (size_t w = 0; w < width; w++) {
            in_idx = (c * height + h) * width + w;
            if (horizontal_mirror && vertical_mirror)
              out_idx =
                  (c * height + (height - 1 - h)) * width + (width - 1 - w);
            else if (horizontal_mirror)
              out_idx = (c * height + h) * width + (width - 1 - w);
            else  /// only do vertical mirror
              out_idx = (c * height + (height - 1 - h)) * width + w;
            out[out_idx] = in[in_idx];
          }
        }
      }
      output.Resize(Shape{channel, height, width});
      output.CopyDataFromHostPtr<float>(out, height * width * channel);
      delete[] out;
    } else if (image_dim_order == "HWC") {
      size_t height = input.shape(0), width = input.shape(1),
             channel = input.shape(2);
      float* out = new float[height * width * channel];
      for (size_t c = 0; c < channel; c++) {
        for (size_t h = 0; h < height; h++) {
          for (size_t w = 0; w < width; w++) {
            in_idx = (h * width + w) * channel + c;
            if (horizontal_mirror && vertical_mirror)
              out_idx =
                  ((height - 1 - h) * width + (width - 1 - w)) * channel + c;
            else if (horizontal_mirror)
              out_idx = (h * width + (width - 1 - w)) * channel + c;
            else  /// only do vertical mirror
              out_idx = ((height - 1 - h) * width + w) * channel + c;
            out[out_idx] = in[in_idx];
          }
        }
      }
      output.Resize(Shape{height, width, channel});
      output.CopyDataFromHostPtr<float>(out, height * width * channel);
      delete[] out;
    } else {
      LOG(FATAL) << "Unknow dimension order for images " << image_dim_order
                 << " Only support 'HWC' and 'CHW'";
    }
  } else {  /// 2D gray image
    size_t height = input.shape(0), width = input.shape(1);
    float* out = new float[height * width];
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        in_idx = h * width + w;
        if (horizontal_mirror && vertical_mirror)
          out_idx = (height - 1 - h) * width + (width - 1 - w);
        else if (horizontal_mirror)
          out_idx = h * width + (width - 1 - w);
        else  /// only do vertical mirror
          out_idx = (height - 1 - h) * width + w;
        out[out_idx] = in[in_idx];
      }
    }
    output.Resize(Shape{height, width});
    output.CopyDataFromHostPtr<float>(out, height * width);
    delete[] out;
  }
  return output;
}
}  // namespace singa
