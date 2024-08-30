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
#ifndef DISABLE_WARNINGS

#include "singa/io/encoder.h"

#ifdef USE_OPENCV

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace singa {

std::string JPGEncoder::Encode(vector<Tensor>& data) {
  // suppose image: image, data[1]: label
  CHECK_LE(data.size(), 2u);
  const Tensor& image = data.at(0);
  CHECK_EQ(image.nDim(), 3u);
  CHECK_EQ(image.data_type(), kUChar)
      << "Data type " << image.data_type() << " is invalid for an raw image";
  const auto* raw = image.data<unsigned char>();
  cv::Mat mat;
  if (image_dim_order_ == "HWC") {
    size_t height = image.shape(0), width = image.shape(1),
           channel = image.shape(2);
    mat = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 0; i < height; i++)
      for (size_t j = 0; j < width; j++)
        for (size_t k = 0; k < channel; k++)
          mat.at<cv::Vec3b>(i, j)[k] =
              raw[i * width * channel + j * channel + k];
  } else if (image_dim_order_ == "CHW") {
    size_t channel = image.shape(0), height = image.shape(1),
           width = image.shape(2);
    mat = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 0; i < height; i++)
      for (size_t j = 0; j < width; j++)
        for (size_t k = 0; k < channel; k++)
          mat.at<cv::Vec3b>(i, j)[k] = raw[k * height * width + i * width + j];
  } else {
    LOG(FATAL) << "Unknow dimension order for images " << image_dim_order_
               << " Only support 'HWC' and 'CHW'";
  }

  // encode image with jpg format
  std::vector<uchar> buff;
  std::vector<int> param = std::vector<int>(2);
  param[0] = CV_IMWRITE_JPEG_QUALITY;
  param[1] = 100;  // default is 95
  cv::imencode(".jpg", mat, buff, param);
  std::string buf(buff.begin(), buff.end());

  std::string output;
  ImageRecord record;
  for (size_t i = 0; i < image.nDim(); i++) record.add_shape(image.shape(i));
  record.set_pixel(buf);

  // suppose each image is attached with at most one label
  if (data.size() == 2) {
    const int* label = data[1].data<int>();
    // CHECK_EQ(label[0], 2);
    record.add_label(label[0]);
  }

  record.SerializeToString(&output);
  return output;
}
}  // namespace singa
#endif  // USE_OPENCV

#endif
