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

#include "singa/io/decoder.h"

#ifdef USE_OPENCV

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace singa {

std::vector<Tensor> JPGDecoder::Decode(std::string value) {
  std::vector<Tensor> output;

  ImageRecord record;
  record.ParseFromString(value);
  std::vector<uchar> pixel(record.pixel().begin(), record.pixel().end());

  // decode image
  cv::Mat mat = cv::imdecode(cv::Mat(pixel), CV_LOAD_IMAGE_COLOR);
  size_t height = mat.size().height, width = mat.size().width,
         channel = mat.channels();
  Shape shape(record.shape().begin(), record.shape().end());
  // CHECK_EQ(shape[0], height);
  // CHECK_EQ(shape[1], width);
  // CHECK_EQ(shape[2], channel);
  Tensor image(shape);

  float* data = new float[image.Size()];
  if (image_dim_order_ == "CHW") {
    for (size_t i = 0; i < height; i++)
      for (size_t j = 0; j < width; j++)
        for (size_t k = 0; k < channel; k++)
          data[k * height * width + i * width + j] =
              static_cast<float>(static_cast<int>(mat.at<cv::Vec3b>(i, j)[k]));
  } else if (image_dim_order_ == "HWC") {
    for (size_t i = 0; i < height; i++)
      for (size_t j = 0; j < width; j++)
        for (size_t k = 0; k < channel; k++)
          data[i * width * channel + j * channel + k] =
              static_cast<float>(static_cast<int>(mat.at<cv::Vec3b>(i, j)[k]));
  } else {
    LOG(FATAL) << "Unknow dimension order for images " << image_dim_order_
               << " Only support 'HWC' and 'CHW'";
  }
  image.CopyDataFromHostPtr<float>(data, image.Size());
  output.push_back(image);
  delete[] data;

  if (record.label_size()) {
    Tensor label(Shape{1}, kInt);
    int labelid = record.label(0);
    label.CopyDataFromHostPtr(&labelid, 1);
    output.push_back(label);
  }
  return output;
}
}  // namespace singa
#endif

#endif
