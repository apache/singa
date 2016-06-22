/************************************************************
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
*************************************************************/

#include "../include/singa/io/encoder.h"
#include "../include/singa/io/decoder.h"
#include "gtest/gtest.h"
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace singa;

TEST(Decoder, Decode) {
  io::JPG2ProtoEncoder encoder;
  io::Proto2JPGDecoder decoder;

  // initial random seed
  srand(time(NULL));

  size_t height = 40, width = 30;
  size_t nheight = 256, nwidth = 256, channel = 3;
  size_t total = nheight * nwidth * channel;
  int raw_label = 2;
  cv::Mat image(height, width, CV_8UC3); 
  for (size_t i = 0; i < height; i++)
    for (size_t j = 0; j < width; j++)
      for (size_t k = 0; k < channel; k++)
        image.at<cv::Vec3b>(i, j)[k] = static_cast<uchar>(rand() % 256);
  
  cv::Mat transformed;
  cv::Size size(nwidth, nheight);
  cv::resize(image, transformed, size);
  EXPECT_EQ(nwidth, transformed.cols);
  EXPECT_EQ(nheight, transformed.rows);
  EXPECT_EQ(channel, transformed.channels());
  
  unsigned char* buff = transformed.data;
  int* buf = new int[total];
  for (size_t i = 0; i < total; i++)
    buf[i] = static_cast<int>(buff[i]);
  Shape shape{nheight, nwidth, channel};
  Tensor pixel(shape, kInt), label(Shape{1}, kInt);
  pixel.CopyDataFromHostPtr<int>(buf, total);
  label.CopyDataFromHostPtr<int>(&raw_label, 1);

  std::vector<Tensor> input;
  input.push_back(pixel);
  input.push_back(label); 
  const int* in_pixel = input[0].data<const int *>();
  for(size_t i = 0; i < total; i++) 
    EXPECT_EQ(buf[i], in_pixel[i]);
  const int* in_label = input[1].data<const int *>();
  EXPECT_EQ(2, in_label[0]);
  EXPECT_EQ(2, input.size());

  std::string tmp = encoder.Encode(input);
  std::vector<Tensor> output = decoder.Decode(tmp);
  EXPECT_EQ(2, output.size());
  EXPECT_EQ(kFloat32, output[0].data_type());
  Shape out_shape = output[0].shape();
  for (size_t i = 0; i < shape.size(); i++)
    EXPECT_EQ(shape[i], out_shape[i]);
  const float* out_label = output[1].data<const float*>();
  EXPECT_EQ(raw_label, out_label[0]);
  // opencv imencode will have some information loss
  //const float* out_pixel = output[0].data<const float*>();
  //for(size_t i = 0; i < total; i++) 
  //  EXPECT_LE(fabs(in_pixel[i]-out_pixel[i]), 10.f);
}
