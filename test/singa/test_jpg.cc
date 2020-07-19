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

#include <time.h>

#include "gtest/gtest.h"
#include "singa/io/decoder.h"
#include "singa/io/encoder.h"

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using singa::Shape;
using singa::Tensor;
TEST(Decoder, Decode) {
  singa::JPGEncoder encoder;
  singa::JPGDecoder decoder;

  // initial random seed
  srand(time(NULL));

  singa::EncoderConf encoder_conf;
  encoder_conf.set_image_dim_order("HWC");
  encoder.Setup(encoder_conf);
  EXPECT_EQ("HWC", encoder.image_dim_order());

  singa::DecoderConf decoder_conf;
  decoder_conf.set_image_dim_order("HWC");
  decoder.Setup(decoder_conf);
  EXPECT_EQ("HWC", decoder.image_dim_order());

  size_t height = 4, width = 2;
  size_t nheight = 4, nwidth = 2, channel = 3;
  size_t total = nheight * nwidth * channel;
  cv::Mat image(height, width, CV_8UC3);
  for (size_t i = 0; i < height; i++)
    for (size_t j = 0; j < width; j++)
      for (size_t k = 0; k < channel; k++)
        image.at<cv::Vec3b>(i, j)[k] = static_cast<uchar>(rand() % 256);

  cv::Mat transformed;
  cv::Size size(nwidth, nheight);
  cv::resize(image, transformed, size);
  EXPECT_EQ(static_cast<int>(nwidth), transformed.size().width);
  EXPECT_EQ(static_cast<int>(nheight), transformed.size().height);
  EXPECT_EQ(static_cast<int>(channel), transformed.channels());

  unsigned char* buff = transformed.data;
  Shape shape{nheight, nwidth, channel};
  Tensor pixel(shape, singa::kUChar), label(Shape{1}, singa::kInt);
  pixel.CopyDataFromHostPtr<unsigned char>(buff, total);
  int raw_label = 2;
  label.CopyDataFromHostPtr<int>(&raw_label, 1);

  std::vector<Tensor> input;
  input.push_back(pixel);
  input.push_back(label);
  const auto* in_pixel = input[0].data<unsigned char>();
  for (size_t i = 0; i < total; i++) EXPECT_EQ(buff[i], in_pixel[i]);
  const int* in_label = input[1].data<int>();
  EXPECT_EQ(2, in_label[0]);
  EXPECT_EQ(2u, input.size());

  std::string tmp = encoder.Encode(input);
  std::vector<Tensor> output = decoder.Decode(tmp);
  EXPECT_EQ(2u, output.size());
  EXPECT_EQ(singa::kFloat32, output[0].data_type());
  Shape out_shape = output[0].shape();
  for (size_t i = 0; i < shape.size(); i++) EXPECT_EQ(shape[i], out_shape[i]);
  const int* out_label = output[1].data<int>();
  EXPECT_EQ(raw_label, out_label[0]);
  // opencv imencode will have some information loss
  /*const float* out_pixel = output[0].data<const float>();
  cv::Mat out(height, width, CV_8UC3);
  for (size_t i = 0; i < height; i++)
    for (size_t j = 0; j < width; j++)
      for (size_t k = 0; k < channel; k++)
        out.at<cv::Vec3b>(i, j)[k] =
            out_pixel[i * width * channel + j * channel + k];
  for(size_t i = 0; i < total; i++)
    EXPECT_LE(fabs(in_pixel[i]-out_pixel[i]), 10.f);*/
}
#endif
