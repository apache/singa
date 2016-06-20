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

#include "../include/singa/io/image2jpg_encoder.h"
#include "../include/singa/io/image2jpg_decoder.h"
#include "gtest/gtest.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace singa;
using namespace singa::io;
using namespace cv;

TEST(Decoder, Decode) {
  Encoder encoder;
  Decoder decoder;
  string path = "test/samples/test.jpeg";
  size_t resize_height = 256;
  size_t resize_width = 256;
  size_t channel = 3;
  int raw_label[] = {1};
  Mat image = imread(path, CV_LOAD_IMAGE_COLOR);
  Mat transformed;
  Size size(resize_height, resize_width);
  resize(image, transformed, size);
  
  vector<int> buff;
  buff.assign(transformed.datastart, transformed.dataend);
  Shape shape{channel, (size_t)transformed.size().height, (size_t)transformed.size().width};
  Tensor pixel(shape), label(Shape{1});
  pixel.CopyDataFromHostPtr<vector<int>>(&buff, buff.size());
  label.CopyDataFromHostPtr<int>(raw_label, 1);

  vector<Tensor> input;
  input.push_back(pixel);
  input.push_back(label);
  string str0 = encoder.Encode(input);
  vector<Tensor> output = decoder.Decode(str0);
  Shape out_shape = output.at(0).shape();
  const int* out_pixel = output.at(0).data<const int *>();
  const int* out_label = output.at(1).data<const int *>();
  EXPECT_EQ(raw_label[0], out_label[0]);
  for (size_t i = 0; i < shape.size(); i++)
    EXPECT_EQ(shape[i], out_shape[i]);
  for(size_t i = 0; i < 10; i++) 
    EXPECT_EQ(buff[i], out_pixel[i]);
}
