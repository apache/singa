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

#include <iostream>

#include "gtest/gtest.h"
#include "singa/io/transformer.h"

// decide whether to use opencv
// #include "singa/singa_config.h"

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

using singa::Shape;
TEST(ImageTransformer, Setup) {
  singa::ImageTransformer img_transformer;

  singa::TransformerConf conf;
  conf.set_resize_height(256);
  conf.set_resize_width(256);
  conf.set_horizontal_mirror(true);
  conf.set_image_dim_order("HWC");
  conf.add_crop_shape(224u);
  conf.add_crop_shape(200u);

  img_transformer.Setup(conf);
  EXPECT_EQ(256, img_transformer.resize_height());
  EXPECT_EQ(256, img_transformer.resize_width());
  EXPECT_EQ(true, img_transformer.horizontal_mirror());
  EXPECT_EQ("HWC", img_transformer.image_dim_order());
  EXPECT_EQ(224u, img_transformer.crop_shape()[0]);
  EXPECT_EQ(200u, img_transformer.crop_shape()[1]);
}

TEST(ImageTransformer, Apply3D) {
  size_t n = 180;
  float* x = new float[n];
  size_t channel = 3, height = 6, width = 10;
  singa::Tensor in(singa::Shape{height, width, channel});
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < n; i++) x[i] = (float)(rand() % 256);
  in.CopyDataFromHostPtr<float>(x, n);
  int resize_height = 4, resize_width = 6;

  singa::ImageTransformer img_transformer;
  singa::TransformerConf conf;
  conf.set_resize_height(resize_height);
  conf.set_resize_width(resize_width);
  conf.set_horizontal_mirror(false);
  conf.set_image_dim_order("HWC");
  conf.add_crop_shape(2u);
  conf.add_crop_shape(3u);
  img_transformer.Setup(conf);

  singa::Tensor out = img_transformer.Apply(singa::kEval, in);
  EXPECT_EQ(2u, out.shape(0));
  EXPECT_EQ(3u, out.shape(1));
  const float* y = out.data<float>();
#ifdef USE_OPENCV
  cv::Mat mat(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
  for (size_t i = 0; i < height; i++)
    for (size_t j = 0; j < width; j++)
      for (size_t k = 0; k < channel; k++)
        mat.at<cv::Vec3f>(i, j)[k] = x[i * width * channel + j * channel + k];
  cv::Size size(resize_width, resize_height);
  cv::Mat resized;
  cv::resize(mat, resized, size);
  EXPECT_EQ(resize_height, resized.size().height);
  EXPECT_EQ(resize_width, resized.size().width);
  size_t new_size = resize_height * resize_width * channel;
  float* xt = new float[new_size];
  for (int i = 0; i < resize_height; i++)
    for (int j = 0; j < resize_width; j++)
      for (size_t k = 0; k < channel; k++)
        xt[i * resize_width * channel + j * channel + k] =
            resized.at<cv::Vec3f>(i, j)[k];
  for (size_t c = 0; c < 3; c++)
    for (size_t h = 0; h < 2; h++)
      for (size_t w = 0; w < 3; w++) {
        // size_t in_idx = (c * height + 1 + h) * width + 1 + w,
        //    out_idx = (c * 2 + h) * 3 + w;
        // test for HWC
        size_t in_idx = ((h + 1) * resize_width + 1 + w) * channel + c,
               out_idx = (h * 3 + w) * channel + c;
        EXPECT_EQ(xt[in_idx], y[out_idx]);
      }
  delete[] xt;
#else
  for (size_t c = 0; c < 3; c++)
    for (size_t h = 0; h < 2; h++)
      for (size_t w = 0; w < 3; w++) {
        // size_t in_idx = (c * height + 2 + h) * width + 3 + w,
        //    out_idx = (c * 2 + h) * 3 + w;
        // test for HWC
        size_t in_idx = ((h + 2) * width + 3 + w) * channel + c,
               out_idx = (h * 3 + w) * channel + c;
        EXPECT_EQ(x[in_idx], y[out_idx]);
      }
#endif
  delete[] x;
}

TEST(ImageTransformer, Apply2D) {
  size_t n = 60;
  float* x = new float[n];
  size_t height = 6, width = 10;
  singa::Tensor in(singa::Shape{height, width});
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < n; i++) x[i] = (float)(rand() % 256);
  in.CopyDataFromHostPtr<float>(x, n);
  int resize_height = 4, resize_width = 6;

  singa::ImageTransformer img_transformer;
  singa::TransformerConf conf;
  conf.set_resize_height(resize_height);
  conf.set_resize_width(resize_width);
  conf.set_horizontal_mirror(false);
  conf.set_image_dim_order("HWC");
  conf.add_crop_shape(2u);
  conf.add_crop_shape(3u);
  img_transformer.Setup(conf);

  singa::Tensor out = img_transformer.Apply(singa::kEval, in);
  EXPECT_EQ(2u, out.shape(0));
  EXPECT_EQ(3u, out.shape(1));
  const float* y = out.data<float>();
#ifdef USE_OPENCV
  cv::Mat mat(height, width, CV_32FC1, cv::Scalar(0, 0, 0));
  for (size_t i = 0; i < height; i++)
    for (size_t j = 0; j < width; j++)
      mat.at<cv::Vec<float, 1>>(i, j)[0] = x[i * width + j];
  cv::Size size(resize_width, resize_height);
  cv::Mat resized;
  cv::resize(mat, resized, size);
  EXPECT_EQ(resize_height, resized.size().height);
  EXPECT_EQ(resize_width, resized.size().width);
  size_t new_size = resize_height * resize_width;
  float* xt = new float[new_size];
  for (int i = 0; i < resize_height; i++)
    for (int j = 0; j < resize_width; j++)
      xt[i * resize_width + j] = resized.at<cv::Vec<float, 1>>(i, j)[0];

  for (size_t h = 0; h < 2; h++)
    for (size_t w = 0; w < 3; w++) {
      size_t in_idx = (h + 1) * resize_width + 1 + w, out_idx = h * 3 + w;
      EXPECT_EQ(xt[in_idx], y[out_idx]);
    }
  delete[] xt;
#else
  for (size_t h = 0; h < 2; h++)
    for (size_t w = 0; w < 3; w++) {
      size_t in_idx = (h + 2) * width + 3 + w, out_idx = h * 3 + w;
      EXPECT_EQ(x[in_idx], y[out_idx]);
    }
#endif
  delete[] x;
}

#ifdef USE_OPENCV
TEST(ImageTransformer, Resize) {
  size_t n = 180;
  float* x = new float[n];
  size_t channel = 3, height = 6, width = 10;
  singa::Tensor in(singa::Shape{height, width, channel});
  srand(time(NULL));
  for (size_t i = 0; i < n; i++) x[i] = (float)(rand() % 256);
  in.CopyDataFromHostPtr<float>(x, n);
  int resize_height = 4, resize_width = 5;
  singa::Tensor out = singa::resize(in, resize_height, resize_width, "HWC");
  const float* y = out.data<float>();

  cv::Mat mat(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
  for (size_t i = 0; i < height; i++)
    for (size_t j = 0; j < width; j++)
      for (size_t k = 0; k < channel; k++)
        mat.at<cv::Vec3f>(i, j)[k] = x[i * width * channel + j * channel + k];
  cv::Size size(resize_width, resize_height);
  cv::Mat resized;
  cv::resize(mat, resized, size);
  EXPECT_EQ(resize_height, resized.size().height);
  EXPECT_EQ(resize_width, resized.size().width);
  size_t new_size = resize_height * resize_width * channel;
  float* xt = new float[new_size];
  for (int i = 0; i < resize_height; i++)
    for (int j = 0; j < resize_width; j++)
      for (size_t k = 0; k < channel; k++)
        xt[i * resize_width * channel + j * channel + k] =
            resized.at<cv::Vec3f>(i, j)[k];

  for (size_t i = 0; i < new_size; i++) EXPECT_EQ(xt[i], y[i]);
  delete[] x;
  delete[] xt;
}
#endif

TEST(ImageTransformer, Crop) {
  size_t n = 180;
  float* x = new float[n];
  size_t channel = 3, height = 6, width = 10;
  singa::Tensor in(singa::Shape{channel, height, width});
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < n; i++) x[i] = (float)(rand() % 256);
  in.CopyDataFromHostPtr<float>(x, n);
  size_t crop_height = 3, crop_width = 4, crop_h_offset = 2, crop_w_offset = 5;
  singa::Tensor out = singa::crop(in, crop_height, crop_width, crop_h_offset,
                                  crop_w_offset, "CHW");

  const float* y = out.data<float>();
  for (size_t h = 0; h < crop_height; h++)
    for (size_t w = 0; w < crop_width; w++)
      for (size_t c = 0; c < channel; c++) {
        size_t out_idx = c * crop_height * crop_width + h * crop_width + w;
        size_t in_idx = c * height * width + (h + crop_h_offset) * width + w +
                        crop_w_offset;
        EXPECT_EQ(x[in_idx], y[out_idx]);
      }
  delete[] x;
}

TEST(ImageTransformer, Mirror) {
  size_t n = 30;
  float* x = new float[n];
  size_t channel = 3, height = 2, width = 5;
  singa::Tensor in(singa::Shape{height, width, channel});
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < n; i++) x[i] = (float)(rand() % 256);
  in.CopyDataFromHostPtr<float>(x, n);
  singa::Tensor out = singa::mirror(in, true, false, "HWC");

  const float* y = out.data<float>();
  for (size_t h = 0; h < height; h++)
    for (size_t w = 0; w < width; w++)
      for (size_t c = 0; c < channel; c++) {
        size_t out_idx = h * width * channel + (width - 1 - w) * channel + c;
        size_t in_idx = h * width * channel + w * channel + c;
        EXPECT_EQ(x[in_idx], y[out_idx]);
      }
  delete[] x;
}
