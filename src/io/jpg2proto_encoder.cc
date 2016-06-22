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

#include "singa/io/encoder.h"
#include "singa/proto/model.pb.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace singa {

namespace io{
  void JPG2ProtoEncoder::Setup(const EncoderConf& conf) { return; }

  std::string JPG2ProtoEncoder::Encode(vector<Tensor>& data) {
    // suppose data[0]: data, data[1]: label
    // data[0] has a shape as {height, width, channel}
    CHECK_EQ(data[0].nDim(), 3u);
    int height = data[0].shape()[0];
    int width = data[0].shape()[1];
    int channel = data[0].shape()[2];
    cv::Mat mat = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    CHECK_EQ(height, mat.rows);
    CHECK_EQ(width, mat.cols);
    CHECK_EQ(channel, mat.channels());

    if (data[0].data_type() != kInt)
      LOG(FATAL) << "Data type " << data[0].data_type() <<" is invalid for an raw image";
    const int* raw = data[0].data<const int*>();
    for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++)
        for (int k = 0; k < channel; k++)
	  mat.at<cv::Vec3b>(i, j)[k] = static_cast<uchar>(raw[i * width * channel + j * channel + k]);
    // suppose each image is attached with only one label
    const int* label = data[1].data<const int*>();
    CHECK_EQ(label[0], 2);

    // encode image with jpg format
    std::vector<uchar> buff;
    std::vector<int> param = std::vector<int>(2);
    param[0] = CV_IMWRITE_JPEG_QUALITY;
    param[1] = 100; // default is 95
    cv::imencode(".jpg", mat, buff, param);
    std::string buf(buff.begin(), buff.end());

    std::string output;
    ImageRecordProto image;
    image.set_label(label[0]);
    for (size_t i = 0; i < data[0].nDim(); i++)
      image.add_shape(data[0].shape()[i]);
    image.set_pixel(buf);
    image.SerializeToString(&output);
    return output;
  }
}
}
