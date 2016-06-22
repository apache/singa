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

#include "singa/io/decoder.h"
#include "singa/proto/model.pb.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace singa {

namespace io {
  void Proto2JPGDecoder::Setup(const DecoderConf& conf) { return; }

  std::vector<Tensor> Proto2JPGDecoder::Decode(std::string value) {
    std::vector<Tensor> output;
    ImageRecordProto image;
    image.ParseFromString(value);
    Shape shape(image.shape().begin(), image.shape().end());
    Tensor raw(shape), label(Shape{1});
    std::vector<uchar> pixel(image.pixel().begin(), image.pixel().end());

    // decode image
    cv::Mat mat = cv::imdecode(cv::Mat(pixel), CV_LOAD_IMAGE_COLOR);
    int height = mat.rows, width = mat.cols, channel = mat.channels();
    CHECK_EQ(shape[0], height);
    CHECK_EQ(shape[1], width);
    CHECK_EQ(shape[2], channel);

    float* data = new float[raw.Size()];
    for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++)
        for (int k = 0; k < channel; k++)
          data[i * width * channel + j * channel + k] = static_cast<float>(static_cast<int>(mat.at<cv::Vec3b>(i, j)[k]));
    raw.CopyDataFromHostPtr<float>(data, raw.Size());
    float l = static_cast<float>(image.label());
    label.CopyDataFromHostPtr(&l, 1);
    output.push_back(raw);
    output.push_back(label);
    return output;
  }
}
}
