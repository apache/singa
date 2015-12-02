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

#include "singa/neuralnet/input_layer/image_preprocess.h"
#include "singa/utils/image_transform.h"
namespace singa {

using std::vector;

void ImagePreprocessLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  InputLayer::Setup(conf, srclayers);
  scale_ = conf.rgbimage_conf().scale();
  cropsize_ = conf.rgbimage_conf().cropsize();
  mirror_ = conf.rgbimage_conf().mirror();
  const auto& src = srclayers.at(0)->data(this);
  const auto& shape = src.shape();
  CHECK_EQ(shape.size(), 4);
  CHECK_EQ(shape.at(2), shape.at(3));
  data_.resize(1);
  if (cropsize_ != 0 && cropsize_ != shape.at(2)) {
    data_.at(0).Reshape(vector<int>{shape.at(0), shape.at(1), cropsize_, cropsize_});
  } else {
    data_.at(0) = src;
  }
}

void ImagePreprocessLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  const auto& srcdata = srclayers.at(0)->data(this);
  int batchsize = srcdata.shape()[0], channel = srcdata.shape()[1];
  int height = srcdata.shape()[2], width = srcdata.shape()[3];
  const float* srcdptr = srcdata.cpu_data();
  float* dptr = data_.at(0).mutable_cpu_data();
  int srcimage_size = channel * height * width;
  int image_size = channel * data_.at(0).shape()[2] * data_.at(0).shape()[3];
  for (int k = 0; k < batchsize; k++) {
    int h_offset = 0, w_offset = 0;
    if (cropsize_> 0 && ((flag & kTrain) == kTrain)) {
      h_offset = rand() % (srcdata.shape()[1] - cropsize_);
      w_offset = rand() % (srcdata.shape()[2] - cropsize_);
    }
    bool do_mirror = mirror_ && rand() % 2 && ((flag & kTrain) == kTrain);
    ImageTransform(srcdptr + k * srcimage_size, nullptr, do_mirror, cropsize_,
        cropsize_, h_offset, w_offset, srcdata.shape()[1], height, width,
        scale_, dptr + image_size);
  }
}

}  // namespace singa
