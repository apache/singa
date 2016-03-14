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

#include "singa/neuralnet/input_layer.h"
#include "singa/utils/image_transform.h"
#include "singa/utils/context.h"
#include "singa/utils/singleton.h"

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
  if (cropsize_ && (cropsize_ != shape.at(2) || cropsize_ != shape.at(3))) {
    data_.Reshape(vector<int>{shape.at(0), shape.at(1), cropsize_, cropsize_});
  } else {
    data_ = src;
  }
}

void ImagePreprocessLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  const auto& srcdata = srclayers.at(0)->data(this);
  int batchsize = srcdata.shape(0), channel = srcdata.shape(1);
  int height = srcdata.shape(2), width = srcdata.shape(3);
  int srcimage_size = channel * height * width;
  int image_size = channel * data_.shape(2) * data_.shape(3);
  std::uniform_int_distribution<int> rand1(0, height - cropsize_);
  std::uniform_int_distribution<int> rand2(0, width - cropsize_);
  auto generator = Singleton<Context>::Instance()->rand_generator();

  const float* srcdptr = srcdata.cpu_data();
  float* dptr = data_.mutable_cpu_data();

  for (int k = 0; k < batchsize; k++) {
    int h_offset = 0, w_offset = 0;
    if (cropsize_> 0 && (flag & kTrain)) {
      h_offset = rand1(*generator);
      w_offset = rand2(*generator);
    }
    bool do_mirror = mirror_
                    && (rand1(*generator) % 2)
                    && (flag & kTrain);
    ImageTransform(srcdptr + k * srcimage_size, nullptr, do_mirror, cropsize_,
        cropsize_, h_offset, w_offset, channel, height, width,
        scale_, dptr + k * image_size);
  }
}

}  // namespace singa
