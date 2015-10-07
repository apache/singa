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

#include "singa/neuralnet/input_layer/rgb_image.h"
#include "singa/neuralnet/input_layer/data.h"
#include "mshadow/tensor.h"
#include "singa/utils/image_transform.h"
#include "singa/utils/tokenizer.h"
namespace singa {

using namespace mshadow;
using mshadow::cpu;
using mshadow::Shape4;
using mshadow::Tensor;

using std::string;
using std::vector;

void RGBImageLayer::ParseRecords(int flag, const vector<Record>& records,
    Blob<float>* blob) {
  const vector<int>& s = blob->shape();
  Tensor<cpu, 4> images(data_.mutable_cpu_data(),
      Shape4(s[0], s[1], s[2], s[3]));
  const SingleLabelImageRecord& r = records.at(0).image();
  Tensor<cpu, 3> raw_image(Shape3(r.shape(0), r.shape(1), r.shape(2)));
  AllocSpace(raw_image);
  Tensor<cpu, 3> croped_image(nullptr, Shape3(s[1], s[2], s[3]));
  if (cropsize_)
    AllocSpace(croped_image);
  int rid = 0;
  const float* meandptr = mean_.cpu_data();
  for (const Record& record : records) {
    auto image = images[rid];
    bool do_crop = cropsize_> 0 && ((flag & kTrain) == kTrain);
    bool do_mirror = mirror_ && rand() % 2 && ((flag & kTrain) == kTrain);
    float* dptr = nullptr;
    if (do_crop || do_mirror)
      dptr = raw_image.dptr;
    else
      dptr = image.dptr;
    if (record.image().pixel().size()) {
      string pixel = record.image().pixel();
      for (size_t i = 0; i < pixel.size(); i++)
        dptr[i] = static_cast<float>(static_cast<uint8_t>(pixel[i]));
    } else {
      memcpy(dptr, record.image().data().data(),
          sizeof(float) * record.image().data_size());
    }
    for (int i = 0; i < mean_.count(); i++)
      dptr[i] -= meandptr[i];
    if (do_crop) {
      int hoff = rand() % (r.shape(1) - cropsize_);
      int woff = rand() % (r.shape(2) - cropsize_);
      Shape<2> cropshape = Shape2(cropsize_, cropsize_);
      if (do_mirror) {
        croped_image = expr::crop(raw_image, cropshape, hoff, woff);
        image = expr::mirror(croped_image);
      } else {
        image = expr::crop(raw_image, cropshape, hoff, woff);
      }
    } else if (do_mirror) {
      image = expr::mirror(raw_image);
    }
    rid++;
  }
  if (scale_)
    images = images * scale_;
  FreeSpace(raw_image);
  if (cropsize_)
    FreeSpace(croped_image);
}

void RGBImageLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  ParserLayer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  scale_ = proto.rgbimage_conf().scale();
  cropsize_ = proto.rgbimage_conf().cropsize();
  mirror_ = proto.rgbimage_conf().mirror();
  int batchsize = dynamic_cast<DataLayer*>(srclayers[0])->batchsize();
  Record sample = dynamic_cast<DataLayer*>(srclayers[0])->sample();
  vector<int> shape;
  shape.push_back(batchsize);
  for (int x : sample.image().shape()) {
    shape.push_back(x);
  }
  CHECK_EQ(shape.size(), 4);
  if (cropsize_) {
    shape[2] = cropsize_;
    shape[3] = cropsize_;
  }
  data_.Reshape(shape);
  mean_.Reshape({shape[1], shape[2], shape[3]});
  if (proto.rgbimage_conf().has_meanfile()) {
    if (proto.rgbimage_conf().meanfile().find("binaryproto") != string::npos) {
      CaffeBlob mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
             sizeof(float)*mean.data_size());
    } else {
      SingleLabelImageRecord mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
             sizeof(float)*mean.data_size());
    }
  } else {
    memset(mean_.mutable_cpu_data(), 0, sizeof(float) * mean_.count());
  }
}

} // namespace singa
