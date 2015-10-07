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

#include "singa/neuralnet/input_layer/mnist.h"
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

void MnistLayer::ParseRecords(int flag, const vector<Record>& records,
    Blob<float>* blob) {
  LOG_IF(ERROR, records.size() == 0) << "Empty records to parse";
  int ndim = records.at(0).image().shape_size();
  int inputsize = records.at(0).image().shape(ndim-1);
  CHECK_EQ(inputsize, blob->shape()[2]);

  float* dptr = blob->mutable_cpu_data();
  for (const Record& record : records) {
    const SingleLabelImageRecord& imagerecord = record.image();
    if (imagerecord.pixel().size()) {
      string pixel = imagerecord.pixel();
      for (int i = 0, k = 0; i < inputsize; i++) {
        for (int j = 0; j < inputsize; j++) {
          // NOTE!!! must cast pixel to uint8_t then to float!!! waste a lot of
          // time to debug this
          float x =  static_cast<float>(static_cast<uint8_t>(pixel[k++]));
          x = x / norm_a_-norm_b_;
          *dptr = x;
          dptr++;
        }
      }
    } else {
      for (int i = 0, k = 0; i < inputsize; i++) {
        for (int j = 0; j < inputsize; j++) {
          *dptr = imagerecord.data(k++) / norm_a_ - norm_b_;
          dptr++;
        }
      }
    }
  }
  CHECK_EQ(dptr, blob->mutable_cpu_data() + blob->count());
}

void MnistLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  Layer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  int batchsize = dynamic_cast<DataLayer*>(srclayers[0])->batchsize();
  Record sample = dynamic_cast<DataLayer*>(srclayers[0])->sample();
  norm_a_ = proto.mnist_conf().norm_a();
  norm_b_ = proto.mnist_conf().norm_b();
  int ndim = sample.image().shape_size();
  CHECK_GE(ndim, 2);
  int s = sample.image().shape(ndim - 1);
  CHECK_EQ(s, sample.image().shape(ndim - 2));
  data_.Reshape(vector<int>{batchsize, 1, s, s});
}

} // namespace singa
