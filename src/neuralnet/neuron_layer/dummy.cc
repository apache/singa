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

#include <glog/logging.h>
#include "singa/neuralnet/neuron_layer.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/context.h"
#include "singa/utils/singleton.h"

namespace singa {

void DummyLayer::Setup(const std::string str,
                       const vector<Layer*>& srclayers) {
  LayerProto conf;
  conf.ParseFromString(str);
  DummyLayer::Setup(conf, srclayers);
}

void DummyLayer::Setup(const LayerProto& proto,
                       const vector<Layer*>& srclayers) {
  NeuronLayer::Setup(proto, srclayers);
  if (proto.dummy_conf().input()) {  // use as input layer
    CHECK_EQ(srclayers.size(), 0);
    input_ = true;
    vector<int> shape;
    for (int s : proto.dummy_conf().shape()) shape.push_back(s);
    data_.Reshape(shape);
    grad_.ReshapeLike(data_);
  } else {
    CHECK_EQ(srclayers.size(), 1);
    data_.ReshapeLike(srclayers[0]->data(this));
    grad_.ReshapeLike(srclayers[0]->grad(this));
  }
  if (proto.dummy_conf().output()) {  // use as output layer
    output_ = true;
  }
}

void DummyLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  std::uniform_real_distribution<float> dis(0, 1);
  auto gen = Singleton<Context>::Instance()->rand_generator();
  if (input_) {
    // randomly init data with [0,1] values
    for (int i = 0; i < data_.count(); ++i)
      data_.mutable_cpu_data()[i] = dis(*gen);
  }
  if (srclayers.size() > 0)
    Copy(srclayers[0]->data(this), &data_);
}

void DummyLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  std::uniform_real_distribution<float> dis(0, 1);
  auto gen = Singleton<Context>::Instance()->rand_generator();
  if (output_) {
    // randomly init data with [0,1] values
    for (int i = 0; i < data_.count(); ++i)
      grad_.mutable_cpu_data()[i] = dis(*gen);
  }
  if (srclayers.size() > 0)
    Copy(grad_, srclayers[0]->mutable_grad(this));
}

void DummyLayer::Feed(vector<int> shape, vector<float>* data, int op){

    //batchsize_ = batchsize;
    batchsize_ = shape[0];
    // dataset
    if (op == 0) {
      /*
      size_t hdim = 1;
      for (size_t i = 1; i < shape.size(); ++i) 
          hdim *= shape[i];
      //data_.Reshape({batchsize, (int)hdim});
      //shape.insert(shape.begin(),batchsize);
      data_.Reshape(shape);
      */
      //reshape data
      data_.Reshape(shape);
      CHECK_EQ(data_.count(), data->size());

      int size = data->size();
      float* ptr = data_.mutable_cpu_data();
      for (int i = 0; i< size; i++) { 
          ptr[i] = data->at(i);
      }
    }
    // label
    else {
      aux_data_.resize(batchsize_);
      for (int i = 0; i< batchsize_; i++) {
          aux_data_[i] = static_cast<int>(data->at(i));
      }
    }

    return;

    /* Wenfeng's input
    batchsize_ = batchsize;
    shape.insert(shape.begin(),batchsize);
    data_.Reshape(shape);

    int size = data_.count() / batchsize_;
    CHECK_EQ(size, data->size());
    float* ptr = data_.mutable_cpu_data();
    for (int i = 0; i< size; i++)
	      ptr[i] = data->at(i);

    return;
    */
}

}  // namespace singa
