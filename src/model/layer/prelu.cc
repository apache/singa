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

#include "./prelu.h"

#include "singa/model/layer.h"
namespace singa {

RegisterLayerClass(singa_prelu, PReLU);
RegisterLayerClass(singacpp_prelu, PReLU);
RegisterLayerClass(singacuda_prelu, PReLU);
RegisterLayerClass(singacl_prelu, PReLU);
void PReLU::Setup(const Shape &in_sample, const LayerConf &conf) {
  Layer::Setup(in_sample, conf);
  out_sample_shape_ = in_sample;
  channel_shared_ = conf.prelu_conf().channel_shared();
  format_ = conf.prelu_conf().format();
  // Push back params into param_values_
  for (const auto &spec : conf.param()) param_specs_.push_back(spec);
  //  param_values_.push_back(a_);
}

const Tensor PReLU::Forward(int flag, const Tensor &input) {
  Tensor output;
  if (!channel_shared_) {
    size_t n, c, h, w;
    Tensor temp = (input <= 0.f);
    if (temp.nDim() == 4) {
      if (format_ == "NCHW") {
        n = temp.shape(0);
        c = temp.shape(1);
        h = temp.shape(2);
        w = temp.shape(3);
        temp.Reshape(Shape{n * c, h * w});
        Tensor temp_a(Shape{n, c}, input.device(), input.data_type());
        Uniform(1.f, 1.f, &temp_a);
        MultRow(a_, &temp_a);
        temp_a.Reshape(Shape{n * c});
        MultColumn(temp_a, &temp);
      } else if (format_ == "NHWC") {
        n = temp.shape(0);
        h = temp.shape(1);
        w = temp.shape(2);
        c = temp.shape(3);
        temp.Reshape(Shape{n * h * w, c});
        MultRow(a_, &temp);
      } else {
        LOG(FATAL) << "Incorrect input format for prelu layer.";
      }
    } else {
      LOG(FATAL) << "Incorrect input format for prelu layer.";
    }
    temp.Reshape(input.shape());
    output = input * ((input > 0.f) + temp);
  } else {
    // share the first param of Tensor A along all channels
    LOG(FATAL) << "Not implemented";
    // TODO(wangwei) cannot access the data in this way. The data could be on
    // GPU.
    auto a = a_.data<float>()[0];
    output = input * ((input > 0.f) + (input <= 0.f) * a);
  }
  if (flag & kTrain) buf_.push(input);
  return output;
}

const std::pair<Tensor, vector<Tensor> > PReLU::Backward(int flag,
                                                         const Tensor &grad) {
  vector<Tensor> param_grad;
  CHECK(!buf_.empty());
  Tensor input_grad, input = buf_.top();
  buf_.pop();
  Tensor da;
  da.ResetLike(a_);
  if (!channel_shared_) {
    size_t n = 0, c = 0, h = 0, w = 0;
    Tensor temp1 = (input <= 0.f);
    if (temp1.nDim() == 4) {
      if (format_ == "NCHW") {
        n = temp1.shape(0);
        c = temp1.shape(1);
        h = temp1.shape(2);
        w = temp1.shape(3);
        temp1.Reshape(Shape{n * c, h * w});
        Tensor temp_a(Shape{n, c}, grad.device(), grad.data_type());
        Uniform(1.f, 1.f, &temp_a);
        MultRow(a_, &temp_a);
        temp_a.Reshape(Shape{n * c});
        MultColumn(temp_a, &temp1);
        temp1.Reshape(Shape{n, c, h, w});
      } else if (format_ == "NHWC") {
        n = temp1.shape(0);
        h = temp1.shape(1);
        w = temp1.shape(2);
        c = temp1.shape(3);
        temp1.Reshape(Shape{n * h * w, c});
        MultRow(a_, &temp1);
        temp1.Reshape(Shape{n, h, w, c});
      } else {
        LOG(FATAL) << "Incorrect input format for prelu layer.";
      }
    } else {
      LOG(FATAL) << "Incorrect input format for prelu layer.";
    }
    input_grad = grad * input * ((input > 0.f) + temp1);
    Tensor temp2 = grad * input * (input <= 0.f);
    if (format_ == "NCHW") {
      Tensor temp3(Shape{n * c}, grad.device(), grad.data_type());
      temp2.Reshape(Shape{n * c, h * w});
      SumColumns(temp2, &temp3);
      temp3.Reshape(Shape{n, c});
      SumRows(temp3, &da);
    } else if (format_ == "NHWC") {
      temp2.Reshape(Shape{n * h * w, c});
      SumRows(temp2, &da);
    }
  } else {
    // share the first param of Tensor A along all channels
    LOG(FATAL) << "Not Implemented";
    // TODO(wangwei) cannot access the data in this way. The data could be on
    // GPU.
    auto a = a_.data<float>()[0];
    input_grad = grad * input * ((input > 0.f) + (input <= 0.f) * a);
    Tensor temp = grad * input * (input <= 0.f);
    float sum = Sum<float>(temp);
    Uniform(1.f, 1.f, &da);
    da *= sum;
  }
  param_grad.push_back(da);
  return std::make_pair(input_grad, param_grad);
}

void PReLU::ToDevice(std::shared_ptr<Device> device) {
  Layer::ToDevice(device);
  a_.ToDevice(device);
}

}  // namespace singa
