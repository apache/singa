/*********************************************************
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
************************************************************/
#include "batchnorm.h"

namespace singa {
RegisterLayerClass(singa_batchnorm, BatchNorm);
RegisterLayerClass(singacpp_batchnorm, BatchNorm);
RegisterLayerClass(singacuda_batchnorm, BatchNorm);
RegisterLayerClass(singacl_batchnorm, BatchNorm);
void BatchNorm::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  out_sample_shape_ = in_sample;
  factor_ = (float) conf.batchnorm_conf().factor();
  channels_ = in_sample.at(0);
  if (in_sample.size() == 3u)
    height_ = in_sample.at(1);
  else
    height_ = 1;
  if (in_sample.size() == 3u)
    width_ = in_sample.at(2);
  else
    width_ = 1;
  if (in_sample.size() == 1u)
    is_2d_ = true;
  else
    is_2d_ = false;

  bnScale_.Reshape(Shape{channels_ * height_ * width_});
  bnBias_.ResetLike(bnScale_);
  runningMean_.ResetLike(bnScale_);
  runningVariance_.ResetLike(bnScale_);

  dbnScale_.ResetLike(bnScale_);
  dbnBias_.ResetLike(bnBias_);
  // Push back params into param_values_
  // Assume the order of param is: bnScale, bnBias, runningMean, runningVariance
  for (const auto& spec : conf.param()) param_specs_.push_back(spec);
}

void BatchNorm::ToDevice(std::shared_ptr<Device> device) {
  bnScale_.ToDevice(device);
  bnBias_.ToDevice(device);
  dbnScale_.ToDevice(device);
  dbnBias_.ToDevice(device);
  runningMean_.ToDevice(device);
  runningVariance_.ToDevice(device);
}

const Tensor BatchNorm::Forward(int flag, const Tensor& input) {
  Tensor x = input.Clone();
  x.Reshape(Shape{input.shape(0), input.Size() / input.shape(0)});
  Tensor output, mean, var, xnorm;
  output.ResetLike(x);

  if ((flag & kTrain) == kTrain) {
    mean = Average(x, 0);
    runningMean_ *= 1.0f - factor_;
    Axpy(factor_, mean, &runningMean_);
    xnorm = x.Clone();
    SubRow(mean, &xnorm);
    xnorm = Square(xnorm);
    var = Average(xnorm, 0);
    runningVariance_ *= 1.0f - factor_;
    Axpy(factor_, var, &runningVariance_);
    Tensor tmp = var.Clone();
    tmp = Sqrt(tmp);
    tmp += 1e-6f;
    xnorm = x.Clone();
    SubRow(mean, &xnorm);
    DivRow(tmp, &xnorm);
    output = xnorm.Clone();
    MultRow(bnScale_, &output);
    AddRow(bnBias_, &output);
    buf_.push(x);
    buf_.push(mean);
    buf_.push(var);
    buf_.push(xnorm);
  } else {
    xnorm = x.Clone();
    SubRow(runningMean_, &xnorm);
    Tensor tmp = runningVariance_.Clone();
    tmp = Sqrt(tmp);
    tmp += 1e-6f;
    DivRow(tmp, &xnorm);
    output = xnorm.Clone();
    MultRow(bnScale_, &output);
    AddRow(bnBias_, &output);
  }

  if (!is_2d_)
    output.Reshape(Shape{output.shape(0), channels_, height_, width_});
  return output;
}

const std::pair<Tensor, vector<Tensor>> BatchNorm::Backward(
    int flag, const Tensor& grad) {
  Tensor dy = grad.Clone();
  dy.Reshape(Shape{grad.shape(0), grad.Size() / grad.shape(0)});
  Tensor xnorm = buf_.top();
  buf_.pop();
  Tensor var = buf_.top();
  buf_.pop();
  Tensor mean = buf_.top();
  buf_.pop();
  Tensor input = buf_.top();
  buf_.pop();

  Tensor dx;
  vector<Tensor> param_grad;

  if ((flag & kTrain) == kTrain) {
    // gxnrom
    Tensor gxnorm = dy.Clone();
    MultRow(bnScale_, &gxnorm);
    // gvar
    Tensor tmp = var.Clone();
    tmp += 1e-6f;
    tmp = Pow(var, -1.5f);
    tmp *= -0.5f;

    Tensor tmpx = input.Clone();
    SubRow(mean, &tmpx);

    tmpx = tmpx * gxnorm;
    MultRow(tmp, &tmpx);
    Tensor gvar;
    gvar.ResetLike(var);
    SumRows(tmpx, &gvar);
    // gmean
    tmp = var.Clone();
    tmp += 1e-6f;
    tmp = Pow(tmp, -0.5f);
    tmp *= -1.0f;
    Tensor tmpx_r;
    tmpx_r.ResetLike(tmp);
    SumRows(gxnorm, &tmpx_r);
    Tensor gmean = tmpx_r * tmp;

    tmpx = input.Clone();
    SubRow(mean, &tmpx);
    SumRows(tmpx, &tmp);
    tmp *= -2.0f / input.shape(0);
    tmp = tmp * gvar;
    gmean = gmean + tmp;
    // dx
    tmp = var.Clone();
    tmp += 1e-6f;
    tmp = Pow(tmp, -0.5f);
    dx = gxnorm.Clone();
    MultRow(tmp, &dx);

    tmpx = input.Clone();
    SubRow(mean, &tmpx);
    tmpx *= 2.0f / input.shape(0);
    MultRow(gvar, &tmpx);
    dx = dx + tmpx;

    tmp = gmean.Clone();
    tmp *= 1.0f / input.shape(0);

    AddRow(tmp, &dx);
    // dbnScale
    tmpx = dy * xnorm;
    SumRows(tmpx, &dbnScale_);
    // dbnBias
    SumRows(dy, &dbnBias_);
    param_grad.push_back(dbnScale_);
    param_grad.push_back(dbnBias_);
    Tensor dummy;
    dummy.ResetLike(runningMean_);
    dummy.SetValue(.0f);
    param_grad.push_back(dummy);
    param_grad.push_back(dummy);
  } else {
    LOG(ERROR) << "Do not call backward for evaluation phase";
  }
  if (!is_2d_)
    dx.Reshape(Shape{dx.shape(0), channels_, height_, width_});
  return std::make_pair(dx, param_grad);
}

}  // namespace
