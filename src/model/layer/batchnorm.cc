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

#include <vector>

namespace singa {
RegisterLayerClass(singa_batchnorm, BatchNorm);
RegisterLayerClass(singacpp_batchnorm, BatchNorm);
RegisterLayerClass(singacuda_batchnorm, BatchNorm);
RegisterLayerClass(singacl_batchnorm, BatchNorm);
void BatchNorm::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  out_sample_shape_ = in_sample;
  factor_ = (float)conf.batchnorm_conf().factor();
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

  bnScale_.Resize(Shape{channels_});
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
  Tensor output;
  output.ResetLike(x);
  // TODO(wangwei) input sample shape check
  if ((flag & kTrain) == kTrain) {  // forward for train
    if (is_2d_) {                   // batchnorm_per_activation mode
      auto mean = Average(x, 0);
      runningMean_ *= 1.0f - factor_;
      Axpy(factor_, mean, &runningMean_);
      auto xnorm = x.Clone();
      SubRow(mean, &xnorm);
      xnorm = Square(xnorm);
      auto var = Average(xnorm, 0);
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
    } else {  // batchnorm_spatial mode
      LOG(FATAL) << "Trainning SpatialBatchNormalization has not been "
                    "implemented yet...";
    }
  } else {         // forward for test
    if (is_2d_) {  // batchnorm_per_activation mode
      auto xnorm = x.Clone();
      SubRow(runningMean_, &xnorm);
      Tensor tmp = runningVariance_.Clone();
      tmp = Sqrt(tmp);
      tmp += 1e-6f;
      DivRow(tmp, &xnorm);
      output = xnorm.Clone();
      MultRow(bnScale_, &output);
      AddRow(bnBias_, &output);
    } else {  // batchnorm_spatial mode
      runningMean_.Reshape(Shape{channels_, 1});
      runningVariance_.Reshape(Shape{channels_, 1});
      bnScale_.Reshape(Shape{channels_, 1});
      bnBias_.Reshape(Shape{channels_, 1});

      std::vector<Tensor> mean_stack, var_stack, scale_stack, bias_stack;
      for (unsigned i = 0; i < height_ * width_; ++i) {
        mean_stack.push_back(runningMean_);
        var_stack.push_back(runningVariance_);
        scale_stack.push_back(bnScale_);
        bias_stack.push_back(bnBias_);
      }
      auto mean = ConcatenateColumns(mean_stack);
      auto var = ConcatenateColumns(var_stack);
      auto scale = ConcatenateColumns(scale_stack);
      auto bias = ConcatenateColumns(bias_stack);

      mean.Reshape(Shape{channels_ * height_ * width_});
      var.Reshape(Shape{channels_ * height_ * width_});
      scale.Reshape(Shape{channels_ * height_ * width_});
      bias.Reshape(Shape{channels_ * height_ * width_});

      auto xnorm = x.Clone();
      SubRow(mean, &xnorm);
      var = Sqrt(var);
      var += 1e-6f;
      DivRow(var, &xnorm);
      output = xnorm.Clone();

      MultRow(scale, &output);
      AddRow(bias, &output);

      runningMean_.Reshape(Shape{channels_});
      runningVariance_.Reshape(Shape{channels_});
      bnScale_.Reshape(Shape{channels_});
      bnBias_.Reshape(Shape{channels_});
    }
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
    if (is_2d_) {
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
      param_grad.push_back(dummy);
      param_grad.push_back(dummy);
    } else {
      LOG(FATAL) << "Trainning SpatialBatchNormalization has not been "
                    "implemented yet...";
    }
  } else {
    LOG(ERROR) << "Do not call backward for evaluation phase";
  }
  if (!is_2d_) dx.Reshape(Shape{dx.shape(0), channels_, height_, width_});
  return std::make_pair(dx, param_grad);
}

}  // namespace singa
