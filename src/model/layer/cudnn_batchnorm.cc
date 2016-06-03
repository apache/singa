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
#include "cudnn_batchnorm.h"
#ifdef USE_CUDNN

namespace singa {

CudnnBatchNorm::~CudnnBatchNorm() {
  if (has_init_cudnn_) {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(shape_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(param_desc_));
  }
}

void CudnnBatchNorm::ToDevice(Device* device) {
  BatchNorm::ToDevice(device);
  resultSaveMean_.ToDevice(device);
  resultSaveVariance_.ToDevice(device);
}

void CudnnBatchNorm::Setup(const LayerConf& conf) {
  BatchNorm::Setup(conf);
  bnScale_.Reshape(Shape{1,channels_,1,1});
  bnBias_.ResetLike(bnScale_);
  dbnScale_.ResetLike(bnScale_);
  dbnBias_.ResetLike(bnScale_);
  runningMean_.ResetLike(bnScale_);
  runningVariance_.ResetLike(bnScale_);
  resultSaveMean_.ResetLike(bnScale_);
  resultSaveVariance_.ResetLike(bnScale_);
}

void CudnnBatchNorm::InitCudnn(const Shape& shape, DataType dtype) {
  CHECK(!has_init_cudnn_);
  mode_ = CUDNN_BATCHNORM_SPATIAL;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&shape_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&param_desc_));
  CHECK_EQ(shape.size(), 4u);
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(shape_desc_,
        CUDNN_TENSOR_NCHW,
        GetCudnnDataType(dtype),
        shape[0],
        shape[1],
        shape[2],
        shape[3]));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(param_desc_,
        CUDNN_TENSOR_NCHW,
        GetCudnnDataType(dtype),
        1,
        shape[1],
        1,
        1));
  has_init_cudnn_ = true;
}
const Tensor CudnnBatchNorm::Forward(int flag, const Tensor& input) {
  auto shape = input.shape();
  auto dtype = input.data_type();
  Tensor output;
  if (!has_init_cudnn_)
    InitCudnn(shape, dtype);
  // TODO(wangji): check device id of input and params
  output.ResetLike(input);
  if ((flag & kTrain) == kTrain) {
    output.device()->Exec(
        [=](Context* ctx) {
          Blob *inBlob = input.blob(), *outBlob = output.blob(),
            *saveMeanBlob = resultSaveMean_.blob(),
            *saveVarBlob = resultSaveVariance_.blob(),
            *runningMeanBlob = runningMean_.blob(),
            *runningVarBlob = runningVariance_.blob(),
            *bnScaleBlob = bnScale_.blob(),
            *bnBiasBlob = bnBias_.blob();
          const float alpha = 1.0f, beta = 0.0f;
          double epsilon = CUDNN_BN_MIN_EPSILON;
          CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
              ctx->cudnn_handle,
              this->mode_,
              &alpha,
              &beta,
              shape_desc_,
              inBlob->data(),
              shape_desc_,
              outBlob->mutable_data(),
              param_desc_,
              bnScaleBlob->data(),
              bnBiasBlob->data(),
              factor_,
              runningMeanBlob->mutable_data(),
              runningVarBlob->mutable_data(),
              epsilon,
              saveMeanBlob->mutable_data(),
              saveVarBlob->mutable_data()));
        },
        {input.blob(),
         bnScale_.blob(),
         bnBias_.blob()},
        {output.blob(),
         runningMean_.blob(),
         runningVariance_.blob(),
         resultSaveMean_.blob(),
         resultSaveVariance_.blob()});
    buf_.push(input);
  } else {
    output.device()->Exec(
        [=](Context* ctx) {
          Blob *inBlob = input.blob(), *outBlob = output.blob(),
            *runningMeanBlob = runningMean_.blob(),
            *runningVarBlob = runningVariance_.blob(),
            *bnScaleBlob = bnScale_.blob(),
            *bnBiasBlob = bnBias_.blob();
          const float alpha = 1.0f, beta = 0.0f;
          double epsilon = CUDNN_BN_MIN_EPSILON;
          CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
              ctx->cudnn_handle,
              this->mode_,
              &alpha,
              &beta,
              shape_desc_,
              inBlob->data(),
              shape_desc_,
              outBlob->mutable_data(),
              param_desc_,
              bnScaleBlob->data(),
              bnBiasBlob->data(),
              runningMeanBlob->data(),
              runningVarBlob->data(),
              epsilon));
        },
        {input.blob(),
         bnScale_.blob(),
         bnBias_.blob(),
         runningMean_.blob(),
         runningVariance_.blob()},
        {output.blob()});
  }
  return output;
}

const std::pair<Tensor, vector<Tensor>> CudnnBatchNorm::Backward(
    int flag, const Tensor& grad) {
  vector <Tensor> param_grad;
  Tensor dx;
  if ((flag & kTrain) == kTrain) {
    Tensor input = buf_.top();
    buf_.pop();
    dx.ResetLike(grad);
    dx.device()->Exec(
        [=](Context* ctx) {
          Blob *dyblob = grad.blob(), *dxblob = dx.blob(),
            *xblob = input.blob(),
            *bnScaleBlob = bnScale_.blob(),
            *dbnScaleBlob = dbnScale_.blob(),
            *dbnBiasBlob = dbnBias_.blob(),
            *saveMeanBlob = resultSaveMean_.blob(),
            *saveVarBlob = resultSaveVariance_.blob();
          const float alpha = 1.0f, beta = .0f;
          double epsilon = CUDNN_BN_MIN_EPSILON;
          CUDNN_CHECK(cudnnBatchNormalizationBackward(ctx->cudnn_handle,
              this->mode_,
              &alpha,
              &beta,
              &alpha,
              &beta,
              shape_desc_,
              xblob->data(),
              shape_desc_,
              dyblob->data(),
              shape_desc_,
              dxblob->mutable_data(),
              param_desc_,
              bnScaleBlob->data(),
              dbnScaleBlob->mutable_data(),
              dbnBiasBlob->mutable_data(),
              epsilon,
              saveMeanBlob->data(),
              saveVarBlob->data()));

        },
        {dx.blob(),
         grad.blob(),
         bnScale_.blob(),
         resultSaveMean_.blob(),
         resultSaveVariance_.blob()},
        {dx.blob(),
         dbnScale_.blob(),
         dbnBias_.blob()});
  } else {
    LOG(ERROR) << "Do not call backward for evaluation phase";
  }
  param_grad.push_back(dbnScale_);
  param_grad.push_back(dbnBias_);
  return std::make_pair(dx, param_grad);
}
}  // namespace

#endif  // USE_CUDNN
