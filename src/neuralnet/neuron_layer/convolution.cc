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
#include "singa/utils/singleton.h"

namespace singa {
using std::vector;

/************ Implementation for ConvolutionLayer*************************/
ConvolutionLayer::~ConvolutionLayer() {
  delete weight_;
  delete bias_;
}
void ConvolutionLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  ConvolutionProto conv_conf = conf.convolution_conf();
  if (conv_conf.has_kernel()) {
    kernel_x_ = kernel_y_ = conv_conf.kernel();
  } else {
    kernel_x_ = conv_conf.kernel_x();
    kernel_y_ = conv_conf.kernel_y();
  }
  CHECK_NE(kernel_x_, 0);
  CHECK_NE(kernel_y_, 0);

  if (conv_conf.has_pad()) {
    pad_x_ = pad_y_ = conv_conf.pad();
  } else {
    pad_x_ = conv_conf.pad_x();
    pad_y_ = conv_conf.pad_y();
  }

  if (conv_conf.has_stride()) {
    stride_x_ = stride_y_ = conv_conf.stride();
  } else {
    stride_x_ = conv_conf.stride_x();
    stride_y_ = conv_conf.stride_y();
  }

  num_filters_ = conv_conf.num_filters();
  // partition filters
  if (partition_dim() > 0)
    num_filters_ /= srclayers.at(0)->num_partitions();

  const vector<int>& srcshape = srclayers[0]->data(this).shape();
  batchsize_ = srcshape[0];
  int dim = srcshape.size();
  CHECK_GT(dim, 2);
  width_ = srcshape[dim - 1];
  height_ = srcshape[dim - 2];
  if (dim > 3)
    channels_ = srcshape[dim - 3];
  else if (dim > 2)
    channels_ = 1;

  conv_height_ = (height_ + 2 * pad_y_ - kernel_y_) / stride_y_ + 1;
  conv_width_ = (width_ + 2 * pad_x_ - kernel_x_) / stride_x_ + 1;
  col_height_ = channels_ * kernel_x_ * kernel_y_;
  col_width_ = conv_height_ * conv_width_;
  vector<int> shape{batchsize_, num_filters_, conv_height_, conv_width_};
  data_.Reshape(shape);
  grad_.Reshape(shape);
  col_data_.Reshape(vector<int>{col_height_, col_width_});
  col_grad_.Reshape(vector<int>{col_height_, col_width_});
  weight_ = Param::Create(conf.param(0));
  weight_->Setup(vector<int>{num_filters_, col_height_});
  if (conf.param_size() > 1) {
    bias_ = Param::Create(conf.param(1));
    bias_->Setup(vector<int>{num_filters_});
  }
}

// TODO(wangwei) remove mshadow's functions
void ConvolutionLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  auto src = Tensor4(srclayers[0]->mutable_data(this));
  auto data = Tensor3(&data_);
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  for (int n = 0; n < batchsize_; n++) {
    if (pad_x_ > 0)
      col = expr::unpack_patch2col(pad(src[n], pad_x_), kernel_x_, stride_x_);
    else
      col = expr::unpack_patch2col(src[n], kernel_x_, stride_x_);
    data[n] = dot(weight, col);
  }
  data += expr::broadcast<1>(bias, data.shape);
}

void ConvolutionLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  auto src = Tensor4(srclayers[0]->mutable_data(this));
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());
  auto grad = Tensor3(&grad_);
  auto gcol = Tensor2(&col_grad_);
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());
  Blob<float>* gsrcblob = srclayers[0]->mutable_grad(this);
  Tensor<cpu, 4> gsrc(nullptr, Shape4(batchsize_, channels_, height_, width_));
  if (gsrcblob != nullptr)
    gsrc.dptr = gsrcblob->mutable_cpu_data();
  gbias = expr::sumall_except_dim<1>(grad);
  gweight = 0.0f;
  Shape<3> padshp(gsrc.shape.SubShape());
  padshp[0] += 2 * pad_y_;
  padshp[1] += 2 * pad_x_;
  Shape<2> imgshp = Shape2(height_, width_);
  for (int n = 0; n < batchsize_; n++) {
    if (pad_x_ > 0)
      col = expr::unpack_patch2col(pad(src[n], pad_x_), kernel_x_, stride_x_);
    else
      col = expr::unpack_patch2col(src[n], kernel_x_, stride_x_);
    gweight += dot(grad[n], col.T());
    if (gsrcblob != nullptr) {
      gcol = dot(weight.T(), grad[n]);
      gsrc[n] = crop(expr::pack_col2patch(gcol, padshp, kernel_x_, stride_x_),
          imgshp);
    }
  }
}

/******************* Implementation for CConvolutionLayer *********/
void CConvolutionLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  auto src = Tensor4(srclayers[0]->mutable_data(this));
  auto data = Tensor3(&data_);
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());

  for (int n = 0; n < batchsize_; n++) {
    Im2col(src[n].dptr, channels_, height_, width_,
        kernel_y_, kernel_x_, pad_y_, pad_x_, stride_y_, stride_x_, col.dptr);
    data[n] = dot(weight, col);
  }
  data += expr::broadcast<1>(bias, data.shape);
}

void CConvolutionLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  auto src = Tensor4(srclayers[0]->mutable_data(this));
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());

  auto grad = Tensor3(&grad_);
  auto gcol = Tensor2(&col_grad_);
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());
  gweight = 0.f;
  Blob<float>* gsrcblob = srclayers[0]->mutable_grad(this);
  Tensor<cpu, 4> gsrc(nullptr, Shape4(batchsize_, channels_, height_, width_));
  if (gsrcblob != nullptr)
    gsrc.dptr = gsrcblob->mutable_cpu_data();
  gbias = expr::sumall_except_dim<1>(grad);
  for (int n = 0; n < batchsize_; n++) {
    Im2col(src[n].dptr, channels_, height_, width_,
        kernel_y_, kernel_x_, pad_y_, pad_x_, stride_y_, stride_x_, col.dptr);
    gweight += dot(grad[n], col.T());
    if (gsrcblob != nullptr) {
      gcol = dot(weight.T(), grad[n]);
      Col2im(gcol.dptr, channels_, height_, width_,
          kernel_y_, kernel_x_, pad_y_, pad_x_, stride_y_, stride_x_,
          gsrc[n].dptr);
    }
  }
}

}  // namespace singa
