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
 
#include "opencl_convolution.h"

#ifdef USE_OPENCL

namespace singa {

RegisterLayerClass(opencl_convolution, OpenclConvolution);

/// \copydoc Layer::Forward(int flag, const Tensor&)
const Tensor OpenclConvolution::Forward(int flag, const Tensor &input) {
  CHECK(buf_.empty());
  CHECK_EQ(input.device()->lang(), kOpencl);
  CHECK_EQ(input.nDim(), 4u);
  std::cout << "Forward!" << std::endl;
  if (flag & kTrain) buf_.push(input);
  
  auto batchsize = input.shape(0);
  auto imagesize = input.Size() / batchsize;
  auto data_type = input.data_type();
  auto device = input.device();
  
  Shape shape{batchsize, num_filters_, conv_height_, conv_width_};
  Tensor output(shape, device, data_type);
  Tensor col_data(Shape{col_height_, col_width_}, device, data_type);
  
  for (size_t b = 0; b < batchsize; b++) {
    int offset = b * imagesize;
    output.device()->Exec([input, output, offset, col_data, this](Context* ctx) mutable {

      this->Im2Col(input.block(), offset, 
                   height_, width_,
                   kernel_h_, kernel_w_, 
                   pad_h_, pad_w_,
                   stride_h_, stride_w_,
                   1, 1, // dilation h w
                   col_height_, col_width_,
                   0, channels_,
                   col_data.block(), ctx);
    },
    {input.block(), weight_.block()},
    {output.block()});
    
    Tensor each = Mult(weight_, col_data);

    if (bias_term_) {
      AddColumn(bias_, &each);
    }
    
    CopyDataToFrom(&output, each, each.Size(), b * each.Size());
  }
  
  return output;
}


/// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
const std::pair<Tensor, std::vector<Tensor>>
OpenclConvolution::Backward(int flag, const Tensor &grad) {
  CHECK(!buf_.empty());
  CHECK_EQ(grad.device()->lang(), kOpencl);
  CHECK_EQ(grad.nDim(), 4u);
  
  std::vector<Tensor> param_grad;
  
  Tensor src_data = buf_.top();
  buf_.pop();
  
  Tensor dx, db, dw;
  dx.ResetLike(src_data);
  db.ResetLike(bias_);
  dw.ResetLike(weight_);
  dw.SetValue(0.0f);
  
  size_t batchsize = grad.shape(0);
  size_t imagesize = src_data.Size() / batchsize;
  
  if (bias_term_) {
    dx.device()->Exec([grad, db, batchsize, this](Context* ctx) mutable {
      Tensor tmp1 = Reshape(grad, Shape{batchsize * num_filters_,
                            grad.Size() / (batchsize * num_filters_)});
      Tensor tmp2(Shape{batchsize * num_filters_}, grad.device(), grad.data_type());
      SumColumns(tmp1, &tmp2);
      Tensor tmp3 = Reshape(tmp2, Shape{batchsize, num_filters_});
      SumRows(tmp3, &db);
    },
    {grad.block(), src_data.block()},
    {dw.block()});
  }
  
  Tensor data_col;
  data_col.ResetLike(src_data);
  
  for (size_t b = 0; b < batchsize; b++) {
  
    dx.device()->Exec([grad, dx, dw, src_data, data_col, b, imagesize, this](Context* ctx) mutable {
      int offset = b * imagesize;
      
      this->Im2Col(src_data.block(), offset, 
                   height_, width_,
                   kernel_h_, kernel_w_, 
                   pad_h_, pad_w_,
                   stride_h_, stride_w_,
                   1, 1, // dilation h w
                   col_height_, col_width_,
                   0, channels_,
                   data_col.block(), ctx);
    
      Tensor grad_b(Shape{num_filters_, conv_height_ * conv_width_}, grad.device(), grad.data_type());
      CopyDataToFrom(&grad_b, grad, grad_b.Size(), 0, b * grad_b.Size());
      
      dw += Mult(grad_b, data_col.T());
      
      Tensor dcol_b = Mult(weight_.T(), grad_b);
      
      this->Col2Im(dcol_b.block(), 0, 
                   height_, width_,
                   kernel_h_, kernel_w_, 
                   pad_h_, pad_w_,
                   stride_h_, stride_w_,
                   1, 1, // dilation h w
                   col_height_, col_width_,
                   offset, channels_,
                   dx.block(), ctx);
      
    },
    {grad.block(), src_data.block()},
    {dw.block()});
  
  }
  
  param_grad.push_back(dw);
  param_grad.push_back(db);
  
  return std::make_pair(dx, param_grad);
}


void OpenclConvolution::Setup(const Shape &in_sample, const LayerConf &conf) {
  Convolution::Setup(in_sample, conf);
  auto conv_conf = conf.convolution_conf();
}


void OpenclConvolution::ToDevice(std::shared_ptr<Device> device) {
  Convolution::ToDevice(device);
}

                           
void OpenclConvolution::Im2Col(Block* src, int data_im_off, 
                               const int height, const int width,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w,
                               const int stride_h, const int stride_w,
                               const int dilation_h, const int dilation_w,
                               const int col_h, const int col_w,
                               const int data_col_off, const int channels, 
                               Block* dst, Context* ctx) {
  cl_int status = CL_SUCCESS;
  
  std::string kname = "im2col";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer src_buf = *(static_cast<cl::Buffer*>(src->mutable_data()));
  cl::Buffer dst_buf = *(static_cast<cl::Buffer*>(dst->mutable_data()));

  kernel.setArg(0, height * width);
  kernel.setArg(1, src_buf);
  kernel.setArg(2, data_im_off);
  kernel.setArg(3, height);
  kernel.setArg(4, width);
  kernel.setArg(5, kernel_h);
  kernel.setArg(6, kernel_w);
  kernel.setArg(7, pad_h);
  kernel.setArg(8, pad_w);
  kernel.setArg(9, stride_h);
  kernel.setArg(10, stride_w);
  kernel.setArg(11, dilation_h);
  kernel.setArg(12, dilation_w);
  kernel.setArg(13, col_h);
  kernel.setArg(14, col_w);
  kernel.setArg(15, dst_buf);
  kernel.setArg(16, data_col_off);
  
  cl::NDRange global(height * width);
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function im2col!");
}

  
void OpenclConvolution::Col2Im(Block* src, const int data_col_off, 
                               const int height, const int width,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w,
                               const int stride_h, const int stride_w,
                               const int dilation_h, const int dilation_w,
                               const int col_h, const int col_w,
                               const int data_im_off, const int channels, 
                               Block* dst, Context* ctx) {
  cl_int status = CL_SUCCESS;
  
  std::string kname = "col2im";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer src_buf = *(static_cast<cl::Buffer*>(src->mutable_data()));
  cl::Buffer dst_buf = *(static_cast<cl::Buffer*>(dst->mutable_data()));

  kernel.setArg(0, height * width);
  kernel.setArg(1, src_buf);
  kernel.setArg(2, data_col_off);
  kernel.setArg(3, channels);
  kernel.setArg(4, height);
  kernel.setArg(5, width);
  kernel.setArg(6, kernel_h);
  kernel.setArg(7, kernel_w);
  kernel.setArg(8, pad_h);
  kernel.setArg(9, pad_w);
  kernel.setArg(10, stride_h);
  kernel.setArg(11, stride_w);
  kernel.setArg(12, dilation_h);
  kernel.setArg(13, dilation_w);
  kernel.setArg(14, col_h);
  kernel.setArg(15, col_w);
  kernel.setArg(16, dst_buf);
  kernel.setArg(17, data_im_off);
  
  cl::NDRange global(height * width);

  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function col2im!");
}


} // namespace singa

#endif // USE_OPENCL
