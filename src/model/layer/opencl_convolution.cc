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

RegisterLayerClass(singacl_convolution, OpenclConvolution);

/// \copydoc Layer::Forward(int flag, const Tensor&)
const Tensor OpenclConvolution::Forward(int flag, const Tensor &input) {
  CHECK(buf_.empty());
  CHECK_EQ(input.device()->lang(), kOpencl);
  CHECK_EQ(input.nDim(), 4u);
  
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
    
    col_data.device()->Exec([input, offset, col_data, this](Context* ctx) mutable {

      this->Im2Col(input.block(), offset, 
                   height_, width_,
                   kernel_h_, kernel_w_, 
                   pad_h_, pad_w_,
                   stride_h_, stride_w_,
                   conv_height_, conv_width_,
                   0, channels_,
                   col_data.block(), ctx);
    },
    {input.block()},
    {col_data.block()});
    
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
    auto tmpshp = Shape{batchsize * num_filters_, grad.Size() / (batchsize * num_filters_)};
    Tensor tmp1 = Reshape(grad, tmpshp);

    Tensor tmp2(Shape{batchsize * num_filters_}, 
                grad.device(), grad.data_type());
    SumColumns(tmp1, &tmp2);
    Tensor tmp3 = Reshape(tmp2, Shape{batchsize, num_filters_});

    SumRows(tmp3, &db);
  }
  
  Tensor col_data(Shape{col_height_, col_width_}, 
                  grad.device(), grad.data_type());
  
  for (size_t b = 0; b < batchsize; b++) {
  
    int im_offset = b * imagesize;
    int col_offset = 0; // Always keep this to zero.
    
    col_data.device()->Exec([src_data, col_data, im_offset, col_offset, this](Context* ctx) mutable {
      
      this->Im2Col(src_data.block(), im_offset, 
                   height_, width_,
                   kernel_h_, kernel_w_, 
                   pad_h_, pad_w_,
                   stride_h_, stride_w_,
                   conv_height_, conv_width_,
                   col_offset, channels_,
                   col_data.block(), ctx);
    },
    {src_data.block()},
    {col_data.block()});
    
    Tensor grad_b(Shape{num_filters_, conv_height_ * conv_width_}, 
                  grad.device(), grad.data_type());
    CopyDataToFrom(&grad_b, grad, grad_b.Size(), 0, b * grad_b.Size());
    
    dw += Mult(grad_b, col_data.T());
    Tensor dcol_b = Mult(weight_.T(), grad_b);
                         
    dx.device()->Exec([dcol_b, dx, im_offset, col_offset, this](Context* ctx) mutable {
      
      this->Col2Im(dcol_b.block(), col_offset, 
                   height_, width_,
                   kernel_h_, kernel_w_, 
                   pad_h_, pad_w_,
                   stride_h_, stride_w_,
                   conv_height_, conv_width_,
                   im_offset, channels_,
                   dx.block(), ctx);
    },
    {dcol_b.block()},
    {dx.block()});
  }
  
  param_grad.push_back(dw);
  param_grad.push_back(db);
  
  return std::make_pair(dx, param_grad);
}


void OpenclConvolution::Setup(const Shape &in_sample, const LayerConf &conf) {
  Convolution::Setup(in_sample, conf);
}


void OpenclConvolution::ToDevice(std::shared_ptr<Device> device) {
  Convolution::ToDevice(device);
}

                           
void OpenclConvolution::Im2Col(Block* src, int data_im_off, 
                               const int height, const int width,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w,
                               const int stride_h, const int stride_w,
                               const int conv_h, const int conv_w,
                               const int col_data_off, const int channels, 
                               Block* dst, Context* ctx) {

  auto ocl_ctx = viennacl::ocl::get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_im2col", "im2col");

  auto src_buf = WrapHandle(static_cast<cl_mem>(src->mutable_data()), ocl_ctx);
  auto dst_buf = WrapHandle(static_cast<cl_mem>(dst->mutable_data()), ocl_ctx);
  
  int num_kernels = channels * conv_h * conv_w;
  
  viennacl::ocl::enqueue(kernel(num_kernels, src_buf, data_im_off,
                                height, width, kernel_h, kernel_w, 
                                pad_h, pad_w, stride_h, stride_w,
                                1, 1, conv_h, conv_w,
                                dst_buf, col_data_off));
}

  
void OpenclConvolution::Col2Im(Block* src, const int col_data_off, 
                               const int height, const int width,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w,
                               const int stride_h, const int stride_w,
                               const int conv_h, const int conv_w,
                               const int data_im_off, const int channels, 
                               Block* dst, Context* ctx) {
                               
  auto ocl_ctx = viennacl::ocl::get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_im2col", "col2im");
  
  auto src_buf = WrapHandle(static_cast<cl_mem>(src->mutable_data()), ocl_ctx);
  auto dst_buf = WrapHandle(static_cast<cl_mem>(dst->mutable_data()), ocl_ctx);
  
  int num_kernels = channels * height * width;
  
  viennacl::ocl::enqueue(kernel(num_kernels, src_buf, col_data_off, channels,
                                height, width, kernel_h, kernel_w, 
                                pad_h, pad_w, stride_h, stride_w,
                                1, 1, conv_h, conv_w,
                                dst_buf, data_im_off));
}


} // namespace singa

#endif // USE_OPENCL
