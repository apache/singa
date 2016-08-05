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
 
#include "opencl_pooling.h"

#ifdef USE_OPENCL

namespace singa {

RegisterLayerClass(opencl_pooling, OpenclPooling);

const Tensor OpenclPooling::Forward(int flag, const Tensor &input) {
  CHECK(buf_.empty());
  CHECK_EQ(input.device()->lang(), kOpencl);
  CHECK_EQ(input.nDim(), 4u);
  
  auto batchsize = input.shape(0);
  auto data_type = input.data_type();
  auto device = input.device();

  Shape shape{batchsize, channels_, pooled_height_, pooled_width_};
  Tensor output = Tensor(shape, device, data_type);
  
  output.device()->Exec([input, output, flag, this](Context *ctx) {
    Block* in_block = input.block();
    Block* outblock = output.block();

    if (pool_ == PoolingConf_PoolMethod_MAX) {
      Tensor mask;
      mask.ResetLike(output);
      
      Pooling_Forward_Max(4, in_block, mask.block(), 
                          height_, width_, pooled_height_, pooled_width_,
                          kernel_h_, kernel_w_, stride_h_, stride_w_,
                          pad_h_, pad_w_,
                          outblock, channels_, ctx);
      
      if (flag & kTrain)
        buf_.push(mask);
      
    } else if (pool_ == PoolingConf_PoolMethod_AVE) {
      Pooling_Forward_Ave(in_block, outblock,
                          height_, width_, pooled_height_, pooled_width_,
                          kernel_h_, kernel_w_, stride_h_, stride_w_,
                          pad_h_, pad_w_, channels_, ctx);
    } else
      LOG(FATAL) << "Unknown pooling method.";
    
  }, {input.block()}, {output.block()});
  
  if (flag & kTrain) {
    buf_.push(input);
    buf_.push(output);
  }
  
  return output;
}


const std::pair<Tensor, std::vector<Tensor>>
OpenclPooling::Backward(int flag, const Tensor &grad) {
  CHECK(!buf_.empty());
  CHECK_EQ(grad.device()->lang(), kOpencl);
  CHECK_EQ(grad.nDim(), 4u);
  
  std::vector<Tensor> param_grad;
  
  Tensor y = buf_.top();
  buf_.pop();
  Tensor x = buf_.top();
  buf_.pop();
  Tensor dx;
  dx.ResetLike(x);

  dx.device()->Exec([dx, grad, x, y, this](Context *ctx) {
    Block* dy_block = grad.block();
    Block* dx_block = dx.block();
    Block* y_block = y.block();
    Block* x_block = x.block();
    
    if (pool_ == PoolingConf_PoolMethod_MAX) {
      Pooling_Backward_Max(y_block, dy_block,
                           height_, width_,
                           pooled_height_, pooled_width_,
                           kernel_h_, kernel_w_,
                           stride_h_, stride_w_,
                           pad_h_, pad_w_,
                           channels_, x_block, dx_block, ctx);
    } else if (pool_ == PoolingConf_PoolMethod_AVE) {
      Pooling_Backward_Ave(y_block, x_block, 
                           height_, width_,
                           pooled_height_, pooled_width_,
                           kernel_h_, kernel_w_,
                           stride_h_, stride_w_,
                           pad_h_, pad_w_,
                           channels_, ctx);
    } else
      LOG(FATAL) << "Unknown pooling method.";
    
  }, {grad.block(), y.block(), x.block()}, {dx.block()});

  return std::make_pair(dx, param_grad);
}


void OpenclPooling::Setup(const Shape& in_sample, const LayerConf &conf) {
  Pooling::Setup(in_sample, conf);
  auto pool_conf = conf.pooling_conf();
}


void OpenclPooling::Pooling_Forward_Max(const int num, Block* src, Block* mask, 
                                        const int height, const int width,
                                        const int pooled_h, const int pooled_w,
                                        const int kernel_h, const int kernel_w,
                                        const int stride_h, const int stride_w,
                                        const int pad_h, const int pad_w,
                                        Block* dst, const int channels,
                                        Context* ctx) {
  cl_int status = CL_SUCCESS;
  
  std::string kname = "max_pool_forward";
  auto kernel = ctx->kernels->at(kname);

  cl::Buffer src_buf = *(static_cast<cl::Buffer*>(src->mutable_data()));
  cl::Buffer dst_buf = *(static_cast<cl::Buffer*>(dst->mutable_data()));
  cl::Buffer mask_buf = *(static_cast<cl::Buffer*>(mask->mutable_data()));

  kernel.setArg(0, pooled_h * pooled_w);
  kernel.setArg(1, src_buf);
  kernel.setArg(2, channels);
  kernel.setArg(3, height);
  kernel.setArg(4, width);
  kernel.setArg(5, pooled_h);
  kernel.setArg(6, pooled_w);
  kernel.setArg(7, kernel_h);
  kernel.setArg(8, kernel_w);
  kernel.setArg(9, stride_h);
  kernel.setArg(10, stride_w);
  kernel.setArg(11, pad_h);
  kernel.setArg(12, pad_w);
  kernel.setArg(13, dst_buf);
  kernel.setArg(14, mask_buf);
  
  cl::NDRange global(num);
  std::cout << "NUM: " << num << std::endl;
  std::cout << "Pooled Ht: " << pooled_h << " Wdt: " << pooled_w << std::endl;
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


void OpenclPooling::Pooling_Forward_Ave(Block* src, Block* dst, 
                                        const int height, const int width,
                                        const int pooled_h, const int pooled_w,
                                        const int kernel_h, const int kernel_w,
                                        const int stride_h, const int stride_w,
                                        const int pad_h, const int pad_w,
                                        const int channels, Context* ctx) {
  cl_int status = CL_SUCCESS;
  
  std::string kname = "ave_pool_forward";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer src_buf = *(static_cast<cl::Buffer*>(src->mutable_data()));
  cl::Buffer dst_buf = *(static_cast<cl::Buffer*>(dst->mutable_data()));

  kernel.setArg(0, height * width);
  kernel.setArg(1, src_buf);
  kernel.setArg(2, channels);
  kernel.setArg(3, height);
  kernel.setArg(4, width);
  kernel.setArg(5, pooled_h);
  kernel.setArg(6, pooled_w);
  kernel.setArg(7, kernel_h);
  kernel.setArg(8, kernel_w);
  kernel.setArg(9, stride_h);
  kernel.setArg(10, stride_w);
  kernel.setArg(11, pad_h);
  kernel.setArg(12, pad_w);
  kernel.setArg(13, dst_buf);
  
  cl::NDRange global(height * width);
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


void OpenclPooling::Pooling_Forward_Sto_Train(Block* src, Block* rand,
                                              const int height, const int width,
                                              const int pooled_h, const int pooled_w,
                                              const int kernel_h, const int kernel_w,
                                              const int stride_h, const int stride_w,
                                              const int channels, 
                                              Block* dst, Context* ctx) {

  cl_int status = CL_SUCCESS;
  
  std::string kname = "sto_pool_forward_train";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer src_buf = *(static_cast<cl::Buffer*>(src->mutable_data()));
  cl::Buffer dst_buf = *(static_cast<cl::Buffer*>(dst->mutable_data()));
  cl::Buffer rand_buf = *(static_cast<cl::Buffer*>(rand->mutable_data()));

  kernel.setArg(0, height * width);
  kernel.setArg(1, src_buf);
  kernel.setArg(2, channels);
  kernel.setArg(3, height);
  kernel.setArg(4, width);
  kernel.setArg(5, pooled_h);
  kernel.setArg(6, pooled_w);
  kernel.setArg(7, kernel_h);
  kernel.setArg(8, kernel_w);
  kernel.setArg(9, stride_h);
  kernel.setArg(10, stride_w);
  kernel.setArg(11, rand_buf);
  kernel.setArg(12, dst_buf);
  
  cl::NDRange global(height * width);
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


void OpenclPooling::Pooling_Forward_Sto_Test(Block* src, Block* dst, 
                                             const int height, const int width,
                                             const int pooled_h, const int pooled_w,
                                             const int kernel_h, const int kernel_w,
                                             const int stride_h, const int stride_w,
                                             const int channels, Context* ctx) {
  
  cl_int status = CL_SUCCESS;
  
  std::string kname = "sto_pool_forward_test";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer src_buf = *(static_cast<cl::Buffer*>(src->mutable_data()));
  cl::Buffer dst_buf = *(static_cast<cl::Buffer*>(dst->mutable_data()));

  kernel.setArg(0, height * width);
  kernel.setArg(1, src_buf);
  kernel.setArg(2, channels);
  kernel.setArg(3, height);
  kernel.setArg(4, width);
  kernel.setArg(5, pooled_h);
  kernel.setArg(6, pooled_w);
  kernel.setArg(7, kernel_h);
  kernel.setArg(8, kernel_w);
  kernel.setArg(9, stride_h);
  kernel.setArg(10, stride_w);
  kernel.setArg(11, dst_buf);
  
  cl::NDRange global(height * width);
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


void OpenclPooling::Pooling_Backward_Max(Block* src, Block* srcDiff,
                                         const int height, const int width,
                                         const int pooled_h, const int pooled_w,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int pad_h, const int pad_w,
                                         const int channels, 
                                         Block* dst, Block* dstDiff,
                                         Context* ctx) {
  cl_int status = CL_SUCCESS;
  
  std::string kname = "max_pool_backward";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer src_buf = *(static_cast<cl::Buffer*>(src->mutable_data()));
  cl::Buffer srcDiff_buf = *(static_cast<cl::Buffer*>(srcDiff->mutable_data()));
  cl::Buffer dst_buf = *(static_cast<cl::Buffer*>(dst->mutable_data()));
  cl::Buffer dstDiff_buf = *(static_cast<cl::Buffer*>(dstDiff->mutable_data()));
  
  kernel.setArg(0, height * width);
  kernel.setArg(1, srcDiff_buf);
  kernel.setArg(2, cl_int(0)); // use_mask
  kernel.setArg(3, src_buf);
  kernel.setArg(4, dst_buf);
  kernel.setArg(5, channels);
  kernel.setArg(6, height);
  kernel.setArg(7, width);
  kernel.setArg(8, pooled_h);
  kernel.setArg(9, pooled_w);
  kernel.setArg(10, kernel_h);
  kernel.setArg(11, kernel_w);
  kernel.setArg(12, stride_h);
  kernel.setArg(13, stride_w);
  kernel.setArg(14, pad_h);
  kernel.setArg(15, pad_w);
  kernel.setArg(16, dstDiff_buf);
  
  cl::NDRange global(height * width);
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


void OpenclPooling::Pooling_Backward_Ave(Block* src, Block* dst, 
                                         const int height, const int width,
                                         const int pooled_h, const int pooled_w,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int pad_h, const int pad_w,
                                         const int channels, Context* ctx) {
  cl_int status = CL_SUCCESS;
  
  std::string kname = "ave_pool_backward";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer src_buf = *(static_cast<cl::Buffer*>(src->mutable_data()));
  cl::Buffer dst_buf = *(static_cast<cl::Buffer*>(dst->mutable_data()));
  
  kernel.setArg(0, height * width);
  kernel.setArg(1, src_buf);
  kernel.setArg(2, channels);
  kernel.setArg(3, height);
  kernel.setArg(4, width);
  kernel.setArg(5, pooled_h);
  kernel.setArg(6, pooled_w);
  kernel.setArg(7, kernel_h);
  kernel.setArg(8, kernel_w);
  kernel.setArg(9, stride_h);
  kernel.setArg(10, stride_w);
  kernel.setArg(11, pad_h);
  kernel.setArg(12, pad_w);
  kernel.setArg(13, dst_buf);
  
  cl::NDRange global(height * width);
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


void OpenclPooling::Pooling_Backward_Sto(Block* src, Block* rand, Block* dst,
                                         const int height, const int width,
                                         const int pooled_h, const int pooled_w,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int pad_h, const int pad_w,
                                         const int channels, Context* ctx) {
  cl_int status = CL_SUCCESS;
  
  std::string kname = "sto_pool_backward";
  auto kernel = ctx->kernels->at(kname);
  
  cl::Buffer src_buf = *(static_cast<cl::Buffer*>(src->mutable_data()));
  cl::Buffer dst_buf = *(static_cast<cl::Buffer*>(dst->mutable_data()));
  cl::Buffer rand_buf = *(static_cast<cl::Buffer*>(rand->mutable_data()));
  
  kernel.setArg(0, height * width);
  kernel.setArg(1, rand_buf);
  kernel.setArg(2, src_buf);
  kernel.setArg(3, channels);
  kernel.setArg(4, height);
  kernel.setArg(5, width);
  kernel.setArg(6, pooled_h);
  kernel.setArg(7, pooled_w);
  kernel.setArg(8, kernel_h);
  kernel.setArg(9, kernel_w);
  kernel.setArg(10, stride_h);
  kernel.setArg(11, stride_w);
  kernel.setArg(12, pad_h);
  kernel.setArg(13, pad_w);
  kernel.setArg(14, dst_buf);
  
  cl::NDRange global(height * width);
  
  status = ctx->ocl_cmdq.enqueueNDRangeKernel(kernel, cl::NDRange(0), global);
  OCL_CHECK(status, "Failed to enqueue kernel function!");
}


} // namespace singa

#endif // USE_OPENCL
