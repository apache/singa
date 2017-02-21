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

RegisterLayerClass(singacl_pooling, OpenclPooling);

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
      
      Pooling_Forward_Max((int)output.Size(), in_block, mask.block(), 
                          height_, width_,
                          pooled_height_, pooled_width_,
                          kernel_h_, kernel_w_,
                          stride_h_, stride_w_,
                          pad_h_, pad_w_,
                          outblock, channels_, ctx);
      
      if (flag & kTrain)
        buf_.push(mask);
      
    } else if (pool_ == PoolingConf_PoolMethod_AVE) {
      Pooling_Forward_Ave((int)output.Size(), in_block, outblock,
                          height_, width_, pooled_height_, pooled_width_,
                          kernel_h_, kernel_w_, stride_h_, stride_w_,
                          pad_h_, pad_w_, channels_, ctx);
    } else
      LOG(FATAL) << "Unknown pooling method.";
    
  }, {input.block()}, {output.block()});
  
  return output;
}


const std::pair<Tensor, std::vector<Tensor>>
OpenclPooling::Backward(int flag, const Tensor &grad) {
  CHECK_EQ(grad.device()->lang(), kOpencl);
  CHECK_EQ(grad.nDim(), 4u);
  
  std::vector<Tensor> param_grad;
  
  auto batchsize = grad.shape(0);
  auto data_type = grad.data_type();
  auto device = grad.device();
  Shape shape{batchsize, channels_, height_, width_};
  
  Tensor dx(shape, device, data_type);

  dx.device()->Exec([dx, grad, this](Context *ctx) {
    if (pool_ == PoolingConf_PoolMethod_MAX) {
      CHECK(!buf_.empty());
      Tensor mask = buf_.top();
      buf_.pop();

      Pooling_Backward_Max(grad.block(), mask.block(),
                           dx.Size(), channels_,
                           height_, width_,
                           pooled_height_, pooled_width_,
                           kernel_h_, kernel_w_,
                           pad_h_, pad_w_,
                           stride_h_, stride_w_,
                           dx.block(), ctx);
                           
    } else if (pool_ == PoolingConf_PoolMethod_AVE) {
      Pooling_Backward_Ave(grad.block(), grad.shape(0), channels_, 
                           height_, width_,
                           pooled_height_, pooled_width_,
                           kernel_h_, kernel_w_,
                           pad_h_, pad_w_,
                           stride_h_, stride_w_,
                           dx.block(), ctx);
                           
    } else
      LOG(FATAL) << "Unknown pooling method.";
    
  }, {grad.block()}, {dx.block()});

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
  auto ocl_ctx = viennacl::ocl::get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_pooling", "max_pool_forward");
  
  auto src_buf = WrapHandle(static_cast<cl_mem>(src->mutable_data()), ocl_ctx);
  auto dst_buf = WrapHandle(static_cast<cl_mem>(dst->mutable_data()), ocl_ctx);
  auto maskbuf = WrapHandle(static_cast<cl_mem>(mask->mutable_data()), ocl_ctx);

  viennacl::ocl::enqueue(kernel(num, src_buf, channels,
                                height, width, pooled_h, pooled_w,
                                kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, dst_buf, maskbuf));
}


void OpenclPooling::Pooling_Forward_Ave(const int num, Block* src, Block* dst, 
                                        const int height, const int width,
                                        const int pooled_h, const int pooled_w,
                                        const int kernel_h, const int kernel_w,
                                        const int stride_h, const int stride_w,
                                        const int pad_h, const int pad_w,
                                        const int channels, Context* ctx) {
  auto ocl_ctx = viennacl::ocl::get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_pooling", "ave_pool_forward");
  
  auto src_buf = WrapHandle(static_cast<cl_mem>(src->mutable_data()), ocl_ctx);
  auto dst_buf = WrapHandle(static_cast<cl_mem>(dst->mutable_data()), ocl_ctx);
                                   
  viennacl::ocl::enqueue(kernel(num, src_buf, channels,
                                height, width, pooled_h, pooled_w,
                                kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, dst_buf));
}


void OpenclPooling::Pooling_Forward_Sto_Train(Block* src, Block* rand,
                                              const int height, const int width,
                                              const int pooled_h, const int pooled_w,
                                              const int kernel_h, const int kernel_w,
                                              const int stride_h, const int stride_w,
                                              const int channels, 
                                              Block* dst, Context* ctx) {
  auto ocl_ctx = viennacl::ocl::get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_pooling", "sto_pool_forward_train");
  
  auto src_buf = WrapHandle(static_cast<cl_mem>(src->mutable_data()), ocl_ctx);
  auto dst_buf = WrapHandle(static_cast<cl_mem>(dst->mutable_data()), ocl_ctx);
  auto randbuf = WrapHandle(static_cast<cl_mem>(rand->mutable_data()), ocl_ctx);

  viennacl::ocl::enqueue(kernel(height * width, src_buf, channels,
                                height, width, pooled_h, pooled_w,
                                kernel_h, kernel_w, stride_h, stride_w,
                                randbuf, dst_buf));
}


void OpenclPooling::Pooling_Forward_Sto_Test(Block* src, Block* dst, 
                                             const int height, const int width,
                                             const int pooled_h, const int pooled_w,
                                             const int kernel_h, const int kernel_w,
                                             const int stride_h, const int stride_w,
                                             const int channels, Context* ctx) {
  auto ocl_ctx = viennacl::ocl::get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_pooling", "sto_pool_forward_test");
  
  auto src_buf = WrapHandle(static_cast<cl_mem>(src->mutable_data()), ocl_ctx);
  auto dst_buf = WrapHandle(static_cast<cl_mem>(dst->mutable_data()), ocl_ctx);

  viennacl::ocl::enqueue(kernel(height * width, src_buf, channels,
                                height, width, pooled_h, pooled_w,
                                kernel_h, kernel_w, stride_h, stride_w,
                                dst_buf));
}


void OpenclPooling::Pooling_Backward_Max(Block* top, Block* mask,
                                         const int num, const int channels,
                                         const int height, const int width,
                                         const int pooled_h, const int pooled_w,
                                         const int kernel_h, const int kernel_w,
                                         const int pad_h, const int pad_w,
                                         const int stride_h, const int stride_w,
                                         Block* bottom, Context* ctx) {
  auto ocl_ctx = viennacl::ocl::get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_pooling", "max_pool_backward");
  
  auto src_buf = WrapHandle(static_cast<cl_mem>(top->mutable_data()), ocl_ctx);
  auto dst_buf = WrapHandle(static_cast<cl_mem>(bottom->mutable_data()), ocl_ctx);
  auto mask_buf = WrapHandle(static_cast<cl_mem>(mask->mutable_data()), ocl_ctx);

  viennacl::ocl::enqueue(kernel(num, src_buf, mask_buf, channels,
                                height, width, pooled_h, pooled_w,
                                kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, dst_buf));
}


void OpenclPooling::Pooling_Backward_Ave(Block* bottom,
                                         const int num, const int channels, 
                                         const int height, const int width,
                                         const int pooled_h, const int pooled_w,
                                         const int kernel_h, const int kernel_w,
                                         const int pad_h, const int pad_w,
                                         const int stride_h, const int stride_w,
                                         Block* top, Context* ctx) {
  auto ocl_ctx = viennacl::ocl::get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_pooling", "ave_pool_backward");
  
  auto src_buf = WrapHandle(static_cast<cl_mem>(bottom->mutable_data()), ocl_ctx);
  auto dst_buf = WrapHandle(static_cast<cl_mem>(top->mutable_data()), ocl_ctx);
                                   
  viennacl::ocl::enqueue(kernel(num, src_buf, channels,
                                height, width, pooled_h, pooled_w,
                                kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, dst_buf));
}


void OpenclPooling::Pooling_Backward_Sto(Block* src, Block* rand, Block* dst,
                                         const int height, const int width,
                                         const int pooled_h, const int pooled_w,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int channels, Context* ctx) {
  auto ocl_ctx = viennacl::ocl::get_context(ctx->vcl_ctx_id);
  auto kernel = ocl_ctx.get_kernel("opencl_pooling", "sto_pool_backward");
  
  auto src_buf = WrapHandle(static_cast<cl_mem>(src->mutable_data()), ocl_ctx);
  auto dst_buf = WrapHandle(static_cast<cl_mem>(dst->mutable_data()), ocl_ctx);
  auto randbuf = WrapHandle(static_cast<cl_mem>(rand->mutable_data()), ocl_ctx);
                                   
  viennacl::ocl::enqueue(kernel(height * width, randbuf, src_buf, channels,
                                height, width, pooled_h, pooled_w,
                                kernel_h, kernel_w, stride_h, stride_w,
                                dst_buf));
}


} // namespace singa

#endif // USE_OPENCL
