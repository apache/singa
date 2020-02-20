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
#include "singa/singa_config.h"

#ifdef USE_CBLAS

#include "../src/model/operation/convolution.h"

#include "gtest/gtest.h"
#include <chrono>
#include <iostream>

using namespace singa;
#ifdef USE_DNNL

#include "dnnl.hpp"
#include <stdio.h>

using namespace dnnl;
inline memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
}

inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
    for (size_t i = 0; i < bytes; ++i)
        dst[i] = ((uint8_t *)handle)[i];
}


TEST(MYTEST, Forward) {
  // dnnl setup
    using tag = memory::format_tag;
    using dt = memory::data_type;
    auto eng = engine(dnnl::engine::kind::cpu, 0);
    stream s(eng);

    const int batch = 64;
    const int image_h = 28;
    const int in_chan = 1;
    const int out_chan = 20;
    const int ker = 5;
    const int stride = 1;
    const int out_size = 24;
    const int group = 1;

    //const int batch = 32;
    //const int image_h = 227;
    //const int in_chan = 3;
    //const int out_chan = 96;
    //const int ker = 11;
    //const int stride = 4;
    //const int out_size = 55;

  //singa setup
  Tensor in(Shape{batch, in_chan, image_h, image_h});
  Tensor in_reo(Shape{batch, in_chan, image_h, image_h});
  Tensor out(Shape{batch, out_chan, out_size, out_size});
  Tensor weights(Shape{group, out_chan, in_chan, ker, ker});
  Tensor weights_reo(Shape{out_chan, in_chan, ker, ker});
  Tensor bias(Shape{out_chan});
  Gaussian(0.0f, 1.0f, &in);
  Gaussian(0.0f, 1.0f, &in_reo);
  Gaussian(0.0f, 1.0f, &weights);
  Gaussian(0.0f, 1.0f, &weights_reo);
  Gaussian(0.0f, 1.0f, &bias);
  Gaussian(0.0f, 1.0f, &out);
  //singa setup

    //std::vector<float> net_src(batch * in_chan * image_h * image_h);
    // std::vector<float> net_dst(batch * out_chan * 27 * 27);

    // initializing non-zero values for src
    //for (size_t i = 0; i < net_src.size(); ++i)
        //net_src[i] = sinf((float)i);


    // AlexNet: conv
    // {batch, in_chan, image_h, image_h} (x) {out_chan, in_chan, ker, ker} -> {batch, out_chan, out_size, out_size}
    // strides: {stride, stride}
    memory::dims conv_src_tz = {batch, in_chan, image_h, image_h};
    memory::dims conv_weights_tz = {group, out_chan, in_chan, ker, ker};
    memory::dims conv_bias_tz = {out_chan};
    memory::dims conv_dst_tz = {batch, out_chan, out_size, out_size};
    memory::dims conv_strides = {stride, stride};
    memory::dims conv_padding = {0, 0};


//    std::vector<float> conv_weights(product(conv_weights_tz));
//    std::vector<float> conv_bias(product(conv_bias_tz));

    // initializing non-zero values for weights and bias
//    for (size_t i = 0; i < conv_weights.size(); ++i)
//        conv_weights[i] = sinf((float)i);
//    for (size_t i = 0; i < conv_bias.size(); ++i)
//        conv_bias[i] = sinf((float)i);

    // create memory for user data
    auto conv_user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nchw}, eng, in.block()->mutable_data());
    auto conv_user_weights_memory = memory({{conv_weights_tz}, dt::f32, tag::goihw}, eng, weights.block()->mutable_data());
    auto conv_user_bias_memory = memory({{conv_bias_tz}, dt::f32, tag::x}, eng, bias.block()->mutable_data());
    //auto conv_user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nchw}, eng);
    //write_to_dnnl_memory(net_src.data(), conv_user_src_memory);
    //auto conv_user_weights_memory = memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);
    //write_to_dnnl_memory((void *)conv_weights.data(), conv_user_weights_memory);
    //auto conv_user_bias_memory = memory({{conv_bias_tz}, dt::f32, tag::x}, eng);
    //write_to_dnnl_memory(conv_bias.data(), conv_user_bias_memory);

    // create memory descriptors for convolution data w/ no specified
    // format tag(`any`)
    // tag `any` lets a primitive(convolution in this case)
    // chose the memory format preferred for best performance.
    //
    // auto conv_src_md = memory::desc({conv_src_tz}, dt::f32, tag::any);
    // auto conv_bias_md = memory::desc({conv_bias_tz}, dt::f32, tag::any);
    // auto conv_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::any);
    // auto conv_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::any);

    auto conv_src_md = memory::desc({conv_src_tz}, dt::f32, tag::nchw);
    auto conv_bias_md = memory::desc({conv_bias_tz}, dt::f32, tag::x);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::goihw);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::nchw);

    // create a convolution primitive descriptor
    auto conv_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            conv_bias_md, conv_dst_md, conv_strides, conv_padding,
            conv_padding);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, eng);

    // create reorder primitives between user input and conv src if needed
    auto conv_src_memory = conv_user_src_memory;
    auto conv_weights_memory = conv_user_weights_memory;
    // create memory for conv dst
    //auto conv_dst_memory = memory(conv_pd.dst_desc(), eng);
    auto conv_dst_memory = memory(conv_pd.dst_desc(), eng, out.block()->mutable_data());


    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // time start
    const int times = 100;
    for (int i=0; i<times; i++){
    /*
    if (conv_pd.src_desc() != conv_user_src_memory.get_desc()) {
        //conv_src_memory = memory(conv_pd.src_desc(), eng);
        conv_src_memory = memory(conv_pd.src_desc(), eng, in_reo.block()->mutable_data());
        reorder(conv_user_src_memory, conv_src_memory).execute(s,{{DNNL_ARG_FROM, conv_user_src_memory},
                {DNNL_ARG_TO, conv_src_memory}});
    }

    if (conv_pd.weights_desc() != conv_user_weights_memory.get_desc()) {
        // conv_weights_memory = memory(conv_pd.weights_desc(), eng);
        conv_weights_memory = memory(conv_pd.weights_desc(), eng, weights_reo.block()->mutable_data());
        reorder(conv_user_weights_memory, conv_weights_memory).execute(s,{{DNNL_ARG_FROM, conv_user_weights_memory},
                {DNNL_ARG_TO, conv_weights_memory}});
    }
    */
    // finally create a convolution primitive
    convolution_forward(conv_pd).execute(s,{{DNNL_ARG_SRC, conv_src_memory},
            {DNNL_ARG_WEIGHTS, conv_weights_memory},
            {DNNL_ARG_BIAS, conv_user_bias_memory},
            {DNNL_ARG_DST, conv_dst_memory}});

    s.wait();

    // time end

    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "[total]Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[mu s]" << std::endl;
    //std::cout << "[avg]Time difference = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/times << "[mu s]" << std::endl;

  printf("mytestok\n");
}


TEST(DNNLOperation_Convolution, Forward) {
  const size_t batch_size = 2, c = 1, h = 3, w = 3;
  const float x[batch_size * c * h * w] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                           7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f,
                                           4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  Tensor in(Shape{batch_size, c, h, w});
  in.CopyDataFromHostPtr(x, batch_size * c * h * w);

  const size_t num_filters = 1;
  const size_t kernel_w = 3;
  const size_t kernel_h = 3;
  const std::vector<size_t> stride = {2, 2};
  const std::vector<size_t> padding = {1, 1};
  const bool bias_flag = true;

  const float we[num_filters * kernel_w * kernel_h] = {
      1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
  Tensor weight(Shape{num_filters, num_filters, 3, 3});
  weight.CopyDataFromHostPtr(we,
                             num_filters * num_filters * kernel_w * kernel_h);

  const float b[num_filters] = {1.0f};
  Tensor bias(Shape{num_filters});
  bias.CopyDataFromHostPtr(b, num_filters);

  ConvHandle conv_handle(in, {kernel_w, kernel_h}, stride, padding, c,
                         num_filters, bias_flag);
  Tensor out1 = CpuConvForward(in, weight, bias, conv_handle);

  const float *out_ptr1 = out1.data<float>();
  // Input: 3*3; kernel: 3*3; stride: 2*2; padding: 1*1.
  EXPECT_EQ(8u, out1.Size());

  EXPECT_EQ(3.0f, out_ptr1[0]);
  EXPECT_EQ(7.0f, out_ptr1[1]);
  EXPECT_EQ(-3.0f, out_ptr1[2]);
  EXPECT_EQ(12.0f, out_ptr1[3]);
  EXPECT_EQ(3.0f, out_ptr1[4]);
  EXPECT_EQ(7.0f, out_ptr1[5]);
  EXPECT_EQ(-3.0f, out_ptr1[6]);
  EXPECT_EQ(12.0f, out_ptr1[7]);
}

TEST(DNNLOperation_Convolution, Performance) {
    const int batch = 64;
    const int image_h = 28;
    const int in_chan = 1;
    const int out_chan = 20;
    const int ker = 5;
    const int stride = 1;
    const int out_size = 24;
    const int group = 1;
  const bool bias_flag = true;

  Tensor grad(Shape{batch, out_chan, out_size, out_size});
  Tensor in(Shape{batch, in_chan, image_h, image_h});
  Tensor weight(Shape{out_chan, in_chan, ker, ker});
  Tensor bias(Shape{out_chan});
  Gaussian(0.0f, 1.0f, &grad);
  Gaussian(0.0f, 1.0f, &in);
  Gaussian(0.0f, 1.0f, &weight);
  Gaussian(0.0f, 1.0f, &bias);
  ConvHandle conv_handle(in, {ker, ker}, {stride, stride}, {0, 0}, in_chan, out_chan, bias_flag);

  const int times = 100;

  {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i=0;i<times;i++){
      Tensor out = CpuConvForward(in, weight, bias, conv_handle);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[avg]forward Time difference = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/times << "[microsec]" << std::endl;
  }

  {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i=0;i<times;i++){
  Tensor in_grad = CpuConvBackwardx(grad, weight, in, conv_handle);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[avg]backwardx Time difference = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/times << "[microsec]" << std::endl;
  }

  {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i=0;i<times;i++){
      Tensor dw = CpuConvBackwardW(grad, in, weight, conv_handle);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[avg]backwardW Time difference = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/times << "[microsec]" << std::endl;
  }

}


TEST(DNNLOperation_Convolution, Backward) {
  const size_t batch_size = 2, c = 1, h = 3, w = 3;
  const float x[batch_size * c * h * w] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                           7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f,
                                           4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  Tensor in(Shape{batch_size, c, h, w});
  in.CopyDataFromHostPtr(x, batch_size * c * h * w);

  const size_t num_filters = 1;
  const size_t kernel_w = 3;
  const size_t kernel_h = 3;
  const std::vector<size_t> stride = {2, 2};
  const std::vector<size_t> padding = {1, 1};
  const bool bias_flag = true;

  const float we[num_filters * kernel_w * kernel_h] = {
      1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
  Tensor weight(Shape{num_filters, num_filters, 3, 3});
  weight.CopyDataFromHostPtr(we,
                             num_filters * num_filters * kernel_w * kernel_h);

  const float b[num_filters] = {1.0f};
  Tensor bias(Shape{num_filters});
  bias.CopyDataFromHostPtr(b, num_filters);

  ConvHandle conv_handle(in, {kernel_w, kernel_h}, stride, padding, c,
                         num_filters, bias_flag);
  Tensor out1 = CpuConvForward(in, weight, bias, conv_handle);

  // grad
  const size_t grad_h = 2, grad_w = 2;
  const float dy[batch_size * num_filters * grad_h * grad_w] = {
      0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};
  Tensor grad(Shape{batch_size, num_filters, grad_h, grad_w});
  grad.CopyDataFromHostPtr(dy, batch_size * num_filters * grad_h * grad_w);

  Tensor in_grad = CpuConvBackwardx(grad, weight, in, conv_handle);

  const float *dx = in_grad.data<float>();
  const float *wptr = we;
  EXPECT_EQ(18u, in_grad.Size());
  EXPECT_EQ(dy[0] * wptr[4], dx[0]);
  EXPECT_EQ(dy[0] * wptr[5] + dy[1] * wptr[3], dx[1]);
  EXPECT_EQ(dy[1] * wptr[4], dx[2]);
  EXPECT_EQ(dy[0] * wptr[7] + dy[2] * wptr[1], dx[3]);
  EXPECT_EQ(dy[0] * wptr[8] + dy[1] * wptr[6] + dy[2] * wptr[2] +
                dy[3] * wptr[0],
            dx[4]);
  EXPECT_EQ(dy[1] * wptr[7] + dy[3] * wptr[1], dx[5]);
  EXPECT_EQ(dy[2] * wptr[4], dx[6]);
  EXPECT_EQ(dy[2] * wptr[5] + dy[3] * wptr[3], dx[7]);
  EXPECT_EQ(dy[3] * wptr[4], dx[8]);
  EXPECT_EQ(dy[4] * wptr[4], dx[9]);
  EXPECT_EQ(dy[4] * wptr[5] + dy[1] * wptr[3], dx[10]);
  EXPECT_EQ(dy[5] * wptr[4], dx[11]);
  EXPECT_EQ(dy[4] * wptr[7] + dy[2] * wptr[1], dx[12]);
  EXPECT_EQ(dy[4] * wptr[8] + dy[5] * wptr[6] + dy[6] * wptr[2] +
                dy[7] * wptr[0],
            dx[13]);
  EXPECT_EQ(dy[5] * wptr[7] + dy[7] * wptr[1], dx[14]);
  EXPECT_EQ(dy[6] * wptr[4], dx[15]);
  EXPECT_EQ(dy[6] * wptr[5] + dy[7] * wptr[3], dx[16]);
  EXPECT_EQ(dy[7] * wptr[4], dx[17]);


  Tensor dw = CpuConvBackwardW(grad, in, weight, conv_handle);

  Tensor db = CpuConvBackwardb(grad, bias, conv_handle);

  const float *dbptr = db.data<float>();
  EXPECT_FLOAT_EQ(dy[0] + dy[1] + dy[2] + dy[3] + dy[4] + dy[5] + dy[6] + dy[7],
                  dbptr[0]);

  const float *dwptr = dw.data<float>();
  EXPECT_EQ(9u, dw.Size());
  EXPECT_FLOAT_EQ(dy[3] * x[4] + dy[7] * x[13], dwptr[0]);
  EXPECT_FLOAT_EQ(dy[3] * x[5] + dy[7] * x[14] + dy[2] * x[3] + dy[6] * x[12],
                  dwptr[1]);
  EXPECT_FLOAT_EQ(dy[2] * x[4] + dy[6] * x[13], dwptr[2]);
  EXPECT_FLOAT_EQ(dy[1] * x[1] + dy[5] * x[10] + dy[3] * x[7] + dy[7] * x[16],
                  dwptr[3]);
  EXPECT_FLOAT_EQ(dy[0] * x[0] + dy[4] * x[9] + dy[1] * x[2] + dy[5] * x[11] +
                      dy[2] * x[6] + dy[6] * x[15] + dy[3] * x[8] +
                      dy[7] * x[17],
                  dwptr[4]);
  EXPECT_FLOAT_EQ(dy[0] * x[1] + dy[4] * x[10] + dy[2] * x[7] + dy[6] * x[16],
                  dwptr[5]);
  EXPECT_FLOAT_EQ(dy[1] * x[4] + dy[5] * x[13], dwptr[6]);
  EXPECT_FLOAT_EQ(dy[0] * x[3] + dy[4] * x[12] + dy[1] * x[5] + dy[5] * x[14],
                  dwptr[7]);
  EXPECT_FLOAT_EQ(dy[0] * x[4] + dy[4] * x[13], dwptr[8]);
}

#endif // USE_DNNL


#endif  // USE_CBLAS
