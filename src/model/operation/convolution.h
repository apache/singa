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
#ifndef SINGA_MODEL_OPERATION_CONVOLUTION_H_
#define SINGA_MODEL_OPERATION_CONVOLUTION_H_

#include <string>
#include <vector>

#include "singa/core/tensor.h"
#include "singa/singa_config.h"
#include "singa/utils/logging.h"

#ifdef USE_CUDNN
#include <cudnn.h>

#include "../layer/cudnn_utils.h"
#endif  // USE_CUDNN

#ifdef USE_DNNL
#include <singa/utils/dnnl_utils.h>
#endif  // USE_DNNL

namespace singa {

class ConvHandle {
 public:
  ConvHandle(const Tensor &input, const std::vector<size_t> &kernel_size,
             const std::vector<size_t> &stride,
             const std::vector<size_t> &padding, const size_t in_channels,
             const size_t out_channels, const bool bias,
             const size_t groups = 1);

  ~ConvHandle();

  size_t kernel_w;
  size_t pad_w;
  size_t stride_w;
  size_t kernel_h;
  size_t pad_h;
  size_t stride_h;

  size_t channels;
  size_t num_filters;
  size_t group;

  bool bias_term;

  size_t height;
  size_t width;
  size_t conv_height;
  size_t conv_width;
  size_t batchsize;

  size_t col_height;
  size_t col_width;
  size_t imagesize;

  bool use_dnnl =
      false;  // useful flag if both USE_CUDNN and USE_DNNL are enabled

#ifdef USE_DNNL
  dnnl::memory::data_type dtype;
  dnnl::memory::dims b_dims;
  dnnl::memory::dims s_dims;
  dnnl::memory::dims p_dims;
  dnnl::memory::dims x_dims;
  dnnl::memory::dims o_dims;
  dnnl::memory::dims w_dims;

  Tensor *db;
#endif  // USE_DNNL
};

Tensor CpuConvForward(const Tensor &x, Tensor &W, Tensor &b,
                      const ConvHandle &ch);

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x,
                        const ConvHandle &ch);

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W,
                        const ConvHandle &ch);

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b,
                        const ConvHandle &ch);

#ifdef USE_CUDNN
class CudnnConvHandle : public ConvHandle {
 public:
  CudnnConvHandle(const Tensor &input, const std::vector<size_t> &kernel_size,
                  const std::vector<size_t> &stride,
                  const std::vector<size_t> &padding, const size_t in_channels,
                  const size_t out_channels, const bool bias,
                  const size_t groups = 1,
                  const size_t workspace_byte_limit = 1024 * 1024 * 1024,
                  const std::string &prefer = "fastest");
  ~CudnnConvHandle();
  // TODO(wangwei) add the destructor

  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnTensorDescriptor_t bias_desc = nullptr;
  cudnnFilterDescriptor_t filter_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  cudnnConvolutionFwdAlgo_t fp_alg;
  cudnnConvolutionBwdFilterAlgo_t bp_filter_alg;
  cudnnConvolutionBwdDataAlgo_t bp_data_alg;

  size_t workspace_count;
  Tensor workspace;
  size_t channels_per_filter;
};

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b,
                      const CudnnConvHandle &cch);

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x,
                        const CudnnConvHandle &cch);

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W,
                        const CudnnConvHandle &cch);

Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b,
                        const CudnnConvHandle &cch);
#endif  // USE_CUDNN

}  // namespace singa
#endif  // SINGA_MODEL_OPERATION_CONVOLUTION_H_
