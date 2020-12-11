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
#ifndef SINGA_MODEL_OPERATION_POOLING_H_
#define SINGA_MODEL_OPERATION_POOLING_H_

#include <string>

#include "singa/core/tensor.h"

#ifdef USE_CUDNN
#include <cudnn.h>

#include "../layer/cudnn_utils.h"
#endif

#ifdef USE_DNNL
#include <singa/utils/dnnl_utils.h>
#endif  // USE_DNNL

namespace singa {

class PoolingHandle {
 public:
  PoolingHandle(const Tensor &input, const std::vector<int> &kernel_size,
                const std::vector<int> &stride, const std::vector<int> &padding,
                const bool is_max = true);
  ~PoolingHandle();

  int kernel_w;
  int pad_w;
  int stride_w;
  int kernel_h;
  int pad_h;
  int stride_h;

  int batchsize;
  int channels;
  int height;
  int width;

  int pooled_height;
  int pooled_width;

  bool is_max_pooling;

#ifdef USE_DNNL
  dnnl::memory::desc x_md;
  dnnl::memory::desc y_md;
  dnnl::memory ws_mem;
  dnnl::pooling_forward::primitive_desc pool_fwd_pd;
  dnnl::pooling_backward::primitive_desc pool_bwd_pd;
#endif  // USE_DNNL
};

#ifdef USE_DNNL
Tensor CpuPoolingForward(const PoolingHandle &ph, const Tensor &x);
Tensor CpuPoolingBackward(const PoolingHandle &ph, const Tensor &dy,
                          const Tensor &x, const Tensor &y);
#endif  // USE_DNNL

#ifdef USE_CUDNN
class CudnnPoolingHandle : public PoolingHandle {
 public:
  CudnnPoolingHandle(const Tensor &input, const std::vector<int> &kernel_size,
                     const std::vector<int> &stride,
                     const std::vector<int> &padding, const bool is_max = true);
  ~CudnnPoolingHandle();

  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnPoolingDescriptor_t pool_desc = nullptr;
  cudnnNanPropagation_t nan_prop = CUDNN_PROPAGATE_NAN;
};

Tensor GpuPoolingForward(const CudnnPoolingHandle &cph, const Tensor &x);

Tensor GpuPoolingBackward(const CudnnPoolingHandle &cph, const Tensor &dy,
                          const Tensor &x, const Tensor &y);

#endif  // USE_CUDNN

}  // namespace singa

#endif  // SINGA_MODEL_OPERATION_POOLING_H_
