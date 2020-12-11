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
#ifndef SINGA_MODEL_OPERATION_BATCHNORM_H_
#define SINGA_MODEL_OPERATION_BATCHNORM_H_

#include <vector>

#include "singa/core/tensor.h"

#ifdef USE_CUDNN
#include <cudnn.h>

#include "../layer/cudnn_utils.h"  // check_cudnn
#endif                             // USE_CUDNN

#ifdef USE_DNNL
#include <singa/utils/dnnl_utils.h>

// combine scale and bias into weight format required by dnnl
static inline singa::Tensor get_bn_weight_from(const singa::Tensor &s,
                                               const singa::Tensor &b) {
  singa::Tensor w(singa::Shape{s.Size(), b.Size()});
  CopyDataToFrom(&w, s, s.Size(), 0, 0);
  CopyDataToFrom(&w, b, b.Size(), s.Size(), 0);
  return w;
}
#endif  // USE_DNNL

namespace singa {

class BatchNormHandle {
 public:
  BatchNormHandle(const float momentum, const Tensor &input);
  ~BatchNormHandle();

  float factor;

  size_t batchsize;
  size_t channels;
  size_t height;
  size_t width;
  bool is_2d;
  // bool train = true;
  bool use_dnnl =
      false;  // useful flag if both USE_CUDNN and USE_DNNL are enabled

#ifdef USE_DNNL
  float epsilon;
  dnnl::memory::dims x_dims;
  dnnl::memory::desc x_md;
  // as no default constructor, we need to declare it as pointer
  dnnl::batch_normalization_forward::desc *bn_fwd_training_d;
  dnnl::batch_normalization_forward::primitive_desc *bn_fwd_training_pd;
#endif  // USE_DNNL
};

#ifdef USE_DNNL
Tensor CpuBatchNormForwardInference(const BatchNormHandle &bnh, const Tensor &x,
                                    const Tensor &bnScale, const Tensor &bnBias,
                                    Tensor &running_mean, Tensor &running_var);

const std::vector<Tensor> CpuBatchNormForwardTraining(
    const BatchNormHandle &bnh, const Tensor &x, const Tensor &bnScale,
    const Tensor &bnBias, Tensor &running_mean, Tensor &running_var);

const std::vector<Tensor> CpuBatchNormBackwardx(
    const BatchNormHandle &bnh, const Tensor &y, const Tensor &dy,
    const Tensor &x, const Tensor &bnScale, const Tensor &bnBias,
    const Tensor &mean, const Tensor &var);
#endif  // USE_DNNL

#ifdef USE_CUDNN

class CudnnBatchNormHandle : public BatchNormHandle {
 public:
  CudnnBatchNormHandle(const float momentum, const Tensor &input);

  //~CudnnBatchNormHandle();

  cudnnBatchNormMode_t mode;
  cudnnTensorDescriptor_t shape_desc = nullptr;
  cudnnTensorDescriptor_t param_desc = nullptr;
};

const std::vector<Tensor> GpuBatchNormForwardTraining(
    const CudnnBatchNormHandle &cbnh, const Tensor &x, const Tensor &bnScale,
    const Tensor &bnBias, Tensor &running_mean, Tensor &running_var);

Tensor GpuBatchNormForwardInference(const CudnnBatchNormHandle &cbnh,
                                    const Tensor &x, const Tensor &bnScale,
                                    const Tensor &bnBias,
                                    const Tensor &running_mean,
                                    const Tensor &running_var);

const std::vector<Tensor> GpuBatchNormBackward(
    const CudnnBatchNormHandle &cbnh, const Tensor &dy, const Tensor &x,
    const Tensor &bnScale, const Tensor &mean, const Tensor &var);

#endif  // USE_CUDNN

}  // namespace singa

#endif  // SINGA_MODEL_OPERATION_BATCHNORM_H_
