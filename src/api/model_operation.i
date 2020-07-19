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

%module model_operation

%include "config.i"
%include "std_vector.i"
%include "std_string.i"
%{
#include "../src/model/operation/convolution.h"
#include "../src/model/operation/batchnorm.h"
#include "../src/model/operation/pooling.h"

%}

namespace singa {

class ConvHandle {
 public:
  ConvHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
             const std::vector<size_t>& stride, const std::vector<size_t>& padding,
             const size_t in_channels, const size_t out_channels,
             const bool bias, const size_t groups);
  bool bias_term;
  size_t batchsize;
  size_t pad_w;
  size_t pad_h;
  size_t stride_h;
  size_t stride_w;
  size_t kernel_h;
  size_t kernel_w;
  size_t channels;
  size_t num_filters;
  size_t group;
};

Tensor CpuConvForward(const Tensor &x, Tensor &W,  Tensor &b, const ConvHandle &ch);

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x, const ConvHandle &ch);

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const ConvHandle &ch);

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b, const ConvHandle &ch);


class BatchNormHandle{
  public:
    BatchNormHandle(const float momentum, const Tensor& input);

    size_t batchsize;
    float factor;
};

#if USE_DNNL
Tensor CpuBatchNormForwardInference(const BatchNormHandle &bnh,
                                    const Tensor &x,
                                    const Tensor &bnScale,
                                    const Tensor &bnBias,
                                    Tensor &running_mean,
                                    Tensor &running_var);

const std::vector<Tensor> CpuBatchNormForwardTraining(const BatchNormHandle &bnh,
                                                      const Tensor &x,
                                                      const Tensor &bnScale,
                                                      const Tensor &bnBias,
                                                      Tensor &running_mean,
                                                      Tensor &running_var);

const std::vector<Tensor> CpuBatchNormBackwardx(const BatchNormHandle &bnh,
                                                const Tensor &y, const Tensor &dy,
                                                const Tensor &x,
                                                const Tensor &bnScale, const Tensor &bnBias,
                                                const Tensor &mean, const Tensor &var);
#endif //USE_DNNL


class PoolingHandle {
 public:
  PoolingHandle(const Tensor &input, const std::vector<int>& kernel_size,
                const std::vector<int>& stride, const std::vector<int>& padding,
                const bool is_max=true);

  int batchsize;
  int stride_h;
  int stride_w;
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;
  int pooled_height;
  int pooled_width;
  bool is_max_pooling;
};

#if USE_DNNL
Tensor CpuPoolingForward(const PoolingHandle &ph, const Tensor &x);
Tensor CpuPoolingBackward(const PoolingHandle &ph, const Tensor &dy,
                              const Tensor& x, const Tensor& y);
#endif //USE_DNNL


#if USE_CUDNN
class CudnnConvHandle: public ConvHandle {
 public:
  CudnnConvHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
                  const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                  const size_t in_channels, const size_t out_channels,
                  const bool bias, const size_t groups = 1, const size_t workspace_byte_limit = 1024 * 1024 * 1024,
                  const std::string& prefer = "fastest");
  bool bias_term;
  size_t batchsize;
  size_t pad_w;
  size_t pad_h;
  size_t stride_h;
  size_t stride_w;
  size_t kernel_h;
  size_t kernel_w;
  size_t channels;
  size_t num_filters;
  size_t group;
};

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b, const CudnnConvHandle &cch);

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandle &cch);

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandle &cch);

Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandle &cch);


class CudnnBatchNormHandle: public BatchNormHandle{
    public:
      CudnnBatchNormHandle(const float momentum, const Tensor& input);
    size_t channels;
    size_t batchsize;
    float factor;
};

const std::vector<Tensor> GpuBatchNormForwardTraining(const CudnnBatchNormHandle &cbnh,
  const Tensor& x, const Tensor& bnScale, const Tensor& bnBias, Tensor& running_mean, Tensor& running_var);

Tensor GpuBatchNormForwardInference(const CudnnBatchNormHandle &cbnh, const Tensor& x,
  const Tensor& bnScale, const Tensor& bnBias,  const Tensor& running_mean, const Tensor& running_var);

const std::vector<Tensor> GpuBatchNormBackward(const CudnnBatchNormHandle &cbnh,
  const Tensor& dy, const Tensor& x, const Tensor& bnScale, const Tensor& mean, const Tensor& var);


class CudnnPoolingHandle : public PoolingHandle {
 public:
  CudnnPoolingHandle(const Tensor &input, const std::vector<int>& kernel_size,
                     const std::vector<int>& stride, const std::vector<int>& padding,
                     const bool is_max=true);

  int batchsize;

  int pooled_height;
  int pooled_width;
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;

  int stride_h;
  int stride_w;

};

Tensor GpuPoolingForward(const CudnnPoolingHandle &cph, const Tensor &x);

Tensor GpuPoolingBackward(const CudnnPoolingHandle &cph, const Tensor &dy, const Tensor& x, const Tensor& y);

#endif  // USE_CUDNN

}  //namespace singa
