#ifndef SINGA_MODEL_OPERATION_CONVOLUTION_H_
#define SINGA_MODEL_OPERATION_CONVOLUTION_H_

#include <string>
#include <vector>
#include "singa/core/tensor.h"
#include "singa/utils/logging.h"
#include "singa/singa_config.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#include "../layer/cudnn_utils.h"
#endif // USE_CUDNN


namespace singa {

class ConvHandle {

 public:
  ConvHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
             const std::vector<size_t>& stride, const std::vector<size_t>& padding,
             const size_t in_channels, const size_t out_channels,
             const bool bias);
 
  size_t kernel_w;
  size_t pad_w;
  size_t stride_w;
  size_t kernel_h;
  size_t pad_h;
  size_t stride_h;

  size_t channels;
  size_t num_filters;

  bool bias_term;

  size_t height;
  size_t width;
  size_t conv_height;
  size_t conv_width;
  size_t batchsize;

  size_t col_height;
  size_t col_width;
  size_t imagesize;
};


Tensor CpuConvForward(const Tensor &x, Tensor &W,  Tensor &b, const ConvHandle &ch);

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x, const ConvHandle &ch);

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const ConvHandle &ch);

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b, const ConvHandle &ch);



#ifdef USE_CUDNN
class CudnnConvHandle: public ConvHandle {
 public:
  CudnnConvHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
                  const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                  const size_t in_channels, const size_t out_channels,
                  const bool bias, const size_t workspace_byte_limit = 1024 * 1024 * 1024,
                  const std::string& prefer = "fastest");
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
};

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b, const CudnnConvHandle &cch);

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandle &cch);

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandle &cch);

Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandle &cch);
#endif  // USE_CUDNN

}  // namespace singa
#endif  // SINGA_MODEL_OPERATION_CONVOLUTION_H_
