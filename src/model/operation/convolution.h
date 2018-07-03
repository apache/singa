#ifndef SINGA_MODEL_OPERATION_CONVOLUTION_H_
#define SINGA_MODEL_OPERATION_CONVOLUTION_H_

#include <string>
#include <vector>
#include "singa/utils/logging.h"

#ifdef USE_CUDNN
#include <cudnn.h>
// #include "../layer/cudnn_convolution.h"
// #include "../layer/cudnn_utils.h"
#endif // USE_CUDNN


namespace singa {

class ConvHandle {

 public:
  ConvHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
             const std::vector<size_t>& stride, const std::vector<size_t>& padding,
             const size_t in_channels, const size_t out_channels,
             const bool bias);
 protected:
  size_t kernel_w_;
  size_t pad_w_;
  size_t stride_w_;
  size_t kernel_h_;
  size_t pad_h_;
  size_t stride_h_;

  size_t channels_;
  size_t num_filters_;

  bool bias_term_;

  size_t height_;
  size_t width_;
  size_t conv_height_;
  size_t conv_width_;
  size_t batchsize;

  size_t col_height_;
  size_t col_width_;
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
                  const bool bias, const size_t workspace_byte_limit_ = 1024 * 1024 * 1024,
                  const std::string& prefer_ = "fastest");
  ~CudnnConvHandle();
  // TODO(wangwei) add the destructor
 protected:
  cudnnTensorDescriptor_t x_desc_ ;
  cudnnTensorDescriptor_t y_desc_ ;
  cudnnTensorDescriptor_t bias_desc_ ;
  cudnnFilterDescriptor_t filter_desc_ ;
  cudnnConvolutionDescriptor_t conv_desc_ ;
  cudnnConvolutionFwdAlgo_t fp_alg_;
  cudnnConvolutionBwdFilterAlgo_t bp_filter_alg_;
  cudnnConvolutionBwdDataAlgo_t bp_data_alg_;

  size_t workspace_count_;
  Tensor workspace_;
};

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b, const CudnnConvHandle &cch);

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandle &cch);

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandle &cch);

Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandle &cch);
#endif  // USE_CUDNN

}  // namespace singa
#endif  // SINGA_MODEL_OPERATION_CONVOLUTION_H_
