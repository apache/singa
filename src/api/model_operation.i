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
             const bool bias);
  bool bias_term;
  size_t batchsize;
};

Tensor CpuConvForward(const Tensor &x, Tensor &W,  Tensor &b, const ConvHandle &ch);

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x, const ConvHandle &ch);

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const ConvHandle &ch);

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b, const ConvHandle &ch);


class BatchNormHandle{
  public:
    BatchNormHandle(const float momentum, const Tensor& input);

    size_t batchsize;
};


class PoolingHandle {
 public:
  PoolingHandle(const Tensor &input, const std::vector<int>& kernel_size,
                const std::vector<int>& stride, const std::vector<int>& padding,
                const bool is_max=true);

  int batchsize;

  int pooled_height;
  int pooled_width;
};


#if USE_CUDNN
class CudnnConvHandle: public ConvHandle {
 public:
  CudnnConvHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
                  const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                  const size_t in_channels, const size_t out_channels,
                  const bool bias, const size_t workspace_byte_limit = 1024 * 1024 * 1024,
                  const std::string& prefer = "fastest");
  bool bias_term;
  size_t batchsize;
};

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b, const CudnnConvHandle &cch);

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandle &cch);

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandle &cch);

Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandle &cch);


class CudnnBatchNormHandle: public BatchNormHandle{
    public:
      CudnnBatchNormHandle(const float momentum, const Tensor& input);

    size_t batchsize;
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
};

Tensor GpuPoolingForward(const CudnnPoolingHandle &cph, const Tensor &x);

Tensor GpuPoolingBackward(const CudnnPoolingHandle &cph, const Tensor &dy, const Tensor& x, const Tensor& y);

#endif  // USE_CUDNN

}  //namespace singa
