#ifndef SINGA_MODEL_OPERATION_POOLING_H_
#define SINGA_MODEL_OPERATION_POOLING_H_

#include <string>
#include "singa/core/tensor.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#include "../layer/cudnn_utils.h"
#endif

namespace singa {

class PoolingHandle {
public:
  PoolingHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
                const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                const bool ceil_mode = false, const std::string pooling_method = "MAX");

  size_t kernel_w;
  size_t pad_w;
  size_t stride_w;
  size_t kernel_h;
  size_t pad_h;
  size_t stride_h;

  size_t batchsize;
  size_t channels;
  size_t height;
  size_t width;

  size_t pooled_height;
  size_t pooled_width;

  std::string method;
};

#ifdef USE_CUDNN
class CudnnPoolingHandle : public PoolingHandle {
public:
  CudnnPoolingHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
                     const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                     const bool ceil_mode = false, const std::string pooling_method = "MAX",
                     const bool NaN_prop = false);
  ~CudnnPoolingHandle();

  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnPoolingDescriptor_t pool_desc = nullptr;
  cudnnNanPropagation_t nan_prop;

};

Tensor GpuPoolingForward(const Tensor &x, const CudnnPoolingHandle &cph);

Tensor GpuPoolingBackward(const Tensor &dy, const Tensor& x, const Tensor& y,
                          const CudnnPoolingHandle &cph);

#endif  //USE_CUDNN

}  // namespace singa

#endif  // SINGA_MODEL_OPERATION_POOLING_H_