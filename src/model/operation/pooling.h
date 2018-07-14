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
  PoolingHandle(const Tensor &input, const std::vector<int>& kernel_size,
                const std::vector<int>& stride, const std::vector<int>& padding,
                const bool is_max = true);

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
};

#ifdef USE_CUDNN
class CudnnPoolingHandle : public PoolingHandle {
 public:
  CudnnPoolingHandle(const Tensor &input, const std::vector<int>& kernel_size,
                     const std::vector<int>& stride, const std::vector<int>& padding,
                     const bool is_max = true);
  ~CudnnPoolingHandle();

  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnPoolingDescriptor_t pool_desc = nullptr;
  cudnnNanPropagation_t nan_prop = CUDNN_PROPAGATE_NAN;

};

Tensor GpuPoolingForward(const CudnnPoolingHandle &cph, const Tensor &x);

Tensor GpuPoolingBackward(const CudnnPoolingHandle &cph, const Tensor &dy,
                          const Tensor& x, const Tensor& y);

#endif  //USE_CUDNN

}  // namespace singa

#endif  // SINGA_MODEL_OPERATION_POOLING_H_
