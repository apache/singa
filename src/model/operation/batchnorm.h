//#ifndef SINGA_MODEL_OPERATION_BATCHNORM_H_
//#define SINGA_MODEL_OPERATION_BATCHNORM_H_

#include <vector>
#include "singa/core/tensor.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#include "../layer/cudnn_utils.h" // check_cudnn
#endif // USE_CUDNN 

namespace singa {

class BatchNormHandle {
 public:
  BatchNormHandle(const float momentum, const Tensor& input);

  float factor;

  size_t batchsize;
  size_t channels;
  size_t height;
  size_t width;
  bool is_2d;
  //bool train = true;
};

//Tensor CpuBatchNormForwardTraining();

//Tensor CpuBatchNormForwardInference();

//Tensor CpuBatchNormBackwardx();


#ifdef USE_CUDNN

class CudnnBatchNormHandle: public BatchNormHandle {
 public:
  CudnnBatchNormHandle(const float momentum, const Tensor& input);

  //~CudnnBatchNormHandle();

  cudnnBatchNormMode_t mode;
  cudnnTensorDescriptor_t shape_desc = nullptr;
  cudnnTensorDescriptor_t param_desc = nullptr;
};

const std::vector<Tensor> GpuBatchNormForwardTraining(const CudnnBatchNormHandle
    &cbnh, const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
    Tensor& running_mean, Tensor& running_var);

Tensor GpuBatchNormForwardInference(const CudnnBatchNormHandle &cbnh,
                                    const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
                                    const Tensor& running_mean, const Tensor& running_var);

const std::vector<Tensor> GpuBatchNormBackward(const CudnnBatchNormHandle &cbnh,
    const Tensor& dy, const Tensor& x, const Tensor& bnScale, const Tensor& mean,
    const Tensor& var);

#endif  // USE_CUDNN

}  // namespace singa
