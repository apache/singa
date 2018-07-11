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
  BatchNormHandle(const float momentum, const Tensor& input, const Tensor& RunningMean, const Tensor& RunningVariance);

  float factor;

  size_t batchsize;
  size_t channels;
  size_t height;
  size_t width;

  Tensor runningMean;
  Tensor runningVariance;

  bool is_2d;
  //bool train = true;
};

//Tensor CpuBatchNormForwardTraining();

//Tensor CpuBatchNormForwardInference();

//Tensor CpuBatchNormBackwardx();


#ifdef USE_CUDNN

class CudnnBatchNormHandle: public BatchNormHandle {
public:
  CudnnBatchNormHandle(const float momentum, const Tensor& input, const Tensor& RunningMean, const Tensor& RunningVariance);

  //~CudnnBatchNormHandle();

  cudnnBatchNormMode_t mode;
  cudnnTensorDescriptor_t shape_desc = nullptr;
  cudnnTensorDescriptor_t param_desc = nullptr;
};

Tensor GpuBatchNormForwardTraining(const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
                                   std::vector<Tensor>& cache, const CudnnBatchNormHandle &cbnh);

Tensor GpuBatchNormForwardInference(const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
                                    const CudnnBatchNormHandle &cbnh);

std::vector<Tensor> GpuBatchNormBackward(const Tensor& dy, const std::vector<Tensor>& cache, const CudnnBatchNormHandle &cbnh);

#endif  // USE_CUDNN

}  // namespace singa
