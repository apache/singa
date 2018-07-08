//#ifndef SINGA_MODEL_OPERATION_BATCHNORM_H_
//#define SINGA_MODEL_OPERATION_BATCHNORM_H_

#include <vector>
#include "singa/core/tensor.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#include "../layer/cudnn_utils.h" // check_cudnn
#endif // USE_CUDNN 

namespace singa{

class BatchNormHandle{
  public:
  	BatchNormHandle(const float momentum, const Tensor& input, const Tensor& RunningMean_, const Tensor& RunningVariance_);

  	float factor_;
  	size_t channels_;
  	size_t batchsize;

  	Tensor runningMean_;
  	Tensor runningVariance_;

  	bool is_2d_ ;
  	//bool train = true;

  	size_t height_;
  	size_t width_;
};

//Tensor CpuBatchNormForwardTraining();

//Tensor CpuBatchNormForwardInference();

//Tensor CpuBatchNormBackwardx();


#ifdef USE_CUDNN

class CudnnBatchNormHandle: public BatchNormHandle{
    public:
      CudnnBatchNormHandle(const float momentum, const Tensor& input, const Tensor& RunningMean_, const Tensor& RunningVariance_);

      //~CudnnBatchNormHandle();

      cudnnBatchNormMode_t mode_;
      cudnnTensorDescriptor_t shape_desc_ = nullptr;
      cudnnTensorDescriptor_t param_desc_ = nullptr;
};

Tensor GpuBatchNormForwardTraining(const Tensor& x, const Tensor& bnScale_, const Tensor& bnBias_, 
  std::vector<Tensor>& cache, const CudnnBatchNormHandle &cbnh);

Tensor GpuBatchNormForwardInference(const Tensor& x, const Tensor& bnScale_, const Tensor& bnBias_, 
	const CudnnBatchNormHandle &cbnh);

std::vector<Tensor> GpuBatchNormBackward(const Tensor& dy, const std::vector<Tensor>& cache, const CudnnBatchNormHandle &cbnh);

#endif  // USE_CUDNN

}  // namespace singa
