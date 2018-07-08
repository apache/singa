#include "./batchnorm.h"

namespace singa{

BatchNormHandle::BatchNormHandle(const float momentum, const Tensor& input, const Tensor& RunningMean_, 
  const Tensor& RunningVariance_){
  factor_ = momentum;
  batchsize = input.shape()[0];
  channels_= input.shape()[2];
  if (input.nDim()== 4u){
    height_= input.shape()[3];
    width_=input.shape()[4];
    is_2d_= false;
  }else{
    size_t height_ = 1;
    size_t width_ = 1;
    bool is_2d_ = true;
  }
  runningMean_= RunningMean_;
  runningVariance_= RunningVariance_;
};

CudnnBatchNormHandle::CudnnBatchNormHandle(const float momentum, const Tensor& input, const Tensor& RunningMean_, 
  const Tensor& RunningVariance_):BatchNormHandle(momentum, input, RunningMean_, RunningVariance_){
  if (is_2d_)
      mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
  else
      mode_ = CUDNN_BATCHNORM_SPATIAL;
  auto dtype = input.data_type();
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&shape_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&param_desc_));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(shape_desc_, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize,
                                         channels_, height_, width_));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(param_desc_, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), 1, channels_,
                                         1, 1));
  };

Tensor GpuBatchNormForwardTraining(const Tensor& x, const Tensor& bnScale_, const Tensor& bnBias_, 
  std::vector<Tensor>& cache, const CudnnBatchNormHandle &cbnh) {
  
  auto shape = x.shape();
  Tensor output;
  Tensor input;  //for unification of 2d and 4d cases.
  if (cbnh.is_2d_)
    input = Reshape(x, Shape{shape.at(0), shape.at(1), 1, 1});
  else
    input = x;
  output.ResetLike(x);

  Tensor resultSaveMean_;
  Tensor resultSaveVariance_;

  resultSaveMean_.Reshape(Shape{cbnh.channels_});
  resultSaveVariance_.Reshape(Shape{cbnh.channels_});

  cache.push_back(resultSaveMean_);
  cache.push_back(resultSaveVariance_);
  cache.push_back(bnScale_);
  //cache={x, mean, var, scale}

    output.device()->Exec(
        [&output, &input, &bnScale_, &bnBias_, &cache, &cbnh](Context* ctx) {
          Block* inBlock = input.block(), * outBlock = output.block(),
                 * saveMeanBlock = cache[1].block(),
                 * saveVarBlock = cache[2].block(),
                 * runningMeanBlock = cbnh.runningMean_.block(),
                 * runningVarBlock = cbnh.runningVariance_.block(),
                 * bnScaleBlock = bnScale_.block(),
                 * bnBiasBlock = bnBias_.block();
          const float alpha = 1.0f, beta = 0.0f;
          double epsilon = CUDNN_BN_MIN_EPSILON;
          CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
              ctx->cudnn_handle, cbnh.mode_, &alpha, &beta, cbnh.shape_desc_,
              inBlock->data(), cbnh.shape_desc_, outBlock->mutable_data(),
              cbnh.param_desc_, bnScaleBlock->data(), bnBiasBlock->data(), cbnh.factor_,
              runningMeanBlock->mutable_data(), runningVarBlock->mutable_data(),
              epsilon, saveMeanBlock->mutable_data(),
              saveVarBlock->mutable_data()));
        },
        {input.block(), bnScale_.block(), bnBias_.block()},
        {output.block(), cbnh.runningMean_.block(), cbnh.runningVariance_.block(),
         cache[1].block(), cache[2].block()}); 
  if (cbnh.is_2d_) output.Reshape(Shape{shape.at(0), shape.at(1)});
  return output;
};

Tensor GpuBatchNormForwardInference(const Tensor& x, const Tensor& bnScale_, const Tensor& bnBias_, 
   const CudnnBatchNormHandle &cbnh) {
  auto shape = x.shape();
  Tensor output;
  Tensor input;  //for unification of 2d and 4d cases.
  if (cbnh.is_2d_)
    input = Reshape(x, Shape{shape.at(0), shape.at(1), 1, 1});
  else
    input = x;
  output.ResetLike(x);
    output.device()->Exec(
        [&output, &input, &bnScale_, &bnBias_, &cbnh](Context* ctx) {
          Block* inBlock = input.block(), * outBlock = output.block(),
                 * runningMeanBlock = cbnh.runningMean_.block(),
                 * runningVarBlock = cbnh.runningVariance_.block(),
                 * bnScaleBlock = bnScale_.block(),
                 * bnBiasBlock = bnBias_.block();
          const float alpha = 1.0f, beta = 0.0f;
          double epsilon = CUDNN_BN_MIN_EPSILON;
          CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
              ctx->cudnn_handle, cbnh.mode_, &alpha, &beta, cbnh.shape_desc_,
              inBlock->data(), cbnh.shape_desc_, outBlock->mutable_data(),
              cbnh.param_desc_, bnScaleBlock->data(), bnBiasBlock->data(),
              runningMeanBlock->data(), runningVarBlock->data(), epsilon));
        },
        {input.block(), bnScale_.block(), bnBias_.block(), cbnh.runningMean_.block(),
         cbnh.runningVariance_.block()},
        {output.block()});
  if (cbnh.is_2d_) output.Reshape(Shape{shape.at(0), shape.at(1)});
  return output;
};


std::vector<Tensor> GpuBatchNormBackward(const Tensor& dy, const std::vector<Tensor>& cache, const CudnnBatchNormHandle &cbnh){

  vector<Tensor> out_grads;
  Tensor dx;
  dx.ResetLike(dy);

  Tensor dbnScale_;
  dbnScale_.ResetLike(cache[3]);

  Tensor dbnBias_;
  dbnBias_.ResetLike(cache[3]);
  //dbnBias_.ResetLike(bnBias_);

  dx.device()->Exec(
      [&dx, &dbnScale_, &dbnBias_, &dy, &cache, &cbnh](Context* ctx) {
        Block* dyblock = dy.block(), * dxblock = dx.block(),
               * xblock = cache[0].block(), * bnScaleBlock = cache[3].block(),
               * dbnScaleBlock = dbnScale_.block(),
               * dbnBiasBlock = dbnBias_.block(),
               * saveMeanBlock = cache[1].block(),
               * saveVarBlock = cache[2].block();
        const float alpha = 1.0f, beta = .0f;
        double epsilon = CUDNN_BN_MIN_EPSILON;
        CUDNN_CHECK(cudnnBatchNormalizationBackward(
            ctx->cudnn_handle, cbnh.mode_, &alpha, &beta, &alpha, &beta,
            cbnh.shape_desc_, xblock->data(), cbnh.shape_desc_, dyblock->data(),
            cbnh.shape_desc_, dxblock->mutable_data(), cbnh.param_desc_,
            bnScaleBlock->data(), dbnScaleBlock->mutable_data(),
            dbnBiasBlock->mutable_data(), epsilon, saveMeanBlock->data(),
            saveVarBlock->data()));
      },
      {cache[0].block(), dy.block(), cache[3].block(), cache[1].block(),
       cache[2].block()},
      {dx.block(), dbnScale_.block(), dbnBias_.block()});
  
  if (cbnh.is_2d_) dx.Reshape(Shape{dx.shape().at(0), dx.shape().at(1)});
  out_grads.push_back(dx);
  out_grads.push_back(dbnScale_);
  out_grads.push_back(dbnBias_);
return out_grads;
};

}
