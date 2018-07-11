#include "./batchnorm.h"

namespace singa {

BatchNormHandle::BatchNormHandle(const float momentum, const Tensor& input, const Tensor& RunningMean,
                                 const Tensor& RunningVariance) {
  factor = momentum;
  batchsize = input.shape(0);
  channels = input.shape(1);
  if (input.nDim() == 4u) {
    height = input.shape(2);
    width = input.shape(3);
    is_2d = false;
  } else if (input.nDim() == 2u) {
    height = 1;
    width = 1;
    is_2d = true;
  } else {
    LOG(FATAL) << "The dimension of input should either be 4D or 2D.";
  }
  runningMean = RunningMean;
  runningVariance = RunningVariance;
};

CudnnBatchNormHandle::CudnnBatchNormHandle(const float momentum, const Tensor& input, const Tensor& RunningMean,
    const Tensor& RunningVariance): BatchNormHandle(momentum, input, RunningMean, RunningVariance) {
  if (is_2d)
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  else
    mode = CUDNN_BATCHNORM_SPATIAL;
  DataType dtype = input.data_type();
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&shape_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&param_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(shape_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype),
                                         batchsize,
                                         channels, height, width));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(param_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), 1, channels,
                                         1, 1));
};

Tensor GpuBatchNormForwardTraining(const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
                                   std::vector<Tensor>& cache, const CudnnBatchNormHandle &cbnh) {

  Shape shape = x.shape();
  Tensor output;
  Tensor input;  //for unification of 2d and 4d cases.
  if (cbnh.is_2d)
    input = Reshape(x, Shape{shape.at(0), shape.at(1), 1, 1});
  else
    input = x;
  output.ResetLike(x);

  Tensor resultSaveMean;
  Tensor resultSaveVariance;

  resultSaveMean.Reshape(Shape{cbnh.channels});
  resultSaveVariance.Reshape(Shape{cbnh.channels});

  cache.push_back(resultSaveMean);
  cache.push_back(resultSaveVariance);
  cache.push_back(bnScale);
  //cache={x, mean, var, scale}

  output.device()->Exec(
  [&output, &input, &bnScale, &bnBias, &cache, &cbnh](Context * ctx) {
    Block* inBlock = input.block(), * outBlock = output.block(),
           * saveMeanBlock = cache[1].block(),
             * saveVarBlock = cache[2].block(),
               * runningMeanBlock = cbnh.runningMean.block(),
                 * runningVarBlock = cbnh.runningVariance.block(),
                   * bnScaleBlock = bnScale.block(),
                     * bnBiasBlock = bnBias.block();
    const float alpha = 1.0f, beta = 0.0f;
    double epsilon = CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
                  ctx->cudnn_handle, cbnh.mode, &alpha, &beta, cbnh.shape_desc,
                  inBlock->data(), cbnh.shape_desc, outBlock->mutable_data(),
                  cbnh.param_desc, bnScaleBlock->data(), bnBiasBlock->data(), cbnh.factor,
                  runningMeanBlock->mutable_data(), runningVarBlock->mutable_data(),
                  epsilon, saveMeanBlock->mutable_data(),
                  saveVarBlock->mutable_data()));
  },
  {input.block(), bnScale.block(), bnBias.block()},
  { output.block(), cbnh.runningMean.block(), cbnh.runningVariance.block(),
    cache[1].block(), cache[2].block()
  });
  if (cbnh.is_2d) output.Reshape(Shape{shape.at(0), shape.at(1)});
  return output;
};

Tensor GpuBatchNormForwardInference(const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
                                    const CudnnBatchNormHandle &cbnh) {
  Shape shape = x.shape();
  Tensor output;
  Tensor input;  //for unification of 2d and 4d cases.
  if (cbnh.is_2d)
    input = Reshape(x, Shape{shape.at(0), shape.at(1), 1, 1});
  else
    input = x;
  output.ResetLike(x);
  output.device()->Exec(
  [&output, &input, &bnScale, &bnBias, &cbnh](Context * ctx) {
    Block* inBlock = input.block(), * outBlock = output.block(),
           * runningMeanBlock = cbnh.runningMean.block(),
             * runningVarBlock = cbnh.runningVariance.block(),
               * bnScaleBlock = bnScale.block(),
                 * bnBiasBlock = bnBias.block();
    const float alpha = 1.0f, beta = 0.0f;
    double epsilon = CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
                  ctx->cudnn_handle, cbnh.mode, &alpha, &beta, cbnh.shape_desc,
                  inBlock->data(), cbnh.shape_desc, outBlock->mutable_data(),
                  cbnh.param_desc, bnScaleBlock->data(), bnBiasBlock->data(),
                  runningMeanBlock->data(), runningVarBlock->data(), epsilon));
  },
  { input.block(), bnScale.block(), bnBias.block(), cbnh.runningMean.block(),
    cbnh.runningVariance.block()
  },
  {output.block()});
  if (cbnh.is_2d) output.Reshape(Shape{shape.at(0), shape.at(1)});
  return output;
};


std::vector<Tensor> GpuBatchNormBackward(const Tensor& dy, const std::vector<Tensor>& cache, const CudnnBatchNormHandle &cbnh) {

  vector<Tensor> out_grads;
  Tensor dx;
  dx.ResetLike(dy);

  Tensor dbnScale;
  dbnScale.ResetLike(cache[3]);

  Tensor dbnBias;
  dbnBias.ResetLike(cache[3]);
  //dbnBias.ResetLike(bnBias);

  dx.device()->Exec(
  [&dx, &dbnScale, &dbnBias, &dy, &cache, &cbnh](Context * ctx) {
    Block* dyblock = dy.block(), * dxblock = dx.block(),
           * xblock = cache[0].block(), * bnScaleBlock = cache[3].block(),
             * dbnScaleBlock = dbnScale.block(),
               * dbnBiasBlock = dbnBias.block(),
                 * saveMeanBlock = cache[1].block(),
                   * saveVarBlock = cache[2].block();
    const float alpha = 1.0f, beta = .0f;
    double epsilon = CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationBackward(
                  ctx->cudnn_handle, cbnh.mode, &alpha, &beta, &alpha, &beta,
                  cbnh.shape_desc, xblock->data(), cbnh.shape_desc, dyblock->data(),
                  cbnh.shape_desc, dxblock->mutable_data(), cbnh.param_desc,
                  bnScaleBlock->data(), dbnScaleBlock->mutable_data(),
                  dbnBiasBlock->mutable_data(), epsilon, saveMeanBlock->data(),
                  saveVarBlock->data()));
  },
  { cache[0].block(), dy.block(), cache[3].block(), cache[1].block(),
    cache[2].block()
  },
  {dx.block(), dbnScale.block(), dbnBias.block()});

  if (cbnh.is_2d) dx.Reshape(Shape{dx.shape().at(0), dx.shape().at(1)});
  out_grads.push_back(dx);
  out_grads.push_back(dbnScale);
  out_grads.push_back(dbnBias);
  return out_grads;
};

}
