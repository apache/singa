#include "./batchnorm.h"

namespace singa {

BatchNormHandle::BatchNormHandle(const float momentum, const Tensor& input) {
  factor = momentum;
  batchsize = input.shape(0);
  channels = input.shape(1);
  if (input.nDim() == 4u) {
    height = input.shape().at(2);
    width = input.shape().at(3);
    is_2d = false;
  } else if (input.nDim() == 2u) {
    height = 1;
    width = 1;
    is_2d = true;
  } else {
    LOG(FATAL) << "The dimension of input should either be 4D or 2D.";
  }
};

#ifdef USE_CUDNN
CudnnBatchNormHandle::CudnnBatchNormHandle(const float momentum,
    const Tensor& input): BatchNormHandle(momentum, input) {
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

const std::vector<Tensor> GpuBatchNormForwardTraining(const CudnnBatchNormHandle &cbnh,
                                   const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
                                   Tensor& running_mean, Tensor& running_var) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(bnScale.device()->lang(), kCuda);
  CHECK_EQ(bnBias.device()->lang(), kCuda);
  CHECK_EQ(running_mean.device()->lang(), kCuda);
  CHECK_EQ(running_var.device()->lang(), kCuda);

  Tensor mean, var;
  mean.ResetLike(running_mean);
  var.ResetLike(running_var);

  Shape shape = x.shape();

  Tensor input = x;  //for unification of 2d and 4d cases.
  if (cbnh.is_2d)
    input.Reshape(Shape{shape.at(0), shape.at(1), 1, 1});

  Tensor output;
  output.ResetLike(x);

  output.device()->Exec(
  [&](Context * ctx) {
    const float alpha = 1.0f, beta = 0.0f;
    double epsilon = CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
                  ctx->cudnn_handle, cbnh.mode, &alpha, &beta, cbnh.shape_desc,
                  input.block()->data(), cbnh.shape_desc, output.block()->mutable_data(),
                  cbnh.param_desc, bnScale.block()->data(), bnBias.block()->data(), cbnh.factor,
                  running_mean.block()->mutable_data(), running_var.block()->mutable_data(),
                  epsilon, mean.block()->mutable_data(),
                  var.block()->mutable_data()));
  },
  {input.block(), bnScale.block(), bnBias.block(), running_mean.block(), running_var.block()}, {
    output.block(), running_mean.block(), running_var.block(),
    mean.block(), var.block()
  });
  if (cbnh.is_2d) output.Reshape(Shape{shape.at(0), shape.at(1)});
  return {output, mean, var};
}

Tensor GpuBatchNormForwardInference(const CudnnBatchNormHandle &cbnh,
                                    const Tensor& x, const Tensor& bnScale,
                                    const Tensor& bnBias, const Tensor& running_mean, const Tensor& running_var) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(bnScale.device()->lang(), kCuda);
  CHECK_EQ(bnBias.device()->lang(), kCuda);
  CHECK_EQ(running_mean.device()->lang(), kCuda);
  CHECK_EQ(running_var.device()->lang(), kCuda);

  Shape shape = x.shape();

  Tensor input = x;  //for unification of 2d and 4d cases.
  if (cbnh.is_2d)
    input.Reshape(Shape{shape.at(0), shape.at(1), 1, 1});

  Tensor output;
  output.ResetLike(x);
  output.device()->Exec(
  [&](Context * ctx) {
    const float alpha = 1.0f, beta = 0.0f;
    double epsilon = CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
                  ctx->cudnn_handle, cbnh.mode, &alpha, &beta, cbnh.shape_desc,
                  input.block()->data(), cbnh.shape_desc, output.block()->mutable_data(),
                  cbnh.param_desc, bnScale.block()->data(), bnBias.block()->data(),
                  running_mean.block()->data(), running_var.block()->data(), epsilon));
  }, { input.block(), bnScale.block(), bnBias.block(), running_mean.block(), running_var.block() },
  {output.block()});
  return output;
}


const std::vector<Tensor> GpuBatchNormBackward(const CudnnBatchNormHandle &cbnh,
    const Tensor& dy, const Tensor& x, const Tensor& bnScale, const Tensor& mean,
    const Tensor& var) {
  CHECK_EQ(dy.device()->lang(), kCuda);
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(bnScale.device()->lang(), kCuda);
  CHECK_EQ(mean.device()->lang(), kCuda);
  CHECK_EQ(var.device()->lang(), kCuda);

  Tensor dx;
  dx.ResetLike(dy);

  Tensor dbnScale;
  dbnScale.ResetLike(bnScale);

  Tensor dbnBias;
  dbnBias.ResetLike(bnScale);

  dx.device()->Exec(
  [&](Context * ctx) {

    const float alpha = 1.0f, beta = .0f;
    double epsilon = CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationBackward(
                  ctx->cudnn_handle, cbnh.mode, &alpha, &beta, &alpha, &beta,
                  cbnh.shape_desc, x.block()->data(), cbnh.shape_desc, dy.block()->data(),
                  cbnh.shape_desc, dx.block()->mutable_data(), cbnh.param_desc,
                  bnScale.block()->data(), dbnScale.block()->mutable_data(),
                  dbnBias.block()->mutable_data(), epsilon, mean.block()->data(),
                  var.block()->data()));
  }, {x.block(), dy.block(), bnScale.block(), mean.block(), var.block()},
  {dx.block(), dbnScale.block(), dbnBias.block()});

  if (cbnh.is_2d) dx.Reshape(Shape{dx.shape().at(0), dx.shape().at(1)});

  return {dx, dbnScale, dbnBias};
}

#endif  //USE_CUDNN
}
