#include "./pooling.h"
#include <cmath>

namespace singa {

PoolingHandle::PoolingHandle(const Tensor &input,
                             const std::vector<int>& kernel_size,
                             const std::vector<int>& stride, const std::vector<int>& padding,
                             const bool is_max) {
  kernel_h = kernel_size[0];
  kernel_w = kernel_size[1];

  pad_h = padding[0];
  pad_w = padding[1];

  stride_h = stride[0];
  stride_w = stride[1];

  batchsize = input.shape(0);
  channels = input.shape(1);
  height = input.shape(2);
  width = input.shape(3);

  pooled_height = 1;

  if (stride_h > 0)
    pooled_height = std::floor(
      ((height + 2 * pad_h - kernel_h) / stride_h)) + 1;
  pooled_width = std::floor(
    ((width + 2 * pad_w - kernel_w) / stride_w)) + 1;
  is_max_pooling = is_max;
}

#ifdef USE_CUDNN

CudnnPoolingHandle::CudnnPoolingHandle(const Tensor &input,
                                       const std::vector<int>& kernel_size,
                                       const std::vector<int>& stride,
                                       const std::vector<int>& padding,
                                       const bool is_max)
  : PoolingHandle(input, kernel_size, stride, padding, is_max) {

//nan_prop = CUDNN_NOT_PROPAGATE_NAN;

  DataType dtype = input.data_type();

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));


  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize,
                                         channels, height, width));
  // LOG(ERROR) << batchsize << " " << channels << " " << pooled_height << " " << pooled_width;
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                y_desc, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize, channels,
                pooled_height, pooled_width));
  auto pool_method = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  if (is_max)
    pool_method = CUDNN_POOLING_MAX;

  CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc, pool_method, nan_prop,
                                          kernel_h, kernel_w, pad_h, pad_w,
                                          stride_h, stride_w));
};

CudnnPoolingHandle::~CudnnPoolingHandle() {
  if (pool_desc != nullptr)
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
  if (x_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  if (y_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
}


Tensor GpuPoolingForward(const CudnnPoolingHandle &cph, const Tensor &x) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(x.nDim(), 4u);

  Tensor output = Tensor({cph.batchsize, cph.channels, cph.pooled_height, cph.pooled_width},
                         x.device(), x.data_type());

  output.device()->Exec([&](Context * ctx) {
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingForward(ctx->cudnn_handle, cph.pool_desc, &alpha,
                        cph.x_desc, x.block()->data(), &beta, cph.y_desc,
                        output.block()->mutable_data());
  }, {x.block()}, {output.block()});
  return output;
}

Tensor GpuPoolingBackward(const CudnnPoolingHandle &cph, const Tensor &dy,
                          const Tensor& x, const Tensor& y) {
  CHECK_EQ(dy.device()->lang(), kCuda);
  CHECK_EQ(dy.nDim(), 4u);

  Tensor dx;
  dx.ResetLike(x);

  dx.device()->Exec([&](Context * ctx) {

    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingBackward(ctx->cudnn_handle, cph.pool_desc, &alpha,
                         cph.y_desc, y.block()->data(), cph.y_desc,
                         dy.block()->data(), cph.x_desc, x.block()->data(), &beta,
                         cph.x_desc, dx.block()->mutable_data());
  }, {dy.block(), y.block(), x.block()}, {dx.block()});
  return dx;
};
#endif  //USE_CUDNN

}  //namespace singa
