#include "./pooling.h"
#include <cmath>

namespace singa {

PoolingHandle::PoolingHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
                             const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                             const bool ceil_mode, const std::string pooling_method) {
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
  if (ceil_mode) {
    if (stride_h > 0)
      pooled_height = static_cast<int>(ceil(static_cast<float>(height + 2 * pad_h - kernel_h) / stride_h)) + 1;
    pooled_width = static_cast<int>(ceil(static_cast<float>(width + 2 * pad_w - kernel_w) / stride_w)) + 1;
  }
  else {
    if (stride_h > 0)
      pooled_height =
        static_cast<size_t>((height + 2 * pad_h - kernel_h) / stride_h) + 1;
    pooled_width =
      static_cast<size_t>((width + 2 * pad_w - kernel_w) / stride_w) + 1;
  }

  method = pooling_method;
  CHECK(method == "MAX" || method == "AVERAGE")
      << "Padding implemented only for average and max pooling.";
}

#ifdef USE_CUDNN

CudnnPoolingHandle::CudnnPoolingHandle(const Tensor &input, const std::vector<size_t>& kernel_size,
                                       const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                                       const bool ceil_mode, const std::string pooling_method, const bool NaN_prop)
  : PoolingHandle(input, kernel_size, stride, padding, ceil_mode, pooling_method) {
  if (NaN_prop)
    nan_prop = CUDNN_PROPAGATE_NAN;
  else
    nan_prop = CUDNN_NOT_PROPAGATE_NAN;

  DataType dtype = input.data_type();

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));


  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize,
                                         channels, height, width));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                y_desc, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize, channels,
                pooled_height, pooled_width));
  auto pool_method = CUDNN_POOLING_MAX;
  if (method == "MAX")
    pool_method = CUDNN_POOLING_MAX;
  else if (method == "AVERAGE")
    pool_method = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  else
    LOG(FATAL) << "Not implemented!";

  CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc, pool_method, nan_prop,
                                          kernel_h, kernel_w, pad_h, pad_w,
                                          stride_h, stride_w));
};

CudnnPoolingHandle::~CudnnPoolingHandle() {
  if (pool_desc != nullptr)
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
  if (x_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  if (y_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
};

Tensor GpuPoolingForward(const Tensor &x, const CudnnPoolingHandle &cph) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(x.nDim(), 4u);

  DataType dtype = x.data_type();
  auto dev = x.device();
  Shape shape{cph.batchsize, cph.channels, cph.pooled_height, cph.pooled_width};
  Tensor output = Tensor(shape, dev, dtype);

  output.device()->Exec([&x, &output, &cph](Context * ctx) {
    Block *inblock = x.block(), *outblock = output.block();
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingForward(ctx->cudnn_handle, cph.pool_desc, &alpha,
                        cph.x_desc, inblock->data(), &beta, cph.y_desc,
                        outblock->mutable_data());
  }, {x.block()}, {output.block()});
  return output;
};

Tensor GpuPoolingBackward(const Tensor &dy, const Tensor& x, const Tensor& y,
                          const CudnnPoolingHandle &cph) {
  CHECK_EQ(dy.device()->lang(), kCuda);
  CHECK_EQ(dy.nDim(), 4u);

  Tensor dx;
  dx.ResetLike(x);

  dx.device()->Exec([&dx, &dy, &x, &y, &cph](Context * ctx) {
    Block *dyblock = dy.block(), *dxblock = dx.block(), *yblock = y.block(),
           *xblock = x.block();
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingBackward(ctx->cudnn_handle, cph.pool_desc, &alpha,
                         cph.y_desc, yblock->data(), cph.y_desc,
                         dyblock->data(), cph.x_desc, xblock->data(), &beta,
                         cph.x_desc, dxblock->mutable_data());
  }, {dy.block(), y.block(), x.block()}, {dx.block()});
  return dx;
};
#endif  //USE_CUDNN

}  //namespace singa
