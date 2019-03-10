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


#ifdef USE_MKLDNN
  dtype = GetMKLDNNDataType(input.data_type());
  x_dims = {batchsize, channels, height, width};
  y_dims = {batchsize, channels, pooled_height, pooled_width};
  s_dims = {stride};
  k_dims = {kernel_size};
  p_dims = {padding};

  auto eng = *input.device()->context(0)->engine;
  x_md = new mkldnn::memory::desc({x_dims}, dtype, mkldnn::memory::format::nchw);
  y_md = new mkldnn::memory::desc({y_dims}, dtype, mkldnn::memory::format::nchw);

  // allow max or avg (follow cudnn implementation convention)
  pooling_algo = mkldnn::pooling_avg_exclude_padding;
  if(is_max_pooling)
    pooling_algo = mkldnn::pooling_max;

  pool_fwd_d = new mkldnn::pooling_forward::desc(mkldnn::forward_training, pooling_algo, *x_md, *y_md, s_dims,
                                                 k_dims, p_dims, p_dims, mkldnn::padding_kind::zero);
  pool_fwd_pd = new mkldnn::pooling_forward::primitive_desc(*pool_fwd_d, eng);

  if (is_max_pooling) {
//    During training max pooling requires workspace on forward (mkldnn_forward_training) and backward
//    (mkldnn_backward) passes to save indices where maximum was found. Workspace layout is opaque and
//    the indices cannot be restored from it. However one can use backward pooling to perform up-sampling
//    (used in some detection topologies).
    auto temp = pool_fwd_pd->workspace_primitive_desc();
    pool_ws_d = &temp;
    ws_mem = new mkldnn::memory(*pool_ws_d);
  }

#endif // USE_MKLDNN
}

PoolingHandle::~PoolingHandle(){
#ifdef USE_MKLDNN
  delete(x_md);
  delete(y_md);
  delete(pool_fwd_d);
  delete(pool_fwd_pd);
  if (is_max_pooling)
    delete(ws_mem);
#endif // USE_MKLDNN
}

#ifdef USE_MKLDNN

Tensor CpuPoolingForward(const PoolingHandle &ph, const Tensor &x) {


  Tensor y({(unsigned long) ph.batchsize, (unsigned long) ph.channels, (unsigned long) ph.pooled_height,
            (unsigned long) ph.pooled_width}, x.device(), x.data_type());


  y.device()->Exec([&y, &x, &ph](Context *ctx) {


    try {

      auto eng = *ctx->engine;
      using namespace mkldnn;

      auto y_mem = memory(ph.pool_fwd_pd->dst_primitive_desc(), y.block()->mutable_data());
      auto x_mem = memory({{{ph.x_dims}, ph.dtype, memory::format::nchw}, eng},
                          x.block()->mutable_data());

      auto p_fwd = ph.is_max_pooling ? pooling_forward(*ph.pool_fwd_pd, x_mem, y_mem, *ph.ws_mem) : pooling_forward(
          *ph.pool_fwd_pd, x_mem, y_mem);

      stream(stream::kind::eager).submit({p_fwd}).wait();
    }
    catch (mkldnn::error &e) {
      LOG(FATAL) << "MKLDNN pooling fwd" << "Status: " << e.status << " Message: " << e.message;
    }

  }, {x.block()}, {y.block()});

  return y;

}

Tensor CpuPoolingBackward(const PoolingHandle &ph, const Tensor &grad, const Tensor &x, const Tensor &y) {


  Tensor in_grad;
  in_grad.ResetLike(x);

  in_grad.device()->Exec([&in_grad, &grad, &ph](Context *ctx) {
    try {
      auto eng = *ctx->engine;
      using namespace mkldnn;
      auto pool_bwd_d = pooling_backward::desc(ph.pooling_algo, *ph.x_md, *ph.y_md, ph.s_dims, ph.k_dims, ph.p_dims,
                                               ph.p_dims,
                                               padding_kind::zero);
      auto pool_bwd_pd = pooling_backward::primitive_desc(pool_bwd_d, eng, *ph.pool_fwd_pd);

      auto dx_mem = memory({{{ph.x_dims}, ph.dtype, memory::format::nchw}, eng},
                           in_grad.block()->mutable_data());
      auto dy_mem = memory({{{ph.y_dims}, memory::data_type::f32, memory::format::nchw}, eng},
                           grad.block()->mutable_data());

      auto p_bwd = ph.is_max_pooling ? pooling_backward(pool_bwd_pd, dy_mem, *ph.ws_mem, dx_mem) : pooling_backward( pool_bwd_pd, dy_mem, dx_mem);

      stream(stream::kind::eager).submit({p_bwd}).wait();
    }
    catch (mkldnn::error &e) {
      LOG(FATAL) << "MKLDNN pooling bwd" << "Status: " << e.status << " Message: " << e.message;
    }

  }, {x.block(), y.block(), grad.block()}, {in_grad.block()});

  return in_grad;

}

#endif

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
