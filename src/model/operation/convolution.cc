/*********************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
************************************************************/
#include <cctype>

#include "./convolution.h"
#include "../layer/convolution.h"

namespace singa {

ConvHandle::ConvHandle(const Tensor &input,
                       const std::vector<size_t>& kernel_size,
                       const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                       const size_t in_channels, const size_t out_channels,
                       const bool bias, const size_t groups) {
  kernel_h = kernel_size[0];
  kernel_w = kernel_size[1];

  pad_h = padding[0];
  pad_w = padding[1];

  stride_h = stride[0];
  stride_w = stride[1];


  channels = in_channels;
  num_filters = out_channels;
  group = groups;

  bias_term = bias;

  batchsize = input.shape(0);
  CHECK(input.shape(1) == in_channels) <<
                                       "the number of input channels mismatched.";
  height = input.shape(2);
  width = input.shape(3);

  conv_height = 1;
  if (stride_h > 0)
    conv_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  conv_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  col_height = in_channels * kernel_w * kernel_h;
  col_width = conv_height * conv_width;
  imagesize = input.Size() / batchsize;

#ifdef USE_MKLDNN
  if (input.device()->lang() == kCpp) {
    const int groups = 1; // only groups 1 is supported for now
    dtype = GetMKLDNNDataType(input.data_type());

    x_dims = {(int)input.shape(0), (int)in_channels, (int)input.shape(2), (int)input.shape(3)};
    b_dims = {(int)out_channels};
    s_dims = {(int)stride_h, (int)stride_w};
    p_dims = {(int)pad_h, (int)pad_w};
    o_dims = {(int)input.shape(0), (int)out_channels, (int)conv_height, (int)conv_width};
    w_dims = {groups, (int)out_channels / groups, (int)in_channels / groups, (int)kernel_size[0], (int)kernel_size[1] };

    x_md = new mkldnn::memory::desc( x_dims, dtype, mkldnn::memory::format::nchw);
    w_md = new mkldnn::memory::desc( w_dims, dtype, mkldnn::memory::format::goihw);
    b_md = new mkldnn::memory::desc( b_dims, dtype, mkldnn::memory::format::x);
    y_md = new mkldnn::memory::desc( o_dims, dtype, mkldnn::memory::format::nchw);

    // convolution forward primitive descriptor is shared between forward and backward process
    conv_d = new mkldnn::convolution_forward::desc(
      mkldnn::prop_kind::forward_inference, mkldnn::convolution_direct, *x_md,
      *w_md, *b_md, *y_md, s_dims,
      p_dims, p_dims, mkldnn::padding_kind::zero);

    auto eng = *input.device()->context(0)->engine;
    conv_pd = new mkldnn::convolution_forward::primitive_desc(*conv_d, eng);

    // mkldnn calculate dw and db in one go, a workaround to be compatible with singa api
    db = new Tensor(Shape{num_filters}, input.device(), input.data_type() );
  }
#endif // USE_MKLDNN
}

ConvHandle::~ConvHandle() {
#ifdef USE_MKLDNN
  if (x_md != nullptr) {
    delete(x_md);
    delete(w_md);
    delete(b_md);
    delete(y_md);
    delete(conv_d);
    delete(conv_pd);
    delete(db);
  }
#endif // USE_MKLDNN
}

Tensor CpuConvForward(const Tensor &x, Tensor &W,  Tensor &b,
                      const ConvHandle &ch) {
  CHECK_EQ(x.device()->lang(), kCpp);

  CHECK(x.shape(1) == ch.channels && x.shape(2) == ch.height &&
        x.shape(3) == ch.width) << "input sample shape should not change";

  CHECK(W.shape(0) == ch.num_filters && W.shape(1) == ch.channels &&
        W.shape(2) == ch.kernel_h
        && W.shape(3) == ch.kernel_w) << "weights shape should not change";

#ifdef USE_MKLDNN

  DataType dtype = x.data_type();
  auto dev = x.device();

  Shape shape{ch.batchsize, ch.num_filters, ch.conv_height, ch.conv_width};
  Tensor output(shape, dev, dtype);

  output.device()->Exec([&output, &x, &W, &b, &ch](Context * ctx) {
    Block *inblock = x.block(), *outblock = output.block(), *wblock = W.block(), *bblock = b.block();

    try {
      using namespace mkldnn;

      auto eng = *ctx->engine;
      auto x_mem = memory({{{ch.x_dims}, ch.dtype, memory::format::nchw}, eng}, inblock->mutable_data());
      auto w_mem = memory({{{ch.w_dims}, ch.dtype, memory::format::goihw}, eng},  wblock->mutable_data());
      auto b_mem = memory({{{ch.b_dims}, ch.dtype, memory::format::x},    eng},  bblock->mutable_data());
      auto y_mem = memory(ch.conv_pd->dst_primitive_desc(), outblock->mutable_data());

      auto conv_fwd = convolution_forward(*ch.conv_pd, x_mem, w_mem, b_mem, y_mem);

      stream(stream::kind::eager).submit({conv_fwd}).wait();
    } catch (mkldnn::error &e) {
      singa::InitLogging("");
      LOG(FATAL) << "MKLDNN conv fwd " << "Status: " << e.status << " Message: " << e.message;
    }

  }, {x.block(), W.block(), b.block()}, {output.block()});

  return output;

#else // ifndef USE_MKLDNN
  Shape w_shape = W.shape();
  Shape b_shape;
  if (ch.bias_term)
    b_shape = b.shape();

  W.Reshape(Shape{ch.num_filters, ch.col_height});
  if (ch.bias_term)
    b.Reshape(Shape{ch.num_filters});

  DataType dtype = x.data_type();
  auto dev = x.device();
  Shape shape{ch.batchsize, ch.num_filters, ch.conv_height, ch.conv_width};
  Tensor output(shape, dev, dtype);

  Tensor col_data(Shape{ch.col_height, ch.col_width});//broadcasted image

  float *data_col = new float[ch.col_height * ch.col_width];
  auto in_data = x.data<float>();
  for (size_t num = 0; num < ch.batchsize; num++) {
    Im2col(in_data + num * ch.imagesize, ch.channels, ch.height, ch.width,
           ch.kernel_h,
           ch.kernel_w, ch.pad_h, ch.pad_w, ch.stride_h, ch.stride_w, data_col);

    col_data.CopyDataFromHostPtr(data_col, ch.col_height * ch.col_width);
    Tensor each = Mult(W, col_data);
    if (ch.bias_term) {
      AddColumn(b, &each);
    }
    CopyDataToFrom(&output, each, each.Size(), num * each.Size());
  };
  W.Reshape(w_shape);
  if (ch.bias_term)
    b.Reshape(b_shape);
  return output;
#endif  // USE_MKLDNN
}

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x,
                        const ConvHandle &ch) {
  CHECK_EQ(dy.device()->lang(), kCpp);

  CHECK(dy.shape(1) == ch.num_filters && dy.shape(2) == ch.conv_height &&
        dy.shape(3) == ch.conv_width) << "input gradients shape should not change";

  CHECK(W.shape(0) == ch.num_filters && W.shape(1) == ch.channels &&
        W.shape(2) == ch.kernel_h
        && W.shape(3) == ch.kernel_w) << "weights shape should not change";


#ifdef USE_MKLDNN

  Tensor dx;
  dx.ResetLike(x);

  dy.device()->Exec([&x, &dx, &dy, &W, &ch](Context * ctx) {
    Block *wblock = W.block(), *dyblock = dy.block(), *dxblock = dx.block(), *inblock = x.block();

    try {
      auto eng = *ctx->engine;
      using namespace mkldnn;
      auto x_mem = memory({{{ch.x_dims}, ch.dtype, memory::format::nchw}, eng}, inblock->mutable_data());
      auto w_mem = memory({{{ch.w_dims}, ch.dtype, memory::format::goihw}, eng}, wblock->mutable_data());
      auto dx_mem = memory({{{ch.x_dims}, ch.dtype, memory::format::nchw}, eng}, dxblock->mutable_data());
      auto dy_mem = memory({{{ch.o_dims}, ch.dtype, memory::format::nchw}, eng}, dyblock->mutable_data());


      auto conv_bwd_data_d = convolution_backward_data::desc(convolution_direct, *ch.x_md, *ch.w_md, *ch.y_md, ch.s_dims,
                             ch.p_dims, ch.p_dims, padding_kind::zero);
      auto conv_bwd_data_pd = convolution_backward_data::primitive_desc(conv_bwd_data_d, eng, *ch.conv_pd);
      auto conv_bwd_data = convolution_backward_data(conv_bwd_data_pd, dy_mem, w_mem, dx_mem);


      stream(stream::kind::eager).submit({conv_bwd_data}).wait();
    } catch (mkldnn::error &e) {
      singa::InitLogging("");
      LOG(FATAL) << "MKLDNN conv fwd " << "Status: " << e.status << " Message: " << e.message;
    }

  }, {x.block(), dy.block(), W.block()}, {dx.block()});

  return dx;

#else // ifndef USE_MKLDNN
  Shape w_shape = W.shape();
  W.Reshape(Shape{ch.num_filters, ch.col_height});

  Tensor dx;
  dx.ResetLike(x);

  float *dx_b = new float[ch.imagesize];

  for (size_t num = 0; num < ch.batchsize; num++) {
    Tensor grad_b(Shape{ch.num_filters, ch.conv_height * ch.conv_width});
    CopyDataToFrom(&grad_b, dy, grad_b.Size(), 0, num * grad_b.Size());
    Tensor dcol_b = Mult(Transpose(W), grad_b);
    auto dcol_data = dcol_b.data<float>();
    Col2im(dcol_data, ch.channels, ch.height, ch.width, ch.kernel_h, ch.kernel_w,
           ch.pad_h,
           ch.pad_w, ch.stride_h, ch.stride_w, dx_b);
    dx.CopyDataFromHostPtr(dx_b, ch.imagesize, num * ch.imagesize);
  }
  W.Reshape(w_shape);
  return dx;
#endif  // USE_MKLDNN
}

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W,
                        const ConvHandle &ch) {
  CHECK_EQ(dy.device()->lang(), kCpp);

  CHECK(dy.shape(1) == ch.num_filters && dy.shape(2) == ch.conv_height &&
        dy.shape(3) == ch.conv_width) << "input gradients shape should not change";

  CHECK(x.shape(1) == ch.channels && x.shape(2) == ch.height &&
        x.shape(3) == ch.width) << "input sample shape should not change";

#ifdef USE_MKLDNN
  Tensor dW;
  dW.ResetLike(W);

  dy.device()->Exec([&x, &dy, &dW, &ch](Context * ctx) {
    Block *dwblock = dW.block(), *dyblock = dy.block(), *inblock = x.block(), *dbblock = ch.db->block();

    try {
      auto eng = *ctx->engine;
      using namespace mkldnn;

      auto x_mem = memory({{{ch.x_dims}, ch.dtype, memory::format::nchw}, eng}, inblock->mutable_data());
      auto dy_mem = memory({{{ch.o_dims}, ch.dtype, memory::format::nchw}, eng}, dyblock->mutable_data());
      auto dw_mem = memory({{{ch.w_dims}, ch.dtype, memory::format::goihw}, eng}, dwblock->mutable_data());
      auto db_mem = memory({{{ch.b_dims}, ch.dtype, memory::format::x}, eng}, dbblock->mutable_data());

      auto conv_dw_d = convolution_backward_weights::desc(convolution_direct, *ch.x_md, *ch.w_md, *ch.b_md, *ch.y_md,
                       ch.s_dims, ch.p_dims, ch.p_dims, padding_kind::zero);
      auto conv_dw_pd = convolution_backward_weights::primitive_desc(conv_dw_d, eng, *ch.conv_pd);
      auto conv_dw = convolution_backward_weights(conv_dw_pd, x_mem, dy_mem, dw_mem, db_mem);

      mkldnn::stream(mkldnn::stream::kind::eager).submit({conv_dw}).wait();
    } catch (mkldnn::error &e) {
      singa::InitLogging("");
      LOG(FATAL) << "MKLDNN Conv backward W " << "Status: " << e.status << " Message: " << e.message;
    }

  }, {x.block(), dy.block(), W.block()}, {dW.block()});

  return dW;


#else // USE_MKLDNN
  Tensor dW;
  dW.ResetLike(W);
  dW.SetValue(0.0f);

  Shape w_shape = W.shape();
  dW.Reshape(Shape{ch.num_filters, ch.col_height});

  Tensor col_data(Shape{ch.col_height, ch.col_width});//broadcasted image

  float *data_col = new float[ch.col_height * ch.col_width];
  auto in_data = dy.data<float>();
  for (size_t num = 0; num < ch.batchsize; num++) {
    Im2col(in_data + num * ch.imagesize, ch.channels, ch.height, ch.width,
           ch.kernel_h,
           ch.kernel_w, ch.pad_h, ch.pad_w, ch.stride_h, ch.stride_w, data_col);
    col_data.CopyDataFromHostPtr(data_col, ch.col_height * ch.col_width);
    Tensor grad_b(Shape{ch.num_filters, ch.conv_height * ch.conv_width});
    CopyDataToFrom(&grad_b, dy, grad_b.Size(), 0, num * grad_b.Size());
    dW += Mult(grad_b, Transpose(col_data));
  }
  dW.Reshape(w_shape);
  return dW;
#endif // USE_MKLDNN
}

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b,
                        const ConvHandle &ch) {
  CHECK_EQ(dy.device()->lang(), kCpp);

  CHECK(dy.shape(1) == ch.num_filters && dy.shape(2) == ch.conv_height &&
        dy.shape(3) == ch.conv_width) << "input gradients shape should not change";

  CHECK(b.shape(0) == ch.num_filters) << "bias shape should not change";

#ifdef USE_MKLDNN
  Tensor db = ch.db->Clone();
  return db;
#else // USE_MKLDNN
  Tensor db;
  db.ResetLike(b);

  auto tmpshp = Shape{ch.batchsize * ch.num_filters, dy.Size() / (ch.batchsize * ch.num_filters)};
  Tensor tmp1 = Reshape(dy, tmpshp);

  Tensor tmp2(Shape{ch.batchsize * ch.num_filters});
  SumColumns(tmp1, &tmp2);
  Tensor tmp3 = Reshape(tmp2, Shape{ch.batchsize, ch.num_filters});

  SumRows(tmp3, &db);

  return db;
#endif // USE_MKLDNN
};

#ifdef USE_CUDNN
CudnnConvHandle::CudnnConvHandle(const Tensor &input,
                                 const std::vector<size_t>& kernel_size,
                                 const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                                 const size_t in_channels, const size_t out_channels, const bool bias,
                                 const size_t groups,
                                 const size_t workspace_byte_limit, const std::string& prefer_)
  : ConvHandle(input, kernel_size, stride, padding, in_channels, out_channels,
               bias, groups) {
  std::string prefer = prefer_;
  if (const char* env_p = std::getenv("CUDNN_CONV_ALG")) {
    prefer = std::string(env_p);
    std::transform(prefer.begin(), prefer.end(), prefer.begin(), tolower);
    LOG(INFO) << "CUDNN_CONV_ALG: " << prefer;
  }
  DataType dtype = input.data_type();
  auto dev = input.device();
  Context *ctx = dev->context(0);
  channels_per_filter = channels/groups;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  if (bias_term)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize,
                                         channels, height, width));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                y_desc, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize,
                num_filters, conv_height, conv_width));
  if (bias_term)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                           GetCudnnDataType(dtype), 1,
                                           num_filters, 1, 1));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w,
              stride_h, stride_w, 1, 1,
              CUDNN_CROSS_CORRELATION
#if CUDNN_MAJOR >= 7
              , GetCudnnDataType(dtype)
#endif
                                             ));
  if (CUDNN_MAJOR >= 7 && groups > 1) {
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, groups));
  } else if (groups > 1) {LOG(FATAL) << "The current version of cuDNN not support grouped convolution.";};

  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, GetCudnnDataType(dtype),
                                         CUDNN_TENSOR_NCHW, num_filters,
                                         channels / groups, kernel_h, kernel_w));
  if (prefer == "fastest" || prefer == "limited_workspace" ||
      prefer == "no_workspace") {
    cudnnConvolutionFwdPreference_t fwd_pref;
    cudnnConvolutionBwdFilterPreference_t bwd_filt_pref;
    cudnnConvolutionBwdDataPreference_t bwd_data_pref;
    if (prefer == "fastest") {
      fwd_pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
      bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
      bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
    } else if (prefer == "limited_workspace") {
      fwd_pref = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
      bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
      bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
    } else {
      fwd_pref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
      bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
      bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
    }
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
                  ctx->cudnn_handle, x_desc, filter_desc, conv_desc, y_desc, fwd_pref,
                  workspace_byte_limit, &fp_alg));
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
                  ctx->cudnn_handle, x_desc, y_desc, conv_desc, filter_desc,
                  bwd_filt_pref, workspace_byte_limit, &bp_filter_alg));
    // deprecated in cudnn v7
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
                  ctx->cudnn_handle, filter_desc, y_desc, conv_desc, x_desc,
                  bwd_data_pref, workspace_byte_limit, &bp_data_alg));
  } else if (prefer == "autotune") {
    const int topk = 1;
    int num_fp_alg, num_bp_filt_alg, num_bp_data_alg;
    cudnnConvolutionFwdAlgoPerf_t fp_algperf[topk];
    cudnnConvolutionBwdFilterAlgoPerf_t bp_filt_perf[topk];
    cudnnConvolutionBwdDataAlgoPerf_t bp_data_perf[topk];
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
                  ctx->cudnn_handle, x_desc, filter_desc, conv_desc, y_desc, topk,
                  &num_fp_alg, fp_algperf));
    fp_alg = fp_algperf[0].algo;
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
                  ctx->cudnn_handle, x_desc, y_desc, conv_desc, filter_desc, topk,
                  &num_bp_filt_alg, bp_filt_perf));
    bp_filter_alg = bp_filt_perf[0].algo;
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
                  ctx->cudnn_handle, filter_desc, y_desc, conv_desc, x_desc, topk,
                  &num_bp_data_alg, bp_data_perf));
    bp_data_alg = bp_data_perf[0].algo;
  } else {
    LOG(FATAL) << "Preferred algorithm is not available :" << prefer;
  }

  size_t fp_byte, bp_data_byte, bp_filter_byte;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                ctx->cudnn_handle, x_desc, filter_desc, conv_desc, y_desc, fp_alg,
                &fp_byte));
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                ctx->cudnn_handle, filter_desc, y_desc, conv_desc, x_desc,
                bp_data_alg, &bp_data_byte));
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                ctx->cudnn_handle, x_desc, y_desc, conv_desc, filter_desc,
                bp_filter_alg, &bp_filter_byte));
  workspace_count = std::max(std::max(fp_byte, bp_data_byte), bp_filter_byte) /
                    sizeof(float) +
                    1;
  if (workspace_count * sizeof(float) > workspace_byte_limit)
    LOG(WARNING) << "The required memory for workspace ("
                 << workspace_count * sizeof(float)
                 << ") is larger than the expected Bytes ("
                 << workspace_byte_limit << ")";
  workspace = Tensor(Shape{workspace_count}, dev, dtype);

}

CudnnConvHandle::~CudnnConvHandle() {
  if (bias_desc != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
  if (filter_desc != nullptr)
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
  if (conv_desc != nullptr)
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
  if (x_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  if (y_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
}

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b,
                      const CudnnConvHandle &cch) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK(x.shape(1) == cch.channels && x.shape(2) == cch.height &&
        x.shape(3) == cch.width) << "input sample shape should not change";

  CHECK(W.shape(0) == cch.num_filters && W.shape(1) == cch.channels_per_filter &&
        W.shape(2) == cch.kernel_h
        && W.shape(3) == cch.kernel_w) << "weights shape should not change";

  DataType dtype = x.data_type();
  auto dev = x.device();

  Shape shape{cch.batchsize, cch.num_filters, cch.conv_height, cch.conv_width};
  Tensor output(shape, dev, dtype);

  output.device()->Exec([&output, &x, &W, &cch](Context * ctx) {
    Block *inblock = x.block(), *outblock = output.block(),
           *wblock = W.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionForward(ctx->cudnn_handle, &alpha, cch.x_desc,
                            inblock->data(), cch.filter_desc, wblock->data(),
                            cch.conv_desc, cch.fp_alg,
                            cch.workspace.block()->mutable_data(),
                            cch.workspace_count * sizeof(float), &beta,
                            cch.y_desc, outblock->mutable_data());
  }, {x.block(), W.block()}, {output.block()}, cch.workspace.block());

  if (cch.bias_term) {
    output.device()->Exec([&output, &b, &cch](Context * ctx) {
      float beta = 1.f, alpha = 1.0f;
      Block *outblock = output.block(), *bblock = b.block();
      cudnnAddTensor(ctx->cudnn_handle, &alpha, cch.bias_desc,
                     bblock->data(), &beta, cch.y_desc,
                     outblock->mutable_data());
    }, {output.block(), b.block()}, {output.block()});
  }

  return output;
}

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x,
                        const CudnnConvHandle &cch) {
  CHECK_EQ(dy.device()->lang(), kCuda);

  Tensor dx;
  dx.ResetLike(x);

  dy.device()->Exec([&dx, &dy, &W, &cch](Context * ctx) {
    Block *wblock = W.block(), *dyblock = dy.block(),
           *dxblock = dx.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardData(ctx->cudnn_handle, &alpha, cch.filter_desc,
                                 wblock->data(), cch.y_desc, dyblock->data(),
                                 cch.conv_desc, cch.bp_data_alg,
                                 cch.workspace.block()->mutable_data(),
                                 cch.workspace_count * sizeof(float), &beta,
                                 cch.x_desc, dxblock->mutable_data());
  }, {dy.block(), W.block()}, {dx.block(), cch.workspace.block()});

  return dx;
}

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W,
                        const CudnnConvHandle &cch) {
  CHECK_EQ(dy.device()->lang(), kCuda);

  Tensor dW;
  dW.ResetLike(W);

  dy.device()->Exec([&dW, &dy, &x, &cch](Context * ctx) {
    Block *inblock = x.block(), *dyblock = dy.block(),
           *dwblock = dW.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardFilter(
      ctx->cudnn_handle, &alpha, cch.x_desc, inblock->data(),
      cch.y_desc, dyblock->data(), cch.conv_desc, cch.bp_filter_alg,
      cch.workspace.block()->mutable_data(),
      cch.workspace_count * sizeof(float), &beta, cch.filter_desc,
      dwblock->mutable_data());
  }, {dy.block(), x.block()}, {dW.block(), cch.workspace.block()});

  return dW;
}

// input Tensor b for Reset db purpose, can avoid this later.
Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b,
                        const CudnnConvHandle &cch) {
  CHECK_EQ(dy.device()->lang(), kCuda);

  Tensor db;
  db.ResetLike(b);

  dy.device()->Exec([&db, &dy, &cch](Context * ctx) {
    Block *dyblock = dy.block(), *dbblock = db.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardBias(ctx->cudnn_handle, &alpha, cch.y_desc,
                                 dyblock->data(), &beta, cch.bias_desc,
                                 dbblock->mutable_data());
  }, {dy.block()}, {db.block()});

  return db;
}
#endif  // USE_CUDNN

}  // namespace singa
