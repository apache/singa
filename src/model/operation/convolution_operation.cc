#include "./convolution_operation.h"
#include "../layer/convolution.h"
#include<iostream>

namespace singa{

ConvHandles::ConvHandles(const Tensor &input, const std::vector<size_t> kernel_size, 
	                const std::vector<size_t> stride, const std::vector<size_t> padding,
	                const size_t in_channels, const size_t out_channels,
	                const bool bias_term_){
    kernel_h_=kernel_size[0];
    kernel_w_=kernel_size[1];

    pad_h_=padding[0];
    pad_w_=padding[1];

    stride_h_=stride[0];
    stride_w_=stride[1];

    channels_=in_channels;
    num_filters_=out_channels;

	batchsize = input.shape(0);
	CHECK(input.shape(1) == in_channels)<<"the number of input channels mismatched.";
    height_ = input.shape(2);
    width_ = input.shape(3);

    conv_height_ = 1;
    if (stride_h_ > 0)
        conv_height_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    conv_width_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

    col_height_ = in_channels * kernel_w_ * kernel_h_;
    col_width_ = conv_height_ * conv_width_;
    imagesize = input.Size() / batchsize;
};	

CudnnConvHandles::CudnnConvHandles(const Tensor &input, const std::vector<size_t> kernel_size, 
                    const std::vector<size_t> stride, const std::vector<size_t> padding,
                    const size_t in_channels, const size_t out_channels,const bool bias_term_, 
                    const size_t workspace_byte_limit_,const std::string prefer_)
                    :ConvHandles(input, kernel_size, stride, padding, in_channels, out_channels, bias_term_){

    DataType dtype = input.data_type();
    auto dev = input.device();
    Context *ctx = dev->context(0);

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    if (bias_term_)
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));


    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW,
                                           GetCudnnDataType(dtype), batchsize,
                                           channels_, height_, width_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            y_desc_, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize,
            num_filters_, conv_height_, conv_width_));
    if (bias_term_)
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW,
                                               GetCudnnDataType(dtype), 1,
                                               num_filters_, 1, 1));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_, pad_h_, pad_w_,
                                                stride_h_, stride_w_, 1, 1,
                                                CUDNN_CROSS_CORRELATION,
                                                GetCudnnDataType(dtype)));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, GetCudnnDataType(dtype),
                                           CUDNN_TENSOR_NCHW, num_filters_,
                                           channels_, kernel_h_, kernel_w_));
    if (prefer_ == "fastest" || prefer_ == "limited_workspace" ||
        prefer_ == "no_workspace") {
        cudnnConvolutionFwdPreference_t fwd_pref;
        cudnnConvolutionBwdFilterPreference_t bwd_filt_pref;
        cudnnConvolutionBwdDataPreference_t bwd_data_pref;
        if (prefer_ == "fastest") {
            fwd_pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
            bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
            bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
        } else if (prefer_ == "limited_workspace") {
            fwd_pref = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
            bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
            bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
        } else {
            fwd_pref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
            bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
            bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
        }
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
                ctx->cudnn_handle, x_desc_, filter_desc_, conv_desc_, y_desc_, fwd_pref,
                workspace_byte_limit_, &fp_alg_));
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
                ctx->cudnn_handle, x_desc_, y_desc_, conv_desc_, filter_desc_,
                bwd_filt_pref, workspace_byte_limit_, &bp_filter_alg_));
        // deprecated in cudnn v7
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
                ctx->cudnn_handle, filter_desc_, y_desc_, conv_desc_, x_desc_,
                bwd_data_pref, workspace_byte_limit_, &bp_data_alg_));
        } else if (prefer_ == "autotune") {
        const int topk = 1;
        int num_fp_alg, num_bp_filt_alg, num_bp_data_alg;
        cudnnConvolutionFwdAlgoPerf_t fp_alg_perf[topk];
        cudnnConvolutionBwdFilterAlgoPerf_t bp_filt_perf[topk];
        cudnnConvolutionBwdDataAlgoPerf_t bp_data_perf[topk];
        CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
                ctx->cudnn_handle, x_desc_, filter_desc_, conv_desc_, y_desc_, topk,
                &num_fp_alg, fp_alg_perf));
        fp_alg_ = fp_alg_perf[0].algo;
        CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
                ctx->cudnn_handle, x_desc_, y_desc_, conv_desc_, filter_desc_, topk,
                &num_bp_filt_alg, bp_filt_perf));
        bp_filter_alg_ = bp_filt_perf[0].algo;
        CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
                ctx->cudnn_handle, filter_desc_, y_desc_, conv_desc_, x_desc_, topk,
                &num_bp_data_alg, bp_data_perf));
        bp_data_alg_ = bp_data_perf[0].algo;
    } else {
        LOG(FATAL) << "Preferred algorithm is not available!";
    }

    size_t fp_byte, bp_data_byte, bp_filter_byte;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            ctx->cudnn_handle, x_desc_, filter_desc_, conv_desc_, y_desc_, fp_alg_,
            &fp_byte));
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            ctx->cudnn_handle, filter_desc_, y_desc_, conv_desc_, x_desc_,
            bp_data_alg_, &bp_data_byte));
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            ctx->cudnn_handle, x_desc_, y_desc_, conv_desc_, filter_desc_,
            bp_filter_alg_, &bp_filter_byte));
    workspace_count_ = std::max(std::max(fp_byte, bp_data_byte), bp_filter_byte) /
                       sizeof(float) +
                       1;
    if (workspace_count_ * sizeof(float) > workspace_byte_limit_)
        LOG(WARNING) << "The required memory for workspace ("
                     << workspace_count_ * sizeof(float)
                     << ") is larger than the expected Bytes ("
                     << workspace_byte_limit_ << ")";
    workspace_ = Tensor(Shape{workspace_count_}, dev, dtype);
};

Convolution C;

Tensor CpuConvForward(const Tensor &x, Tensor &W,  Tensor &b, const ConvHandles ch){
	CHECK_EQ(x.device()->lang(), kCpp);

	CHECK(x.shape(1) == ch.channels_ && x.shape(2) == ch.height_ &&
    x.shape(3) == ch.width_) << "input sample shape should not change";

    CHECK(W.shape(0) == ch.num_filters_ && W.shape(1) == ch.channels_ && 
    W.shape(2) == ch.kernel_h_ && W.shape(3) == ch.kernel_w_) << "weights shape should not change";

    Shape w_shape= W.shape();
    Shape b_shape= b.shape();

    W.Reshape(Shape{ch.num_filters_, ch.col_height_});
    if (ch.bias_term_)
      b.Reshape(Shape{ch.num_filters_});

    DataType dtype = x.data_type();
    auto dev = x.device();
    Shape shape{ch.batchsize, ch.num_filters_, ch.conv_height_, ch.conv_width_};
    Tensor output(shape, dev, dtype);

    Tensor col_data(Shape{ch.col_height_, ch.col_width_});//broadcasted image

    float *data_col = new float[ch.col_height_ * ch.col_width_];
    auto in_data = x.data<float>();
    for (size_t num = 0; num < ch.batchsize; num++) {
      C.Im2col(in_data + num * ch.imagesize, ch.channels_, ch.height_, ch.width_, ch.kernel_h_,
            ch.kernel_w_, ch.pad_h_, ch.pad_w_, ch.stride_h_, ch.stride_w_, data_col);    

      col_data.CopyDataFromHostPtr(data_col, ch.col_height_ * ch.col_width_);
      Tensor each = Mult(W, col_data);
      if (ch.bias_term_) {
          AddColumn(b, &each);
        }
      CopyDataToFrom(&output, each, each.Size(), num * each.Size());
    };
  W.Reshape(w_shape);
  b.Reshape(b_shape);
  return output;
}; 

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x, const ConvHandles ch){
    CHECK_EQ(dy.device()->lang(), kCpp);
    
    CHECK(dy.shape(1) == ch.num_filters_ && dy.shape(2) == ch.conv_height_ &&
    dy.shape(3) == ch.conv_width_) << "input gradients shape should not change";

    CHECK(W.shape(0) == ch.num_filters_ && W.shape(1) == ch.channels_ && 
    W.shape(2) == ch.kernel_h_ && W.shape(3) == ch.kernel_w_) << "weights shape should not change";

    Shape w_shape= W.shape();
    W.Reshape(Shape{ch.num_filters_, ch.col_height_});

    Tensor dx;
    dx.ResetLike(x);
    
    float *dx_b = new float[ch.imagesize];

    for (size_t num = 0; num < ch.batchsize; num++) {
      Tensor grad_b(Shape{ch.num_filters_, ch.conv_height_ * ch.conv_width_});
      CopyDataToFrom(&grad_b, dy, grad_b.Size(), 0, num * grad_b.Size());
      Tensor dcol_b = Mult(W.T(), grad_b);
      auto dcol_data = dcol_b.data<float>();
      C.Col2im(dcol_data, ch.channels_, ch.height_, ch.width_, ch.kernel_h_, ch.kernel_w_, ch.pad_h_,
           ch.pad_w_, ch.stride_h_, ch.stride_w_, dx_b);
      dx.CopyDataFromHostPtr(dx_b, ch.imagesize, num * ch.imagesize);
    }
  W.Reshape(w_shape); 
  return dx;
};

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const ConvHandles ch){
    CHECK_EQ(dy.device()->lang(), kCpp);
    
    CHECK(dy.shape(1) == ch.num_filters_ && dy.shape(2) == ch.conv_height_ &&
    dy.shape(3) == ch.conv_width_) << "input gradients shape should not change";

    CHECK(x.shape(1) == ch.channels_ && x.shape(2) == ch.height_ &&
    x.shape(3) == ch.width_) << "input sample shape should not change";

    Tensor dW;
    dW.ResetLike(W);
    dW.SetValue(0.0f);
    
    Shape w_shape= W.shape();
    dW.Reshape(Shape{ch.num_filters_, ch.col_height_});

    Tensor col_data(Shape{ch.col_height_, ch.col_width_});//broadcasted image

    float *data_col = new float[ch.col_height_ * ch.col_width_];
    auto in_data = dy.data<float>();
    for (size_t num = 0; num < ch.batchsize; num++) {
      C.Im2col(in_data + num * ch.imagesize, ch.channels_, ch.height_, ch.width_, ch.kernel_h_,
            ch.kernel_w_, ch.pad_h_, ch.pad_w_, ch.stride_h_, ch.stride_w_, data_col);
      col_data.CopyDataFromHostPtr(data_col, ch.col_height_ * ch.col_width_);
      Tensor grad_b(Shape{ch.num_filters_, ch.conv_height_ * ch.conv_width_});
      CopyDataToFrom(&grad_b, dy, grad_b.Size(), 0, num * grad_b.Size());
      dW += Mult(grad_b, col_data.T());
    }
   dW.Reshape(w_shape);
   return dW;
};

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b, const ConvHandles ch){
    CHECK_EQ(dy.device()->lang(), kCpp);
    
    CHECK(dy.shape(1) == ch.num_filters_ && dy.shape(2) == ch.conv_height_ &&
    dy.shape(3) == ch.conv_width_) << "input gradients shape should not change";
	
	CHECK(b.shape(0) == ch.num_filters_)<< "bias shape should not change";

    Tensor db;
    db.ResetLike(b);

    auto tmpshp = Shape{ch.batchsize * ch.num_filters_, dy.Size() / (ch.batchsize * ch.num_filters_)};
    Tensor tmp1 = Reshape(dy, tmpshp);

    Tensor tmp2(Shape{ch.batchsize * ch.num_filters_});
    SumColumns(tmp1, &tmp2);
    Tensor tmp3 = Reshape(tmp2, Shape{ch.batchsize, ch.num_filters_});

    SumRows(tmp3, &db);

    return db;
};

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b, const CudnnConvHandles cch){
	CHECK_EQ(x.device()->lang(), kCuda);

    DataType dtype = x.data_type();
    auto dev = x.device();

    Shape shape{cch.batchsize, cch.num_filters_, cch.conv_height_, cch.conv_width_};
    Tensor output(shape, dev, dtype);

    output.device()->Exec([output, x, W, cch](Context *ctx) {
        Block *inblock = x.block(), *outblock = output.block(),
                *wblock = W.block();
        float alpha = 1.f, beta = 0.f;
        cudnnConvolutionForward(ctx->cudnn_handle, &alpha, cch.x_desc_,
                                inblock->data(), cch.filter_desc_, wblock->data(),
                                cch.conv_desc_, cch.fp_alg_,
                                cch.workspace_.block()->mutable_data(),
                                cch.workspace_count_ * sizeof(float), &beta,
                                cch.y_desc_, outblock->mutable_data());
    }, {x.block(), W.block()}, {output.block()}, cch.workspace_.block());

    if (cch.bias_term_) {
        output.device()->Exec([output, b, cch](Context *ctx) {
            float beta = 1.f, alpha = 1.0f;
            Block *outblock = output.block(), *bblock = b.block();
            cudnnAddTensor(ctx->cudnn_handle, &alpha, cch.bias_desc_,
                           bblock->data(), &beta, cch.y_desc_,
                           outblock->mutable_data());
        }, {output.block(), b.block()}, {output.block()});
    }

    return output;
};

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandles cch){
    CHECK_EQ(dy.device()->lang(), kCuda);

    Tensor dx;
    dx.ResetLike(x);

    dy.device()->Exec([dx, dy, W, cch](Context *ctx) {
        Block *wblock = W.block(), *dyblock = dy.block(),
                *dxblock = dx.block();
        float alpha = 1.f, beta = 0.f;
        cudnnConvolutionBackwardData(ctx->cudnn_handle, &alpha, cch.filter_desc_,
                                     wblock->data(), cch.y_desc_, dyblock->data(),
                                     cch.conv_desc_, cch.bp_data_alg_,
                                     cch.workspace_.block()->mutable_data(),
                                     cch.workspace_count_ * sizeof(float), &beta,
                                     cch.x_desc_, dxblock->mutable_data());
    }, {dy.block(), W.block()}, {dx.block(), cch.workspace_.block()});

    return dx;
};

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandles cch){
    CHECK_EQ(dy.device()->lang(), kCuda);

    Tensor dW;
    dW.ResetLike(W);

    dy.device()->Exec([dW, dy, x, W, cch](Context *ctx) {
    Block *inblock = x.block(), *dyblock = dy.block(),
            *dwblock = dW.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardFilter(
            ctx->cudnn_handle, &alpha, cch.x_desc_, inblock->data(),
            cch.y_desc_, dyblock->data(), cch.conv_desc_, cch.bp_filter_alg_,
            cch.workspace_.block()->mutable_data(),
            cch.workspace_count_ * sizeof(float), &beta, cch.filter_desc_,
            dwblock->mutable_data());
    }, {dy.block(), x.block()}, {dW.block(), cch.workspace_.block()});

    return dW;
};

// input Tensor b for Reset db purpose, can avoid this later.
Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandles cch){
    CHECK_EQ(dy.device()->lang(), kCuda);

    Tensor db;
    db.ResetLike(b);

    dy.device()->Exec([db, dy, b, cch](Context *ctx) {
        Block *dyblock = dy.block(), *dbblock = db.block();
        float alpha = 1.f, beta = 0.f;
        cudnnConvolutionBackwardBias(ctx->cudnn_handle, &alpha, cch.y_desc_,
                                     dyblock->data(), &beta, cch.bias_desc_,
                                     dbblock->mutable_data());
    }, {dy.block()}, {db.block()});

    return db;
};

}