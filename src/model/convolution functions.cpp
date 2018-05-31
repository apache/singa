#include <iostream>
#include <cudnn.h>

struct ConvHandle{
    size_t kernel_w_;
    size_t pad_w_;
    size_t stride_w_;
    size_t kernel_h_;
    size_t pad_h_;
    size_t stride_h_;

    size_t channels_;
    size_t num_filters_;

    bool bias_term_;

    size_t workspace_byte_limit_;
    string prefer_;
};

struct CudnnConvHandle{
    cudnnTensorDescriptor_t x_desc_ ;
    cudnnTensorDescriptor_t y_desc_ ;
    cudnnTensorDescriptor_t bias_desc_ ;
    cudnnFilterDescriptor_t filter_desc_ ;
    cudnnConvolutionDescriptor_t conv_desc_ ;
    cudnnConvolutionFwdAlgo_t fp_alg_;
    cudnnConvolutionBwdFilterAlgo_t bp_filter_alg_;
    cudnnConvolutionBwdDataAlgo_t bp_data_alg_;

    size_t workspace_count_;
    Tensor workspace_;

    size_t height_;
    size_t width_;
    size_t conv_height_;
    size_t conv_width_;
    size_t batchsize;
};

// Done in conv2d.__init__()
ConvHandle SetupConv(const size_t in_channels, const LayerConf &conf){

    size_t kernel_w_, pad_w_, stride_w_;
    size_t kernel_h_, pad_h_, stride_h_;

    size_t channels_, num_filters_;

    bool bias_term_;

    size_t workspace_byte_limit_;
    string prefer_;

    ConvolutionConf conv_conf = conf.convolution_conf();

    workspace_byte_limit_ = conv_conf.workspace_byte_limit() << 20;
    prefer_ = ToLowerCase(conv_conf.prefer());
    CHECK(prefer_ == "fastest" || prefer_ == "limited_workspace" ||
          prefer_ == "no_workspace" || prefer_ == "autotune")
            << "CudnnConvolution only supports four algorithm preferences: fastest, "
               "limited_workspace, no_workspace and autotune";

    // store intermediate data, i.e., input tensor
    //std::stack<Tensor> buf_;

    // kernel_size, pad, and stride are repeated fields.
    if (conv_conf.kernel_size_size() > 0) {
    if (conv_conf.kernel_size_size() == 1) {
    kernel_w_ = kernel_h_ = conv_conf.kernel_size(0);
    } else {
    kernel_w_ = conv_conf.kernel_size(0);
    kernel_h_ = conv_conf.kernel_size(1);
    }
    } else {
    kernel_w_ = conv_conf.kernel_w();
    kernel_h_ = conv_conf.kernel_h();
    }
    CHECK_GT(kernel_w_, 0u);
    CHECK_GT(kernel_h_, 0u);

    if (conv_conf.pad_size() > 0) {
    if (conv_conf.pad_size() == 1) {
    pad_w_ = pad_h_ = conv_conf.pad(0);
    } else {
    pad_w_ = conv_conf.pad(0);
    pad_h_ = conv_conf.pad(1);
    }
    } else {
    pad_w_ = conv_conf.pad_w();
    pad_h_ = conv_conf.pad_h();
    }
    CHECK_GE(pad_w_, 0u);
    CHECK_GE(pad_h_, 0u);

    const int kStrideDefault = 1;
    if (conv_conf.stride_size() > 0) {
    if (conv_conf.stride_size() == 1) {
    stride_w_ = stride_h_ = conv_conf.stride(0);
    } else {
    stride_w_ = conv_conf.stride(0);
    stride_h_ = conv_conf.stride(1);
    }
    } else {
    stride_w_ = kStrideDefault;
    stride_h_ = kStrideDefault;
    if (conv_conf.has_stride_w()) {
    stride_w_ = conv_conf.stride_w();
    }
    if (conv_conf.has_stride_h()) {
    stride_h_ = conv_conf.stride_h();
    }
    }
    CHECK_GT(stride_w_, 0u);
    CHECK_GE(stride_h_, 0u);  // 0 for 1D conv

    channels_ = in_channels;
    num_filters_ = conv_conf.num_output();
    bias_term_ = conv_conf.bias_term();

    return ConvHandle{
            kernel_w_,
            pad_w_,
            stride_w_,
            kernel_h_,
            pad_h_,
            stride_h_,

            channels_,
            num_filters_,

            bias_term_,

            workspace_byte_limit_,
            prefer_,
    };
}



// Done in conv2d.__call__():
// if self.cudnnconvhandle is None:
//     self.cudnnconvhandle= InitCudnn(...)
// elif x.shape(0) != self.cudnnconvhandle.batchsize:
//     self.cudnnconvhandle= InitCudnn(...)
CudnnConvHandle InitCudnn(const Tensor &input, const ConvHandle ch){

    cudnnTensorDescriptor_t x_desc_ = nullptr;
    cudnnTensorDescriptor_t y_desc_ = nullptr;
    cudnnTensorDescriptor_t bias_desc_ = nullptr;
    cudnnFilterDescriptor_t filter_desc_ = nullptr;
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
    cudnnConvolutionFwdAlgo_t fp_alg_;
    cudnnConvolutionBwdFilterAlgo_t bp_filter_alg_;
    cudnnConvolutionBwdDataAlgo_t bp_data_alg_;
    size_t workspace_count_;
    Tensor workspace_;

    size_t height_;
    size_t width_;
    size_t conv_height_;
    size_t conv_width_;

    DataType dtype = input.data_type();
    auto dev = input.device();
    Context *ctx = dev->context(0);

    size_t batchsize, channels_;
    batchsize = input.shape(0);
    channels_ = input.shape(1);
    height_ = input.shape(2);
    width_ = input.shape(3);

    CHECK(channels_ == ch.channels_)<<"the number of input channels mismatched.";

    conv_height_ = 1;
    if (ch.stride_h_ > 0)
        conv_height_ = (height_ + 2 * ch.pad_h_ - ch.kernel_h_) / ch.stride_h_ + 1;
    conv_width_ = (width_ + 2 * ch.pad_w_ - ch.kernel_w_) / ch.stride_w_ + 1;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    if (ch.bias_term_)
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));


    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW,
                                           GetCudnnDataType(dtype), batchsize,
                                           ch.channels_, height_, width_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            y_desc_, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize,
            ch.num_filters_, conv_height_, conv_width_));
    if (ch.bias_term_)
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW,
                                               GetCudnnDataType(dtype), 1,
                                               ch.num_filters_, 1, 1));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_, ch.pad_h_, ch.pad_w_,
                                                ch.stride_h_, ch.stride_w_, 1, 1,
                                                CUDNN_CROSS_CORRELATION,
                                                GetCudnnDataType(dtype)));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, GetCudnnDataType(dtype),
                                           CUDNN_TENSOR_NCHW, ch.num_filters_,
                                           channels_, ch.kernel_h_, ch.kernel_w_));
    if (ch.prefer_ == "fastest" || ch.prefer_ == "limited_workspace" ||
        ch.prefer_ == "no_workspace") {
        cudnnConvolutionFwdPreference_t fwd_pref;
        cudnnConvolutionBwdFilterPreference_t bwd_filt_pref;
        cudnnConvolutionBwdDataPreference_t bwd_data_pref;
        if (ch.prefer_ == "fastest") {
            fwd_pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
            bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
            bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
        } else if (ch.prefer_ == "limited_workspace") {
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
                ch.workspace_byte_limit_, &fp_alg_));
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
                ctx->cudnn_handle, x_desc_, y_desc_, conv_desc_, filter_desc_,
                bwd_filt_pref, ch.workspace_byte_limit_, &bp_filter_alg_));
        // deprecated in cudnn v7
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
                ctx->cudnn_handle, filter_desc_, y_desc_, conv_desc_, x_desc_,
                bwd_data_pref, ch.workspace_byte_limit_, &bp_data_alg_));
    } else if (ch.prefer_ == "autotune") {
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
    if (workspace_count_ * sizeof(float) > ch.workspace_byte_limit_)
        LOG(WARNING) << "The required memory for workspace ("
                     << workspace_count_ * sizeof(float)
                     << ") is larger than the expected Bytes ("
                     << ch.workspace_byte_limit_ << ")";
    workspace_ = Tensor(Shape{workspace_count_}, dev, dtype);

    return CudnnConvHandle{
            x_desc_,
            y_desc_,
            bias_desc_,
            filter_desc_,
            conv_desc_,
            fp_alg_,
            bp_filter_alg_,
            bp_data_alg_,

            workspace_count_,
            workspace_,

            height_,
            width_,
            conv_height_,
            conv_width_,
            batchsize,
    };

}

Tensor CudnnConvForward(Tensor x, Tensor W, Tensor b, const ConvHandle ch, const CudnnConvHandle cch){
    CHECK_EQ(x.device()->lang(), kCuda);
    CHECK_EQ(x.nDim(), 4u);
    CHECK_EQ(x.shape()[0],cch.batchsize);
    CHECK_EQ(x.shape()[1],ch.channels_);
    CHECK_EQ(x.shape()[2],cch.height_);
    CHECK_EQ(x.shape()[3],cch.width_);

    DataType dtype = x.data_type();
    auto dev = x.device();

    Shape shape{cch.batchsize, ch.num_filters_, cch.conv_height_, cch.conv_width_};
    Tensor output(shape, dev, dtype);

    output.device()->Exec([x, output](Context *ctx) {
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

    if (ch.bias_term_) {
        output.device()->Exec([output](Context *ctx) {
            float beta = 1.f, alpha = 1.0f;
            Block *outblock = output.block(), *bblock = b.block();
            cudnnAddTensor(ctx->cudnn_handle, &alpha, cch.bias_desc_,
                           bblock->data(), &beta, cch.y_desc_,
                           outblock->mutable_data());
        }, {output.block(), b.block()}, {output.block()});
    }
    return output;
}

// input Tensor W for Reset dW purpose, can avoid this later.
Tensor CudnnConvBackwardW(Tensor dy, Tensor x, Tensor W, CudnnConvHandle cch){
    CHECK_EQ(dy.device()->lang(), kCuda);
    CHECK_EQ(dy.nDim(), 4u);

    Tensor dW;
    dW.ResetLike(W);

    dy.device()->Exec([dy, dW, x](Context *ctx) {
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
}

// input Tensor b for Reset db purpose, can avoid this later.
Tensor CudnnConvBackwardb(Tensor dy, Tensor b, CudnnConvHandle cch){
    CHECK_EQ(dy.device()->lang(), kCuda);
    CHECK_EQ(dy.nDim(), 4u);

    Tensor db;
    db.ResetLike(b);

    dy.device()->Exec([dy, db](Context *ctx) {
        Block *dyblock = dy.block(), *dbblock = db.block();
        float alpha = 1.f, beta = 0.f;
        cudnnConvolutionBackwardBias(ctx->cudnn_handle, &alpha, cch.y_desc_,
                                     dyblock->data(), &beta, cch.bias_desc_,
                                     dbblock->mutable_data());
    }, {dy.block()}, {db.block()});
    return db;
}

// input Tensor x for Reset dx purpose, can avoid this later.
Tensor CudnnConvBackwardx(Tensor dy, Tensor W, Tensor x, CudnnConvHandle cch){
    CHECK_EQ(dy.device()->lang(), kCuda);
    CHECK_EQ(dy.nDim(), 4u);

    Tensor dx;
    dx.ResetLike(x);

    dy.device()->Exec([dx, dy](Context *ctx) {
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
}

