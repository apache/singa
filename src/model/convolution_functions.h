#include <string>
#include <cudnn.h>
#include "./layer/cudnn_convolution.h"
#include "./layer/cudnn_utils.h"
#include "singa/utils/logging.h"

namespace singa{

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

ConvHandle SetupConv(
    const size_t kernel_h_, const size_t kernel_w_,
    const size_t pad_h_, const size_t pad_w_,
    const size_t stride_h_,const size_t stride_w_,
    const size_t channels_, const size_t num_filters_,
    const bool bias_term_ = true ,const size_t workspace_byte_limit_=1024*1024,
    const std::string prefer_="fastest");

void testInitCudnn(const Tensor &input, const ConvHandle ch);

CudnnConvHandle InitCudnn(const Tensor &input, const ConvHandle ch);

Tensor CudnnConvForward(const Tensor &x, const Tensor &W, const Tensor &b,
                        const ConvHandle ch, const CudnnConvHandle cch);

Tensor CudnnConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandle cch);

Tensor CudnnConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandle cch);

Tensor CudnnConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandle cch);

}
