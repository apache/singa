#include <string>
#include <vector>
#include <cudnn.h>
#include "../layer/cudnn_convolution.h"
#include "../layer/cudnn_utils.h"
#include "singa/utils/logging.h"

namespace singa{

struct ConvHandles{
    size_t kernel_w_;
    size_t pad_w_;
    size_t stride_w_;
    size_t kernel_h_;
    size_t pad_h_;
    size_t stride_h_;

    size_t channels_;
    size_t num_filters_;

    bool bias_term_;

    size_t height_;
    size_t width_;
    size_t conv_height_;
    size_t conv_width_;
    size_t batchsize;

    size_t col_height_;
    size_t col_width_;
    size_t imagesize;

    ConvHandles(const Tensor &input, const std::vector<size_t> kernel_size, 
                    const std::vector<size_t> stride, const std::vector<size_t> padding,
                    const size_t in_channels, const size_t out_channels,
                    const bool bias_term_);

};

struct CudnnConvHandles:ConvHandles{
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

    CudnnConvHandles(const Tensor &input, const std::vector<size_t> kernel_size, 
                    const std::vector<size_t> stride, const std::vector<size_t> padding,
                    const size_t in_channels, const size_t out_channels,
                    const bool bias_term_, const size_t workspace_byte_limit_=1024*1024*1024,
                    const std::string prefer_="fastest");
};

Tensor CpuConvForward(const Tensor &x, Tensor &W,  Tensor &b, const ConvHandles ch);

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x, const ConvHandles ch);

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const ConvHandles ch);

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b, const ConvHandles ch);


Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b, const CudnnConvHandles cch);

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandles cch);

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandles cch);

Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandles cch);


}