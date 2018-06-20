%module model_operation

%{
#include "../src/model/convolution_functions.h"
%}
namespace singa{

struct ConvHandle{};

struct CudnnConvHandle{size_t batchsize;};

ConvHandle SetupConv(
    const size_t kernel_h_, const size_t kernel_w_,
    const size_t pad_h_, const size_t pad_w_,
    const size_t stride_h_,const size_t stride_w_,
    const size_t channels_, const size_t num_filters_,
    const bool bias_term_ = true, const size_t workspace_byte_limit_ =1024*1024*1024,
    const std::string prefer_="fastest");

CudnnConvHandle InitCudnn(const Tensor &input, const ConvHandle ch);

Tensor CudnnConvForward(const Tensor &x, const Tensor &W, const Tensor &b,
                        const ConvHandle ch, const CudnnConvHandle cch);

Tensor CudnnConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandle cch);

Tensor CudnnConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandle cch);

Tensor CudnnConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandle cch);

}
