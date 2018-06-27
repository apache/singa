%module model_operation

%{
#include "../src/model/operation/convolution_related.h"
%}
namespace singa{

struct Recorder{size_t batchsize;};

struct CudnnConvHandles{};


Recorder SetupRecorder(const Tensor &input, const std::vector<size_t> kernel_size, 
	                const std::vector<size_t> stride, const std::vector<size_t> padding,
	                const size_t in_channels, const size_t out_channels,
	                const bool bias_term_);

CudnnConvHandles InitCudnnConvHandles(const Tensor &input, const Recorder r, 
     const size_t workspace_byte_limit_=1024*1024*1024, const std::string prefer_="fastest");

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b, const Recorder r, const CudnnConvHandles cch);

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandles cch);

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandles cch);

Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandles cch);


Tensor CpuConvForward(const Tensor &x, Tensor &W,  Tensor &b, const Recorder r);

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x, const Recorder r);

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const Recorder r);

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b, const Recorder r);

}
