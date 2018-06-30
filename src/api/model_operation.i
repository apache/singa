%module model_operation

%{
#include "../src/model/operation/convolution_operation.h"
%}
namespace singa{

struct ConvHandles{

		size_t batchsize;

		ConvHandles(const Tensor &input, const std::vector<size_t> kernel_size, 
                    const std::vector<size_t> stride, const std::vector<size_t> padding,
                    const size_t in_channels, const size_t out_channels,
                    const bool bias_term_);
              	};

struct CudnnConvHandles{

		size_t batchsize;
		
		CudnnConvHandles(const Tensor &input, const std::vector<size_t> kernel_size, 
                    const std::vector<size_t> stride, const std::vector<size_t> padding,
                    const size_t in_channels, const size_t out_channels,
                    const bool bias_term_, const size_t workspace_byte_limit_=1024*1024*1024,
                    const std::string prefer_="fastest");
                };

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b, const CudnnConvHandles cch);

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandles cch);

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandles cch);

Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandles cch);


Tensor CpuConvForward(const Tensor &x, Tensor &W,  Tensor &b, const ConvHandles ch);

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x, const ConvHandles ch);

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const ConvHandles ch);

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b, const ConvHandles ch);

}
