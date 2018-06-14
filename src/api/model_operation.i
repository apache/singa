%module model_operation

%{
#include "../src/model/convolution_functions.h"
using singa::Tensor;
using singa::CudnnConvHandle;
%}
namespace singa{

struct ConvHandle{};

struct CudnnConvHandle{};

ConvHandle SetupConv(const size_t in_channels, const LayerConf &conf);

CudnnConvHandle InitCudnn(const Tensor &input, const ConvHandle ch);

Tensor CudnnConvForward(const Tensor &x, const Tensor &W, const Tensor &b,
                        const ConvHandle ch, const CudnnConvHandle cch);

Tensor CudnnConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W, const CudnnConvHandle cch);

Tensor CudnnConvBackwardb(const Tensor &dy, const Tensor &b, const CudnnConvHandle cch);

Tensor CudnnConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x, const CudnnConvHandle cch);
}
