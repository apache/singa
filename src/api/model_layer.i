/************************************************************
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
*************************************************************/

/*interface file for swig */

%module model_layer
%include "std_vector.i"
%include "std_string.i"
%include "std_pair.i"
%include "std_shared_ptr.i"


%{
// To make the code compatible between py2 and py3, the follow
// macro is required, which forces the
// interface (function) to accept byte string (from python) and
// return byte string (in python) in py3. Otherwise the strings
// should be unicode strings in py3.
// Note that by default the strings in python3 are of type unicode.
// You have to encode it with the correct encoding (default is utf-8)
// to convert it into bytes. Sometimes, the string is already byte string
// e.g. from protobuf SerializeToString, then there is no need to do
// conversion. The output byte strings should be decoded into unicode.
// For python2, the default type of string is byte string.
//
// Because protobuf::SerializeToString cannot be decoded into unicode
// string, we cannot use SWIG_PYTHON_2_UNICODE which forces the
// interface (function) to accept unicode strings as input args
// and return unicode strings.
//
// TODO(wangwei) make strings compatible between py2 and py3.

#define SWIG_PYTHON_STRICT_BYTE_CHAR


#include "singa/model/layer.h"
#include "../src/model/layer/rnn.h"
#include "../src/model/layer/cudnn_rnn.h"
#include "singa/core/tensor.h"
#include "singa/proto/model.pb.h"
#include "singa/singa_config.h"
using singa::Tensor;
using singa::ParamSpec;
using singa::DataType;
using singa::Device;
using singa::LayerConf;
using singa::Shape;
%}

%shared_ptr(singa::Layer)
%shared_ptr(singa::RNN)
#if USE_CUDNN
%shared_ptr(singa::CudnnRNN)
#endif

namespace std {
  %template(VecStr) vector<string>;
  %template(VecParamSpec) vector<singa::ParamSpec>;
  %template(VecTensor) vector<singa::Tensor>;
  %template(VecVecSize) vector<vector<size_t>>;
  %template(PairTensorVecTensor) pair<singa::Tensor, vector<singa::Tensor>>;
  %template(PairVecTensor) pair<vector<singa::Tensor>, vector<singa::Tensor>>;
}

namespace singa {

class Layer {
 public:
  Layer();
  void Setup(const std::vector<size_t>&, const std::string& );
  %rename(SetupWithMultInputs) Setup(const std::vector<std::vector<size_t>>&,
                                     const std::string&);
  void Setup(const std::vector<std::vector<size_t>>&, const std::string&);

  virtual const std::vector<Tensor> param_values();
  virtual const std::vector<size_t> GetOutputSampleShape() const;
  %rename(GetOutputSampleShapeAt) GetOutputSampleShape(int k);
  virtual const std::vector<size_t> GetOutputSampleShape(int k);
  virtual void ToDevice(std::shared_ptr<Device> device);
  virtual void AsType(DataType dtype);

  virtual const Tensor Forward(int flag, const Tensor& input);
  %rename(ForwardWithMultInputs) Forward(int flag, const std::vector<Tensor>&);
  virtual const std::vector<Tensor> Forward(
      int flag, const std::vector<Tensor>& inputs);

  virtual const std::pair<Tensor, std::vector<Tensor>> Backward(
      int flag, const Tensor& grad);
  %rename(BackwardWithMultInputs) Backward(int, const std::vector<Tensor>&);
  virtual const std::pair<std::vector<Tensor>, std::vector<Tensor>>
  Backward(int flag, const std::vector<Tensor>& grads);
};

std::shared_ptr<Layer> CreateLayer(const std::string& type);
const std::vector<std::string> GetRegisteredLayers();

class RNN : public Layer {
};

#if USE_CUDA && USE_CUDNN
#if CUDNN_VERSION >= 5005
class CudnnRNN : public RNN {
 public:
  // note: Must use std::vector instead of vector.
  %rename(ForwardWithMultInputs) Forward(int flag, const std::vector<Tensor>&);
  const std::vector<Tensor> Forward(
      int flag, const std::vector<Tensor>& inputs);
  %rename(BackwardWithMultInputs) Backward(int, const std::vector<Tensor>&);
  const std::pair<std::vector<Tensor>, std::vector<Tensor>>
  Backward(int flag, const std::vector<Tensor>& grads);

  void ToDevice(std::shared_ptr<Device> device) override;
  const std::vector<Tensor> param_values() override;
  const std::vector<size_t> GetOutputSampleShape() const override;
};

#endif  // CUDNN_VERSION >= 5005
#endif  // USE_CUDA && USE_CUDNN
}
