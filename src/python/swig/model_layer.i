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
#include "singa/model/layer.h"
#include "singa/core/tensor.h"
#include "singa/proto/model.pb.h"
using singa::Tensor;
using singa::ParamSpec;
using singa::DataType;
using singa::Device;
using singa::LayerConf;
%}

%shared_ptr(singa::Layer)

namespace std {
  %template(strVector) vector<string>;
  %template(paramVector) vector<singa::ParamSpec>;
  %template(tensorVector) vector<singa::Tensor>;
  %template(ttvecPair) pair<singa::Tensor, vector<singa::Tensor>>;
  %template(tvecPair) pair<vector<singa::Tensor>, vector<singa::Tensor>>;
}


namespace singa {

  class Layer {
    public:
      Layer();
//      virtual void Setup(const std::vector<vector<size_t>>&, const string&);
      virtual void Setup(const std::vector<size_t>& in_sample_shape,
                         const std::string& proto_str);
      const std::vector<Tensor> param_values();
      virtual const std::vector<size_t> GetOutputSampleShape() const;
      virtual void ToDevice(std::shared_ptr<Device> device);
      virtual void AsType(DataType dtype);
      virtual const Tensor Forward(int flag, const Tensor& input);
      virtual const std::vector<Tensor> Forward(
          int flag, const std::vector<Tensor>& inputs);
      virtual const std::pair<Tensor, std::vector<Tensor>> Backward(
          int flag, const Tensor& grad);
      virtual const std::pair<std::vector<Tensor>, std::vector<Tensor>>
      Backward(int flag, const vector<Tensor>& grads);

  };
  std::shared_ptr<Layer> CreateLayer(const std::string& type);
  const std::vector<std::string> GetRegisteredLayers();
}

