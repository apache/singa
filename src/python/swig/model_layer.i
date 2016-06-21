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

namespace std {
  %template(strVector) vector<string>;
  %template(paramVector) vector<ParamSpec>;
  %template(tensorVector) vector<Tensor>;
  %template(tensorPtrVector) vector<Tensor*>;
  %template(ttvecPair) pair<Tensor, vector<Tensor>>;
  %template(tvecPair) pair<vector<Tensor>, vector<Tensor>>;
}


namespace singa {

  class Layer {
    public:
      Layer();
      void Setup(const std::vector<size_t>&, const string&);
      void Setup(const std::vector<vector<size_t>>&, const string&);

      std::string ToProtoStr() const;
      const std::vector<ParamSpec> param_specs();
      const ParamSpec& param_specs(size_t i);
      const std::vector<Tensor*> param_values();
      Tensor* param_value(size_t i);
      const std::vector<std::string> param_names();
      const std::string& param_name(size_t i);
      const std::string name() const;

      /* virtual functions */
      virtual const std::string layer_type() const;
      virtual void Setup(const std::vector<size_t>&,
                         const LayerConf&);
      virtual void Setup(const std::vector<std::vector<size_t>>&,
                         const LayerConf&);
      virtual void ToDevice(std::shared_ptr<Device> device);
      virtual void AsType(DataType dtype);
      virtual void ToProto(LayerConf* conf) const;

      virtual const Tensor
      Forward(int flag, const Tensor& input);
      virtual const std::vector<Tensor>
      Forward(int flag, const std::vector<Tensor>& inputs);
      virtual const std::pair<Tensor, std::vector<Tensor>>
      Backward(int flag, const Tensor& grad);
      virtual const std::pair<std::vector<Tensor>, std::vector<Tensor>>
      Backward(int flag, const vector<Tensor>& grads);
  };
}

