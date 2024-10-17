/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SINGA_MODEL_LAYER_H_
#define SINGA_MODEL_LAYER_H_

#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "singa/core/tensor.h"
#include "singa/proto/model.pb.h"
#include "singa/utils/factory.h"

namespace singa {

typedef vector<size_t> Shape;
/// The base layer class.
/// Generally, a layer conducts feature transformation against a set of Tensor
/// to generate a set of Tensor. Each layer may have some parameters.
class Layer {
 public:
  Layer() = default;

  /// Set meta data fields from a string representing a proto message.
  /// 'in_shape' is the shape of the input feature for one sample
  void Setup(const Shape& in_shape, const string& proto_str) {
    LayerConf conf;
    conf.ParseFromString(proto_str);
    this->Setup(in_shape, conf);
  }

  /// 'in_shapes' is the shape of the input feature for one sample
  void Setup(const vector<Shape>& in_shapes, const string& proto_str) {
    LayerConf conf;
    conf.ParseFromString(proto_str);
    this->Setup(in_shapes, conf);
  }

  // ============= Following Functions could be override =====================
  /// Destruct objects created by this layer.
  virtual ~Layer(){};

  /// Each layer sub-class would optionaly have a type name.
  /// Used for debugging and logging.
  virtual const std::string layer_type() const { return "Unknown"; }

  /// Set meta data fields configured in 'conf' (a proto message).
  /// Some layers would use input tensor shapes for setting its parameter
  /// shapes (e.g, desen layer and convolution layer). 'in_shape' provides such
  /// shape info. It represents the shape of the Tensor (with a single sample)
  /// from the last layer.
  /// After calling Setup, the shape info of parameters should be accssed
  /// correctly. Internal buffer/fields are set assuming batchsize is 1.
  virtual void Setup(const Shape& in_sample, const LayerConf& conf) {
    name_ = conf.name();
    // TODO(wangwei) load param values from checkpoint files.
  }

  /// Used for layers that have multiple input tensors, e.g., concatenate layer.
  virtual void Setup(const vector<Shape>& in_samples, const LayerConf& conf) {
    name_ = conf.name();
    // TODO(wangwei) load param values from checkpoint files.
  }

  /// Return the shape of the generated Tensor without the batchsize dimension
  virtual const Shape GetOutputSampleShape() const {
    LOG(FATAL) << "Pls override this function";
    return vector<size_t>{};
  }
  /// Return the shape of the k-th generated tensor without the batchsize
  /// dimension. Used for layers that generate multiple tensors.
  virtual const Shape GetOutputSampleShape(int k) {
    LOG(FATAL) << "Pls override this function";
    return vector<size_t>{};
  }

  /// Do feature transformation for the given 'input' tensor (denoted as x).
  /// 'flag' is either kTrain or kEval for feed-forward nets, and
  /// would be used for other phases of training other nets. For example, when
  /// training RBM, we may create an alias of this function as ComputeFeature
  /// where flag could be kPositive and kNegative.
  /// It will return a Tensor (denoted as y).
  /// If the 'input' or 'output' is required for computing the gradients in
  /// Backward(), then buffer them as internal data.
  virtual const Tensor Forward(int flag, const Tensor& input) {
    LOG(FATAL) << "Not implemented";
    Tensor t;
    return t;
  }

  /// \copydoc Forward(int flag, const Tensor& input)
  /// Accept multiple input tensors and generate multiple output tensors.
  /// If there is only one input tensor, it will call Forward(int, const
  /// Tensor&) by default. Users can override this function for layers who
  /// generate more than one outputs.
  virtual const vector<Tensor> Forward(int flag, const vector<Tensor>& inputs) {
    vector<Tensor> ret;
    if (inputs.size() == 1) ret.push_back(Forward(flag, inputs.at(0)));

    LOG(FATAL) << "Not implemented";
    return ret;
  }

  /// Compute gradients of this layer.
  /// Specifically, there are two types of gradients:
  /// 1. gradient of the preceding layer, i.e., dx.
  /// 2. gradients of parameters of this layer, e.g., dw for weight matrix.
  /// 1 is an empty tensor if there is no preceding layer or there is no need to
  /// compute dx (e.g., x is from a data layer); 2 is an empty vector if this
  // layer has no parameters.
  /// 'flag' is either kTrain or kEval for feed-forward nets, and
  /// would be used for other phases when training other nets.
  /// 'grad' is a Tensor for gradient (dy) from the upper layer.
  virtual const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                           const Tensor& grad) {
    LOG(FATAL) << "Not implemented!";
    Tensor t;
    return std::make_pair(t, vector<Tensor>{});
  }

  /// \copydoc Backward(int, const vector<Tensor>&)
  /// For Forward(int, const vector<Tensor>&)
  virtual const std::pair<vector<Tensor>, vector<Tensor>> Backward(
      int flag, const vector<Tensor>& grads) {
    vector<Tensor> input_grad, param_grad;
    if (grads.size() == 1u) {
      auto ret = Backward(flag, grads.at(0));
      input_grad.push_back(ret.first);
      param_grad = ret.second;
    } else {
      LOG(FATAL) << "Not implemented";
    }
    return std::make_pair(input_grad, param_grad);
  }

  /// Clone the layer to the given device. Layer data (e.g., parameters) are
  /// deep copied. If 'device' is nullptr, then clone it one the current device.
  // virtual Layer* Clone(std::shared_ptr<Device> device);
  /// Move the layer (including its parameters and other internal Tensor) onto
  /// the given device
  virtual void ToDevice(std::shared_ptr<Device> device) {}

  /// Set the data type of Tensor in this layer.
  virtual void AsType(DataType dtype) {}

  /// Serialize the layer info (including params) into a LayerConf proto message
  virtual void ToProto(LayerConf* conf) const {
    // conf->set_name(name_);
    // for (const auto& spec : param_specs_) {
    //   ParamSpec* p = conf->add_param();
    //   p->CopyFrom(spec);
    // }
    //  TODO(wangwei) add param values into conf;
  }

  // ========================================================================

  /// Serialize the layer info, including params_, into a string representing
  /// a LayerParameter message.
  std::string ToProtoStr() const {
    LayerConf conf;
    ToProto(&conf);
    string str;
    conf.SerializeToString(&str);
    return str;
  }
  /// Return specs/configuration of all parameter instances of this layer.
  /// \ref ParamSpec.
  const vector<ParamSpec> param_specs() { return param_specs_; }

  /// Return the i-th ParamSpec.
  const ParamSpec& param_specs(size_t i) {
    CHECK_LT(i, param_specs_.size());
    return param_specs_.at(i);
  }

  /// Return pointers to parameter Tensor s.
  virtual const vector<Tensor> param_values() { return vector<Tensor>{}; }

  /// Return names of all parmaeters.
  const vector<string> param_names() {
    vector<string> pname;
    for (const auto& spec : param_specs_) pname.push_back(spec.name());
    return pname;
  }

  /// Return the 'i'-th parameter name.
  const string& param_name(size_t i) {
    CHECK_LT(i, param_specs_.size());
    return param_specs_.at(i).name();
  }

  /// Each layer instance would optionally have a name.
  /// Used for debugging and logging.
  const std::string name() const { return name_; }

 protected:
  std::string name_;
  vector<ParamSpec> param_specs_;
};

/// Name should be formated as cudnn_xxx, singacpp_xxx, singacuda_xxx,
/// singacl_xxx, where xxx is the real layer type, e.g., convolution, relu, etc.
/// xxx should only have lower case letters.
/// if the implmentation is transparent to cpp/cuda/opencl, then register all
/// possible identifiers. For instance, Dropout is registered three times,
/// RegisterLayerClass("singacpp_dropout", Dropout)
/// RegisterLayerClass("singacl_dropout", Dropout)
/// RegisterLayerClass("singacuda_dropout", Dropout)
/// to be compatible with previous commits, the following identifier is
/// registered. Better avoid using it, as it would be deprecated.
/// RegisterLayerClass("singa_dropout", Dropout)
#define RegisterLayerClass(Name, SubLayer) \
  static Registra<Layer, SubLayer> Name##SubLayer(#Name);

inline std::shared_ptr<Layer> CreateLayer(const std::string type) {
  std::shared_ptr<Layer> layer(Factory<Layer>::Create(type));
  return layer;
}

inline const std::vector<std::string> GetRegisteredLayers() {
  vector<std::string> ret;
  for (const string type : Factory<Layer>::GetIDs()) {
    auto layer = CreateLayer(type);
    ret.push_back("Register type: " + type);
  }
  return ret;
}
}  // namespace singa
#endif  // SINGA_MODEL_LAYER_H_
