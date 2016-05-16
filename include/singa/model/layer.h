/**
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

#ifndef SINGA_LAYER_H_
#define SINGA_LAYER_H_

#include <vector>
#include <string>
#include <stack>
#include "singa/core/tensor.h"
#include "singa/proto/layer.pb.h"

namespace singa {

/// The base layer class.
/// Generally, a layer conducts feature transformation against a set of Tensor
/// to generate a set of Tensor. Each layer may have some parameters.
class Layer {
 public:
  Layer() = default;

  /// Set meta data fields from a string representing a proto message.
  void Setup(const string& proto_str) {
    LayerConf conf;
    conf.ParseFromString(proto_str);
    this->Setup(conf);
  }

  // ============= Following Functions could be override =====================
  /// Destruct the objecst created by this layer.
  virtual ~Layer() {
    for (Tensor * t : param_values_) {
      delete t;
    }
  }

  /// Each layer sub-class would optionaly have a type name.
  /// Used for debugging and logging.
  virtual const std::string layer_type() const { return "Unknown"; }

  /// Set meta data fields configured in 'conf' (a proto message).
  virtual void Setup(const LayerConf& conf) {
    name_ = conf.name();
    for (const auto& spec : conf.param())
      param_specs_.push_back(spec);
    // TODO(wangwei) load param values from checkpoint blobs.
  }

  /// Do feature transformation for the given 'input' tensor (denoted as x).
  /// 'flag' is either kPhaseTrain or kPhaseTest for feed-forward nets, and
  /// would be used for other phases of training other nets. For example, when
  /// training RBM, we may create an alias of this function as ComputeFeature
  /// where flag could be kPositivePhase and kNegativePhase.
  /// It will return a Tensor (denoted as y).
  /// If the 'input' or 'output' is required for computing the gradients in
  /// Backward(), then push them into the states_ stack.
  virtual const Tensor Forward(int flag, const Tensor& input) {
    LOG(FATAL) << "Not implemented";
    Tensor t;
    return t;
  }

  /// \copydoc Forward(int flag, const Tensor& input)
  /// Accept multiple input tensors and generate multiple output tensors.
  virtual const vector<Tensor> Forward(int flag, const vector<Tensor>& inputs) {
    vector<Tensor> ret;
    if (inputs.size() == 1)
      ret.push_back(Forward(flag, inputs.at(0)));

    LOG(FATAL) << "Not implemented";
    return ret;
  }

  /// Compute gradients of this layer.
  /// Specifically, there are two types of gradients:
  /// 1. gradients of preceding layers, i.e., dx.
  /// 2. gradients of parameters of this layer.
  /// 1 and 2 are returned as a pair of vector<Tensor>
  /// 1 is an empty tensor if there is no preceding layer or there is no need to
  /// compute dx (e.g., x is from a data layer); 2 is empty if this layer has no
  /// parameters.
  /// 'flag' is either kTrainPhase or kTestPhase for feed-forward nets, and
  /// would be used for other phases when training other nets.
  /// 'grad' is a Tensor for gradient (dy) from the upper layer.
  /// Some layer would use 'input' or 'output' from Forward to compute the
  /// gradients of parameters. Backward() pop out the state data.
  /// It is useful for RNN layers, where the same layer is used multiple
  /// times just like unrolling the layer.
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
    } else  {
      LOG(FATAL) << "Not implemented";
    }
    return std::make_pair(input_grad, param_grad);
  }

  /// Move the layer (including its parameters and other internal Tensor) onto
  /// the given device
  virtual void ToDevice(Device* device) {
    for (auto p : param_values_) p->ToDevice(device);
  }

  /// Set the data type of Tensor in this layer.
  virtual void AsType(DataType dtype) {
    for (auto p : param_values_) p->AsType(dtype);
  }

  /// Serialize the layer info (including params) into a LayerConf proto message
  virtual void ToProto(LayerConf* conf) const {
    conf->set_name(name_);
    for (const auto& spec: param_specs_) {
      ParamSpec* p = conf->add_param();
      p->CopyFrom(spec);
    }
    // TODO(wangwei) add param values into conf;
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
  const vector<ParamSpec> param_specs() {
    return param_specs_;
  }

  /// Return the i-th ParamSpec.
  const ParamSpec& param_specs(int i) {
    return param_specs_.at(i);
  }

  /// Return pointers to parameter Tensor s.
  const vector<Tensor*> param_values() {
    return param_values_;
  }

  /// Return a pointer to the 'i'-th parameter Tensor.
  Tensor* param_value(size_t i) {
    CHECK_LT(i, param_values_.size());
    return param_values_[i];
  }

  /// Return names of all parmaeters.
  const vector<string> param_names() {
    vector<string> pname;
    for (const auto& spec: param_specs_)
      pname.push_back(spec.name());
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

  /*
  std::stack<Tensor> states() const {
    return states_;
  }
  */

 protected:
  std::string name_;
  vector<Tensor*> param_values_;
  vector<ParamSpec> param_specs_;
  /// Used to store input or output of Forward(), which would be used in
  /// Backward.  Rules:
  /// 1. push the 'input' or 'output' into states_ if the flag of Forward() is
  ///    for training.
  /// 2. pop data out in Backward().
  /// TODO(wangwei) enable this feature for rnn layers.
  // std::stack<Tensor*> states_;
};

// ===========================================================================
// Order layer sub-classes based on alphabetical order of the first letter.
// ===========================================================================


}  // namespace singa
#endif  // SINGA_LAYER_H_
