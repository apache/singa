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
#include "singa/core/tensor.h"
#include "singa/proto/layer.pb.h"

namespace singa {

/// The base layer class.
/// Generally, a layer conducts feature transformation against a set of Tensor
/// to generate a set of Tensor. Each layer may have some parameters represented
/// by Param instances.
class Layer {
 public:
  Layer() = default;
  /// Each layer sub-class would optionaly have a type name.
  /// Used for debugging and logging.
  virtual const std::string layer_type() const { return "Unknown"; }

  /// Set meta data fields from a string representing a proto message.
  void Setup(const string& proto_str) {
    LayerConf conf;
    conf.ParseFromString(proto_str);
    this->Setup(conf);
  }

  /// Set meta data fields configured in 'conf' (a proto message).
  virtual void Setup(const LayerConf& conf) {}

  /// Do feature transformation for given 'input' Tensor.
  /// It is the forward pass for feed-forward nets and rnn nets.
  /// 'flag' is either kPhaseTrain or kPhaseTest for feed-forward nets, and
  /// would be used for phases of training other nets.
  /// It will return a set of Tensor.
  virtual const vector<Tensor> ComputeFeature(int flag,
                                              const vector<Tensor>& input) {
    return vector<Tensor>{};
  }
  /// Compute gradients of parameters of this layer.
  /// It would also compute the gradients for other layers, e.g., the
  /// preceding layers in topology order. It would return an empty vector if
  /// this layer does not need to compute gradients for other layers.
  /// 'flag' is either kPhaseTrain or kPhaseTest for feed-forward nets, and
  /// would be used for phases of training other nets.
  /// 'input' is a vector of Tensor for gradients from other layers.
  virtual const vector<Tensor> ComputeGradient(int flag,
                                               const vector<Tensor>& input) {
    return vector<Tensor>{};
  }

  /// Move the layer (including its parameters and other Tensor) onto the given
  /// device
  virtual void ToDevice(Device* device) {
    // for (auto p : params_)
      // p->ToDevice(device);
  }

  /// Set the data type of Tensor s in this layer.
  virtual void AsType(DataType dtype) {
  //     for (auto p : params_)
  //     p->AsType(dtype);
  }

  /// Serialize the layer info, including params)_, into a LayerConf message.
  virtual std::string ToProto(LayerConf* param) const = 0;

  /// Serialize the layer info, including params_, into a string representing
  /// a LayerParameter message.
  /*
  std::string ToProtoStr() const {
    std:: string str;
    SerializeToString(&str);
  }
  */

  /// Return all Param instances of this layer.
  const vector<void*> params() const { return params_; }

  /// Each layer instance would optionally have a name.
  /// Used for debugging and logging.
  const std::string name() const { return name_; }


 protected:
  std::string name_;
  std::vector<void*> params_;
};

}  // namespace singa
#endif  // SINGA_LAYER_H_
