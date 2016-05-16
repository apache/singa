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
#ifndef SINGA_MODEL_LAYER_DROPOUT_H_
#define SINGA_MODEL_LAYER_DROPOUT_H_
#include "singa/model/layer.h"
namespace singa {
class Dropout : public Layer {
 public:
  /// \copydoc Layer::layer_type()
  const std::string layer_type() const override { return "Dropout"; }

  /// \copydoc Layer::Setup(const LayerConf&);
  void Setup(const LayerConf& conf) override;

  /// \copydoc Layer::Forward(int flag, const Tensor&)
  /// if flag is kTrain, then do dropout with given dropout_ratio;
  /// otherwise if it is kEval, copy input directly to the output
  /// TODO(wangwei) There are diff implementations, Caffe vs
  /// <a href="https://github.com/nitishsrivastava/deepnet/blob/master/deepnet/fastdropoutnet.py">
  const Tensor Forward(int flag, const Tensor& input) override;

  /// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
  const std::pair<Tensor, vector<Tensor>> Backward(int flag,
                                                   const Tensor& grad) override;

  void ToDevice(Device* device) override;

 protected:
  /// the proability to set each element to 0.
  float dropout_ratio_;
  Tensor mask_;
};
}  // namespace singa
#endif  // SINGA_MODEL_LAYER_DROPOUT_H_
