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

#ifndef SINGA_NEURALNET_NEURON_LAYER_DUMMY_H_
#define SINGA_NEURALNET_NEURON_LAYER_DUMMY_H_

#include <random>
#include <vector>
#include "singa/neuralnet/layer.h"
#include "singa/proto/job.pb.h"

namespace singa {
/**
 * This layer is dummy and do no real work.
 * It is used for testing purpose only.
 *
 * Use it as input layer, it will generate random data;
 * Use it as output layer, it will generate random grad;
 * Use it as neuron layer, it will replicates data and grad.
 */
class DummyLayer: public Layer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
 private:
  bool input_ = false;  // use as input layer
  bool output_ = false;  // use as output layer
};

}  // namespace singa

#endif  // SINGA_NEURALNET_NEURON_LAYER_DUMMY_H_
