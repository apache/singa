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

#ifndef SINGA_NEURALNET_NEURON_LAYER_LRN_H_
#define SINGA_NEURALNET_NEURON_LAYER_LRN_H_

#include <vector>
#include "singa/neuralnet/layer.h"
#include "singa/proto/job.pb.h"

/**
 * \file this file includes the declarations neuron layer classes that conduct
 * the transformation of features.
 */
namespace singa {
/**
 * Local Response Normalization edge
 *
 * b_i=a_i/x_i^beta
 * x_i=knorm+alpha*\sum_{j=max(0,i-n/2}^{min(N,i+n/2}(a_j)^2
 * n is size of local response area.
 * a_i, the activation (after ReLU) of a neuron convolved with the i-th kernel.
 * b_i, the neuron after normalization, N is the total num of kernels
 */
class LRNLayer : public NeuronLayer {
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

 protected:
  //! shape of the bottom layer feature
  int batchsize_, channels_, height_, width_;
  //! size local response (neighbor) area
  int lsize_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
  Blob<float> norm_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_NEURON_LAYER_LRN_H_
