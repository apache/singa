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

#ifndef SINGA_NEURALNET_NEURON_LAYER_GRU_H_
#define SINGA_NEURALNET_NEURON_LAYER_GRU_H_

#include <vector>
#include "singa/neuralnet/layer.h"
namespace singa {

class GRULayer : public NeuronLayer {
 public:
  ~GRULayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_z_hx_, weight_z_hh_, bias_z_,
                               weight_r_hx_, weight_r_hh_, bias_r_,
                               weight_c_hx_, weight_c_hh_, bias_c_};
    return params;
  }

  int vdim() const { return vdim_;}
  int hdim() const { return hdim_;}
  int batchsize() const {return batchsize_;}

 private:
  int batchsize_; // batch size
  int vdim_, hdim_; // dimensions

  Param *weight_z_hx_, *weight_z_hh_, *bias_z_; // update gate
  Param *weight_r_hx_, *weight_r_hh_, *bias_r_; // reset gate
  Param *weight_c_hx_, *weight_c_hh_, *bias_c_; // new memory
};

}  // namespace singa

#endif  // SINGA_NEURALNET_NEURON_LAYER_GRU_H_
