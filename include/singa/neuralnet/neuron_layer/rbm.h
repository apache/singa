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

#ifndef SINGA_NEURALNET_NEURON_LAYER_RBM_H_
#define SINGA_NEURALNET_NEURON_LAYER_RBM_H_

#include <vector>
#include "singa/neuralnet/layer.h"
#include "singa/proto/job.pb.h"

namespace singa {
/**
 * Base layer for RBM models.
 */
class RBMLayer: virtual public Layer {
 public:
  virtual ~RBMLayer() {}
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_, bias_};
    return params;
  }
  virtual Blob<float>* Sample(int flat);

 protected:
  //! if ture, sampling according to guassian distribution
  bool gaussian_;
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int batchsize_;
  bool first_gibbs_;
  Param* weight_, *bias_;
  Blob<float> pos_data_;
  Blob<float> neg_data_;
  Blob<float> neg_sample_;
  Blob<float> pos_sample_;
};

/**
 * RBM visible layer
 */
class RBMVisLayer: public RBMLayer, public LossLayer {
 public:
  ~RBMVisLayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

 private:
  RBMLayer* hid_layer_;
  Layer* input_layer_;
};
/**
 * RBM hidden layer
 */
class RBMHidLayer: public RBMLayer {
 public:
  ~RBMHidLayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

 private:
  RBMLayer *vis_layer_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_NEURON_LAYER_RBM_H_
