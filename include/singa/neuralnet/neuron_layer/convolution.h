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

#ifndef SINGA_NEURALNET_NEURON_LAYER_CONVOLUTION_H_
#define SINGA_NEURALNET_NEURON_LAYER_CONVOLUTION_H_

#include <vector>
#include "singa/neuralnet/layer.h"
#include "singa/proto/job.pb.h"

namespace singa {
/**
 * Convolution layer.
 */
class ConvolutionLayer : public NeuronLayer {
 public:
  ~ConvolutionLayer();

  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_, bias_};
    return params;
  }
  ConnectionType src_neuron_connection(int k) const  override {
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }

 protected:
  int kernel_, pad_,  stride_;
  int batchsize_,  channels_, height_, width_;
  int col_height_, col_width_, conv_height_, conv_width_, num_filters_;
  Param* weight_, *bias_;
  Blob<float> col_data_, col_grad_;
};

/**
 * Use im2col from Caffe
 */
class CConvolutionLayer : public ConvolutionLayer {
 public:
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_NEURON_LAYER_CONVOLUTION_H_
