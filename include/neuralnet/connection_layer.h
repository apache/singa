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

#ifndef SINGA_NEURALNET_CONNECTION_LAYER_H_
#define SINGA_NEURALNET_CONNECTION_LAYER_H_

#include <vector>
#include "neuralnet/layer.h"

/**
 * \file this file includes the declarations of layers that inherit the
 * base ConnectionLayer.
 */
namespace singa {
class BridgeLayer : virtual public ConnectionLayer {
 public:
  void set_ready(bool a) {
    ready_ = a;
  }
  bool ready() const {
    return ready_;
  }
  virtual bool is_bridgesrclayer() const {
    return false;
  }
  virtual bool is_bridgedstlayer() const {
    return false;
  }

 protected:
  //!< true if received grad from BridgeDstLayer
  bool ready_;
};

/**
 * For recv data from layer on other threads which may resident on other nodes
 * due to layer/data partiton
 */
class BridgeDstLayer : public BridgeLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override {
    // reset ready_ for next iteration.
    ready_ = false;
  }
  void ComputeGradient(int flag, Metric* perf) override {}
  bool is_bridgedstlayer() const {
    return true;
  }
};

/**
 * For sending data to layer on other threads which may resident on other nodes
 * due to layer/data partition.
 */
class BridgeSrcLayer : public BridgeLayer {
 public:
  void ComputeFeature(int flag, Metric* perf) override {}
  void ComputeGradient(int flag, Metric* perf) override {
    ready_ = false;
  }
  const Blob<float>& data(const Layer* from) const override {
    return srclayers_[0]->data(this);
  }
  Blob<float>* mutable_data(const Layer* from) override {
    return srclayers_[0]->mutable_data(this);
  }
  const Blob<float>& grad(const Layer* from) const override {
    return srclayers_[0]->grad(this);
  }
  Blob<float>* mutable_grad(const Layer* from) override {
    return srclayers_[0]->mutable_grad(this);
  }
  bool is_bridgesrclayer() const override {
    return true;
  }
};


/**
 * Connect multiple (src) layers with a single (dst) layer.
 *
 * It concates feature Blobs (i.e., matrix) of src layers on one dimension.
 * The concated feature Blob will be fed into the dst layer.
 */
class ConcateLayer : public ConnectionLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
};

/**
 * Connect a single (src) layer with multiple (dst) layers.
 *
 * It slices the feature Blob (i.e., matrix) of the src layer on one dimension.
 * The sliced feature Blobs will be fed into dst layers.
 */
class SliceLayer : public ConnectionLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

 private:
  std::vector<Blob<float>> datavec_;
  std::vector<Blob<float>> gradvec_;
  int slice_dim_;
  int slice_num_;
};

/**
 * Connect a single (src) layer with multiple dst layers.
 *
 * It replicates the feature Blob of the src layer.
 * Each replicated feature Blob will be fed into one dst layer.
 * It aggregates gradients set by all dst layers and set it to the src layer.
 */
class SplitLayer : public ConnectionLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

 protected:
  Blob<float> grads_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_CONNECTION_LAYER_H_
