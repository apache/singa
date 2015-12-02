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

#ifndef SINGA_NEURALNET_CONNECTION_LAYER_BRIDGE_H_
#define SINGA_NEURALNET_CONNECTION_LAYER_BRIDGE_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "singa/comm/socket.h"
#include "singa/neuralnet/layer.h"

namespace singa {

class BridgeLayer : public ConnectionLayer {
 public:
  void set_ready(bool a) { ready_ = a; }
  bool ready() const { return ready_; }
  // Bind the layer with dealer instance by worker at runtime
  void MakePaired(Layer* pair, int grp_id, Dealer* dealer,
                  std::unordered_map<std::string, Layer*>* name2bridge);
  // Send blobs to other workers due to model partitions
  void SendBlobs(bool handle_data);
  // Receive blobs from other workers due to model partitions;
  void ReceiveBlobs(bool handle_data);

 protected:
  //!< true if received grad from BridgeDstLayer
  bool ready_ = false;
  int group_id_ = 0;
  Layer* pair_ = nullptr;
  Dealer* dealer_ = nullptr;
  std::unordered_map<std::string, Layer*>* name2bridge_ = nullptr;
};

/**
 * For sending data to layer on other threads which may resident on other nodes
 * due to layer/data partition.
 */
class BridgeSrcLayer : public BridgeLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
};

/**
 * For recv data from layer on other threads which may resident on other nodes
 * due to layer/data partiton
 */
class BridgeDstLayer : public BridgeLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_CONNECTION_LAYER_BRIDGE_H_
