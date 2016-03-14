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

#include "singa/neuralnet/connection_layer.h"
#include "singa/comm/msg.h"

namespace singa {

using std::vector;

void BridgeLayer::MakePaired(Layer* pair, int grp_id, Dealer* dealer,
    std::unordered_map<std::string, Layer*>* name2bridge) {
  pair_ = pair;
  group_id_ = grp_id;
  dealer_ = dealer;
  name2bridge_ = name2bridge;
}

void BridgeLayer::SendBlobs(bool handle_data) {
  CHECK(dealer_) << "NULL dealer for bridges in worker (" << group_id_
                 << ", " << partition_id() << ")";
  Msg *msg = new Msg();
  msg->set_src(Addr(group_id_, partition_id(), kWorkerLayer));
  msg->set_dst(Addr(group_id_, pair_->partition_id(), kWorkerLayer));
  msg->AddFrame(pair_->name().c_str(), pair_->name().length());
  auto const& blob = handle_data ? data(nullptr) : grad(nullptr);
  msg->AddFrame(blob.cpu_data(), blob.count() * sizeof(float));
  dealer_->Send(&msg);
}

void BridgeLayer::ReceiveBlobs(bool handle_data) {
  CHECK(dealer_) << "NULL dealer for bridges in worker (" << group_id_
                 << ", " << partition_id() << ")";
  while (!ready()) {
    auto msg = dealer_->Receive();
    CHECK_EQ(AddrGrp(msg->src()), group_id_);
    string name(static_cast<char*>(msg->FrameData()), msg->FrameSize());
    auto receive_layer = name2bridge_->at(name);
    auto blob = handle_data ? receive_layer->mutable_data(nullptr) :
                receive_layer -> mutable_grad(nullptr);
    msg->NextFrame();
    memcpy(blob->mutable_cpu_data(), msg->FrameData(), msg->FrameSize());
    dynamic_cast<BridgeLayer*>(receive_layer)->set_ready(true);
    delete msg;
  }
}

void BridgeSrcLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CHECK_GE(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  data_.Reshape(srclayers[0]->data(this).shape());
  grad_.ReshapeLike(data_);
  data_.ShareData(srclayers[0]->mutable_data(this), false);
  grad_.ShareData(srclayers[0]->mutable_grad(this), false);
}

void BridgeSrcLayer::ComputeFeature(int flag, const vector<Layer*>& srcs) {
  // send data
  SendBlobs(true);
  // reset flag for receiving gradient in compute gradient phase
  set_ready(false);
}

void BridgeSrcLayer::ComputeGradient(int flag, const vector<Layer*>& srcs) {
  // receive gradient
  ReceiveBlobs(false);
}

void BridgeDstLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  data_.Reshape(srclayers[0]->data(this).shape());
  grad_.ReshapeLike(data_);
}

void BridgeDstLayer::ComputeFeature(int flag, const vector<Layer*>& srcs) {
  // receive data
  ReceiveBlobs(true);
}

void BridgeDstLayer::ComputeGradient(int flag, const vector<Layer*>& srcs) {
  // send gradient
  SendBlobs(false);
  // reset flag for receiving data in compute feature phase
  set_ready(false);
}

}  // namespace singa
