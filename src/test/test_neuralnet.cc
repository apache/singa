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

#include "gtest/gtest.h"
#include "singa/driver.h"
#include "singa/neuralnet/connection_layer.h"
#include "singa/neuralnet/neuralnet.h"
#include "singa/neuralnet/neuron_layer.h"

using namespace singa;

const int N = 10;  // size of dim 0
const int M = 20;  // size of dim 1
const int K = 2;  // size of partitions

TEST(NeuralNet, RegisterLayers) {
  Driver driver;
  driver.RegisterLayer<DummyLayer, int>(kDummy);
  driver.RegisterLayer<SliceLayer, int>(kSlice);
  driver.RegisterLayer<SplitLayer, int>(kSplit);
  driver.RegisterLayer<ConcateLayer, int>(kConcate);
  driver.RegisterLayer<BridgeSrcLayer, int>(kBridgeSrc);
  driver.RegisterLayer<BridgeDstLayer, int>(kBridgeDst);
}

TEST(NeuralNet, AddModelSplitLayers) {
  NetProto proto;
  // use dummy as input layer
  LayerProto* proto_in = proto.add_layer();
  proto_in->set_name("dummy_input");
  proto_in->set_type(kDummy);
  proto_in->mutable_dummy_conf()->set_input(true);
  proto_in->mutable_dummy_conf()->add_shape(N);
  proto_in->mutable_dummy_conf()->add_shape(M);
  // use 2 dummy neuron layers
  for (int i = 0; i < 2; ++i) {
    LayerProto* proto_neuron = proto.add_layer();
    proto_neuron->set_name("dummy_neuron_" + std::to_string(i));
    proto_neuron->set_type(kDummy);
    proto_neuron->add_srclayers("dummy_input");
  }
  // use dummy as output layer
  for (int i = 0; i < 2; ++i) {
    LayerProto* proto_out = proto.add_layer();
    proto_out->set_name("dummy_output" + std::to_string(i));
    proto_out->set_type(kDummy);
    proto_out->mutable_dummy_conf()->set_output(true);
    proto_out->add_srclayers("dummy_neuron_" + std::to_string(i));
  }
  NeuralNet::Create(proto, kTrain, K);
}

TEST(NeuralNet, DirectConnection) {
  NetProto proto;
  // use dummy as input layer
  LayerProto* proto_in = proto.add_layer();
  proto_in->set_name("dummy_input");
  proto_in->set_type(kDummy);
  proto_in->mutable_dummy_conf()->set_input(true);
  proto_in->mutable_dummy_conf()->add_shape(N);
  proto_in->mutable_dummy_conf()->add_shape(M);
  // use dummy neuron layer
  LayerProto* proto_neuron = proto.add_layer();
  proto_neuron->set_name("dummy_neuron");
  proto_neuron->set_type(kDummy);
  proto_neuron->add_srclayers("dummy_input");
  // use dummy as output layer
  LayerProto* proto_out = proto.add_layer();
  proto_out->set_name("dummy_output");
  proto_out->set_type(kDummy);
  proto_out->mutable_dummy_conf()->set_output(true);
  proto_out->add_srclayers("dummy_neuron");
  NeuralNet::Create(proto, kTrain, K);
}

TEST(NeuralNet, SliceConcate) {
  NetProto proto;
  // use dummy as input layer
  LayerProto* proto_in = proto.add_layer();
  proto_in->set_name("dummy_input");
  proto_in->set_type(kDummy);
  proto_in->mutable_dummy_conf()->set_input(true);
  proto_in->mutable_dummy_conf()->add_shape(N);
  proto_in->mutable_dummy_conf()->add_shape(M);
  // use dummy neuron layer
  LayerProto* proto_neuron = proto.add_layer();
  proto_neuron->set_name("dummy_neuron");
  proto_neuron->set_type(kDummy);
  proto_neuron->add_srclayers("dummy_input");
  // use dummy as output layer
  LayerProto* proto_out = proto.add_layer();
  proto_out->set_name("dummy_output");
  proto_out->set_type(kDummy);
  proto_out->set_partition_dim(1);
  proto_out->mutable_dummy_conf()->set_output(true);
  proto_out->add_srclayers("dummy_neuron");
  NeuralNet::Create(proto, kTrain, K);
}
