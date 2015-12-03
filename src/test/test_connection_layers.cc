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
#include "singa/neuralnet/connection_layer/bridge.h"
#include "singa/neuralnet/neuron_layer/dummy.h"
#include "singa/proto/job.pb.h"

using namespace singa;

TEST(ConnectionLayerTest, DummyTest) {
  // use dummy as input layer
  LayerProto proto_in;
  vector<Layer*> src_in;
  proto_in.set_name("dummy_input");
  proto_in.mutable_dummy_conf()->set_input(true);
  proto_in.mutable_dummy_conf()->add_shape(10);
  proto_in.mutable_dummy_conf()->add_shape(20);
  DummyLayer in;
  in.Setup(proto_in, src_in);
  ASSERT_EQ(in.data(nullptr).shape(0), 10);
  ASSERT_EQ(in.data(nullptr).shape(1), 20);
  in.ComputeFeature(0, src_in);
 
  // use dummy as neuron layer
  LayerProto proto_neu;
  vector<Layer*> src_neu;
  src_neu.push_back(static_cast<Layer*>(&in));
  proto_neu.set_name("dummy_neuron");
  proto_neu.mutable_dummy_conf();
  DummyLayer neu;
  neu.Setup(proto_neu, src_neu);
  ASSERT_EQ(neu.data(nullptr).shape(0), 10);
  ASSERT_EQ(neu.data(nullptr).shape(1), 20);
  neu.ComputeFeature(0, src_neu);
  ASSERT_EQ(in.data(nullptr).count(), neu.data(nullptr).count());
  for (int i = 0; i < in.data(nullptr).count(); ++i)
    ASSERT_EQ(in.data(nullptr).cpu_data()[i], neu.data(nullptr).cpu_data()[i]);

  // use dummy as output layer
  LayerProto proto_out;
  vector<Layer*> src_out;
  src_out.push_back(static_cast<Layer*>(&neu));
  proto_out.set_name("dummy_output");
  proto_out.mutable_dummy_conf()->set_output(true);
  DummyLayer out;
  out.Setup(proto_out, src_out);
  ASSERT_EQ(out.data(nullptr).shape(0), 10);
  ASSERT_EQ(out.data(nullptr).shape(1), 20);
  out.ComputeFeature(0, src_out);
  ASSERT_EQ(in.data(nullptr).count(), out.data(nullptr).count());
  for (int i = 0; i < in.data(nullptr).count(); ++i)
    ASSERT_EQ(in.data(nullptr).cpu_data()[i], out.data(nullptr).cpu_data()[i]);
 
  // test for computing gradient
  out.ComputeGradient(0, src_out);
  neu.ComputeGradient(0, src_neu);
  in.ComputeGradient(0, src_in);
  for (int i = 0; i < in.grad(nullptr).count(); ++i)
    ASSERT_EQ(in.grad(nullptr).cpu_data()[i], out.grad(nullptr).cpu_data()[i]);
}
