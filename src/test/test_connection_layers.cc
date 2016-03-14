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

#include <string>
#include <unordered_map>
#include <vector>
#include "gtest/gtest.h"
#include "singa/comm/msg.h"
#include "singa/comm/socket.h"
#include "singa/neuralnet/connection_layer.h"
#include "singa/neuralnet/neuron_layer.h"
#include "singa/proto/job.pb.h"

using namespace singa;

const int N = 10;  // size of dim 0
const int M = 20;  // size of dim 1
const int K = 5;  // size of partitions

TEST(ConnectionLayerTest, DummyTest) {
  // use dummy as input layer
  vector<Layer*> src_in;
  LayerProto proto_in;
  proto_in.set_name("dummy_input");
  proto_in.mutable_dummy_conf()->set_input(true);
  proto_in.mutable_dummy_conf()->add_shape(N);
  proto_in.mutable_dummy_conf()->add_shape(M);
  DummyLayer in;
  in.Setup(proto_in, src_in);
  ASSERT_EQ(in.data(nullptr).shape(0), N);
  ASSERT_EQ(in.data(nullptr).shape(1), M);
  in.ComputeFeature(0, src_in);

  // use dummy as neuron layer
  vector<Layer*> src_neu;
  src_neu.push_back(static_cast<Layer*>(&in));
  LayerProto proto_neu;
  proto_neu.set_name("dummy_neuron");
  proto_neu.mutable_dummy_conf();
  DummyLayer neu;
  neu.Setup(proto_neu, src_neu);
  ASSERT_EQ(neu.data(nullptr).shape(0), N);
  ASSERT_EQ(neu.data(nullptr).shape(1), M);
  neu.ComputeFeature(0, src_neu);
  ASSERT_EQ(in.data(nullptr).count(), neu.data(nullptr).count());
  for (int i = 0; i < in.data(nullptr).count(); ++i)
    ASSERT_EQ(in.data(nullptr).cpu_data()[i], neu.data(nullptr).cpu_data()[i]);

  // use dummy as output layer
  vector<Layer*> src_out;
  src_out.push_back(static_cast<Layer*>(&neu));
  LayerProto proto_out;
  proto_out.set_name("dummy_output");
  proto_out.mutable_dummy_conf()->set_output(true);
  DummyLayer out;
  out.Setup(proto_out, src_out);
  ASSERT_EQ(out.data(nullptr).shape(0), N);
  ASSERT_EQ(out.data(nullptr).shape(1), M);
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

TEST(ConnectionLayerTest, BridgeTest) {
  // use dummy as input layer
  vector<Layer*> src_in;
  LayerProto proto_in;
  proto_in.set_name("dummy_input");
  proto_in.mutable_dummy_conf()->set_input(true);
  proto_in.mutable_dummy_conf()->add_shape(N);
  proto_in.mutable_dummy_conf()->add_shape(M);
  DummyLayer in;
  in.Setup(proto_in, src_in);

  // add src bridge layer
  vector<Layer*> src_src;
  src_src.push_back(static_cast<Layer*>(&in));
  LayerProto proto_src;
  proto_src.set_name("bridge_src");
  BridgeSrcLayer src;
  src.Setup(proto_src, src_src);
  ASSERT_EQ(src.data(nullptr).shape(0), N);
  ASSERT_EQ(src.data(nullptr).shape(1), M);

  // add dst bridge layer
  vector<Layer*> src_dst;
  src_dst.push_back(static_cast<Layer*>(&src));
  LayerProto proto_dst;
  proto_dst.set_name("bridge_dst");
  BridgeDstLayer dst;
  dst.Setup(proto_dst, src_dst);
  ASSERT_EQ(dst.data(nullptr).shape(0), N);
  ASSERT_EQ(dst.data(nullptr).shape(1), M);

  // bind bridges to socket
  Router router(N);
  router.Bind("inproc://router");
  Dealer dealer(0);
  dealer.Connect("inproc://router");
  std::unordered_map<std::string, Layer*> name2bridge;
  name2bridge[src.name()] = &src;
  name2bridge[dst.name()] = &dst;
  src.MakePaired(static_cast<Layer*>(&dst), 0, &dealer, &name2bridge);
  dst.MakePaired(static_cast<Layer*>(&src), 0, &dealer, &name2bridge);

  // use dummy as output layer
  LayerProto proto_out;
  vector<Layer*> src_out;
  src_out.push_back(static_cast<Layer*>(&dst));
  proto_out.set_name("dummy_output");
  proto_out.mutable_dummy_conf()->set_output(true);
  DummyLayer out;
  out.Setup(proto_out, src_out);

  // test for computing feature
  in.ComputeFeature(0, src_in);
  src.ComputeFeature(0, src_src);
  Msg* msg_data = router.Receive();
  router.Send(&msg_data);
  dst.ComputeFeature(0, src_dst);
  out.ComputeFeature(0, src_out);
  for (int i = 0; i < in.data(nullptr).count(); ++i)
    ASSERT_EQ(in.data(nullptr).cpu_data()[i], out.data(nullptr).cpu_data()[i]);

  // test for computing gradient
  out.ComputeGradient(0, src_out);
  dst.ComputeGradient(0, src_dst);
  Msg* msg_grad = router.Receive();
  router.Send(&msg_grad);
  src.ComputeGradient(0, src_src);
  in.ComputeGradient(0, src_in);
  for (int i = 0; i < in.grad(nullptr).count(); ++i)
    ASSERT_EQ(in.grad(nullptr).cpu_data()[i], out.grad(nullptr).cpu_data()[i]);
}

TEST(ConnectionLayerTest, DataSliceTest) {
  // use dummy as input layer
  vector<Layer*> src_in;
  LayerProto proto_in;
  proto_in.set_name("dummy_input");
  proto_in.mutable_dummy_conf()->set_input(true);
  proto_in.mutable_dummy_conf()->add_shape(N);
  proto_in.mutable_dummy_conf()->add_shape(M);
  DummyLayer in;
  in.Setup(proto_in, src_in);

  // add slice layer
  vector<Layer*> src_slice;
  src_slice.push_back(static_cast<Layer*>(&in));
  LayerProto proto_slice;
  proto_slice.set_name("slice");
  proto_slice.mutable_slice_conf()->set_slice_dim(0);
  proto_slice.mutable_slice_conf()->set_num_slices(K);
  SliceLayer slice;
  slice.Setup(proto_slice, src_slice);
  ASSERT_EQ(slice.data(nullptr).shape(0), N / K);
  ASSERT_EQ(slice.data(nullptr).shape(1), M);

  // use dummy as output layers
  LayerProto proto_out[K];
  vector<Layer*> src_out[K];
  DummyLayer out[K];
  for (int i = 0; i < K; ++i) {
    src_out[i].push_back(static_cast<Layer*>(&slice));
    proto_out[i].set_name("dummy_output_"+std::to_string(i));
    proto_out[i].set_partition_id(i);
    proto_out[i].mutable_dummy_conf()->set_output(true);
    out[i].Setup(proto_out[i], src_out[i]);
  }

  // test for computing feature
  in.ComputeFeature(0, src_in);
  slice.ComputeFeature(0, src_slice);
  for (int i = 0; i < K; ++i)
    out[i].ComputeFeature(0, src_out[i]);
  int step = (N * M) / K;
  for (int i = 0; i < in.data(nullptr).count(); ++i) {
    ASSERT_EQ(in.data(nullptr).cpu_data()[i],
              out[i / step].data(nullptr).cpu_data()[i % step]);
  }

  // test for computing gradient
  for (int i = 0; i < K; ++i)
    out[i].ComputeGradient(0, src_out[i]);
  slice.ComputeGradient(0, src_slice);
  in.ComputeGradient(0, src_in);
  for (int i = 0; i < in.grad(nullptr).count(); ++i) {
    ASSERT_EQ(in.grad(nullptr).cpu_data()[i],
              out[i / step].grad(nullptr).cpu_data()[i % step]);
  }
}

TEST(ConnectionLayerTest, ModelSliceTest) {
  // use dummy as input layer
  vector<Layer*> src_in;
  LayerProto proto_in;
  proto_in.set_name("dummy_input");
  proto_in.mutable_dummy_conf()->set_input(true);
  proto_in.mutable_dummy_conf()->add_shape(N);
  proto_in.mutable_dummy_conf()->add_shape(M);
  DummyLayer in;
  in.Setup(proto_in, src_in);

  // add slice layer
  vector<Layer*> src_slice;
  src_slice.push_back(static_cast<Layer*>(&in));
  LayerProto proto_slice;
  proto_slice.set_name("slice");
  proto_slice.mutable_slice_conf()->set_slice_dim(1);
  proto_slice.mutable_slice_conf()->set_num_slices(K);
  SliceLayer slice;
  slice.Setup(proto_slice, src_slice);
  ASSERT_EQ(slice.data(nullptr).shape(0), N);
  ASSERT_EQ(slice.data(nullptr).shape(1), M / K);

  // use dummy as output layers
  LayerProto proto_out[K];
  vector<Layer*> src_out[K];
  DummyLayer out[K];
  for (int i = 0; i < K; ++i) {
    src_out[i].push_back(static_cast<Layer*>(&slice));
    proto_out[i].set_name("dummy_output_"+std::to_string(i));
    proto_out[i].set_partition_id(i);
    proto_out[i].mutable_dummy_conf()->set_output(true);
    out[i].Setup(proto_out[i], src_out[i]);
  }

  // test for computing feature
  in.ComputeFeature(0, src_in);
  slice.ComputeFeature(0, src_slice);
  for (int i = 0; i < K; ++i)
    out[i].ComputeFeature(0, src_out[i]);
  int step = M / K;
  int offset = 0;
  for (int i = 0; i < in.data(nullptr).count(); ++i) {
    if (i && i % M == 0) offset += step;
    ASSERT_EQ(in.data(nullptr).cpu_data()[i],
              out[(i / step) % K].data(nullptr).cpu_data()[offset + i % step]);
  }

  // test for computing gradient
  for (int i = 0; i < K; ++i)
    out[i].ComputeGradient(0, src_out[i]);
  slice.ComputeGradient(0, src_slice);
  in.ComputeGradient(0, src_in);
  offset = 0;
  for (int i = 0; i < in.grad(nullptr).count(); ++i) {
    if (i && i % M == 0) offset += step;
    ASSERT_EQ(in.grad(nullptr).cpu_data()[i],
              out[(i / step) % K].grad(nullptr).cpu_data()[offset + i % step]);
  }
}

TEST(ConnectionLayerTest, DataConcateTest) {
  // use dummy as input layers
  LayerProto proto_in[K];
  vector<Layer*> src_in[K];
  DummyLayer in[K];
  for (int i = 0; i < K; ++i) {
    proto_in[i].set_name("dummy_input_"+std::to_string(i));
    proto_in[i].set_partition_id(i);
    proto_in[i].mutable_dummy_conf()->set_input(true);
    proto_in[i].mutable_dummy_conf()->add_shape(N / K);
    proto_in[i].mutable_dummy_conf()->add_shape(M);
    in[i].Setup(proto_in[i], src_in[i]);
  }

  // add concate layer
  vector<Layer*> src_concate;
  for (int i = 0; i < K; ++i)
    src_concate.push_back(static_cast<Layer*>(&in[i]));
  LayerProto proto_concate;
  proto_concate.set_name("concate");
  proto_concate.mutable_concate_conf()->set_concate_dim(0);
  proto_concate.mutable_concate_conf()->set_num_concates(K);
  ConcateLayer concate;
  concate.Setup(proto_concate, src_concate);
  ASSERT_EQ(concate.data(static_cast<Layer*>(&concate)).shape(0), N);
  ASSERT_EQ(concate.data(static_cast<Layer*>(&concate)).shape(1), M);

  // use dummy as output layer
  vector<Layer*> src_out;
  src_out.push_back(static_cast<Layer*>(&concate));
  LayerProto proto_out;
  proto_out.set_name("dummy_output");
  proto_out.mutable_dummy_conf()->set_output(true);
  DummyLayer out;
  out.Setup(proto_out, src_out);

  // test for computing feature
  for (int i = 0; i < K; ++i)
    in[i].ComputeFeature(0, src_in[i]);
  concate.ComputeFeature(0, src_concate);
  out.ComputeFeature(0, src_out);
  int step = (N * M) / K;
  for (int i = 0; i < out.data(nullptr).count(); ++i) {
    ASSERT_EQ(in[i / step].data(nullptr).cpu_data()[i % step],
              out.data(nullptr).cpu_data()[i]);
  }

  // test for computing gradient
  out.ComputeGradient(0, src_out);
  concate.ComputeGradient(0, src_concate);
  for (int i = 0; i < K; ++i)
    in[i].ComputeGradient(0, src_in[i]);
  for (int i = 0; i < out.grad(nullptr).count(); ++i) {
    ASSERT_EQ(in[i / step].grad(nullptr).cpu_data()[i % step],
              out.grad(nullptr).cpu_data()[i]);
  }
}

TEST(ConnectionLayerTest, ModelConcateTest) {
  // use dummy as input layers
  LayerProto proto_in[K];
  vector<Layer*> src_in[K];
  DummyLayer in[K];
  for (int i = 0; i < K; ++i) {
    proto_in[i].set_name("dummy_input_"+std::to_string(i));
    proto_in[i].set_partition_id(i);
    proto_in[i].mutable_dummy_conf()->set_input(true);
    proto_in[i].mutable_dummy_conf()->add_shape(N);
    proto_in[i].mutable_dummy_conf()->add_shape(M / K);
    in[i].Setup(proto_in[i], src_in[i]);
  }

  // add concate layer
  vector<Layer*> src_concate;
  for (int i = 0; i < K; ++i)
    src_concate.push_back(static_cast<Layer*>(&in[i]));
  LayerProto proto_concate;
  proto_concate.set_name("concate");
  proto_concate.mutable_concate_conf()->set_concate_dim(1);
  proto_concate.mutable_concate_conf()->set_num_concates(K);
  ConcateLayer concate;
  concate.Setup(proto_concate, src_concate);
  ASSERT_EQ(concate.data(static_cast<Layer*>(&concate)).shape(0), N);
  ASSERT_EQ(concate.data(static_cast<Layer*>(&concate)).shape(1), M);

  // use dummy as output layer
  vector<Layer*> src_out;
  src_out.push_back(static_cast<Layer*>(&concate));
  LayerProto proto_out;
  proto_out.set_name("dummy_output");
  proto_out.mutable_dummy_conf()->set_output(true);
  DummyLayer out;
  out.Setup(proto_out, src_out);

  // test for computing feature
  for (int i = 0; i < K; ++i)
    in[i].ComputeFeature(0, src_in[i]);
  concate.ComputeFeature(0, src_concate);
  out.ComputeFeature(0, src_out);
  int step = M / K;
  int offset = 0;
  for (int i = 0; i < out.grad(nullptr).count(); ++i) {
    if (i && i % M == 0) offset += step;
    ASSERT_EQ(in[(i / step) % K].data(nullptr).cpu_data()[offset + i % step],
              out.data(nullptr).cpu_data()[i]);
  }

  // test for computing gradient
  out.ComputeGradient(0, src_out);
  concate.ComputeGradient(0, src_concate);
  for (int i = 0; i < K; ++i)
    in[i].ComputeGradient(0, src_in[i]);
  offset = 0;
  for (int i = 0; i < out.grad(nullptr).count(); ++i) {
    if (i && i % M == 0) offset += step;
    ASSERT_EQ(in[(i / step) % K].grad(nullptr).cpu_data()[offset + i % step],
              out.grad(nullptr).cpu_data()[i]);
  }
}

TEST(ConnectionLayerTest, SplitTest) {
  // use dummy as input layer
  vector<Layer*> src_in;
  LayerProto proto_in;
  proto_in.set_name("dummy_input");
  proto_in.mutable_dummy_conf()->set_input(true);
  proto_in.mutable_dummy_conf()->add_shape(N);
  proto_in.mutable_dummy_conf()->add_shape(M);
  DummyLayer in;
  in.Setup(proto_in, src_in);

  // add split layer
  vector<Layer*> src_split;
  src_split.push_back(static_cast<Layer*>(&in));
  LayerProto proto_split;
  proto_split.set_name("split");
  proto_split.mutable_split_conf()->set_num_splits(K);
  SplitLayer split;
  split.Setup(proto_split, src_split);
  ASSERT_EQ(split.data(static_cast<Layer*>(&split)).shape(0), N);
  ASSERT_EQ(split.data(static_cast<Layer*>(&split)).shape(1), M);

  // use dummy as output layers
  LayerProto proto_out[K];
  vector<Layer*> src_out[K];
  DummyLayer out[K];
  for (int i = 0; i < K; ++i) {
    src_out[i].push_back(static_cast<Layer*>(&split));
    proto_out[i].set_name("dummy_output_"+std::to_string(i));
    proto_out[i].set_partition_id(i);
    proto_out[i].mutable_dummy_conf()->set_output(true);
    out[i].Setup(proto_out[i], src_out[i]);
  }

  // test for computing feature
  in.ComputeFeature(0, src_in);
  split.ComputeFeature(0, src_split);
  for (int i = 0; i < K; ++i)
    out[i].ComputeFeature(0, src_out[i]);
  for (int i = 0; i < in.data(nullptr).count(); ++i) {
    for (int k = 0; k < K; ++k)
      ASSERT_EQ(in.data(nullptr).cpu_data()[i],
                out[k].data(nullptr).cpu_data()[i]);
  }

  // test for computing gradient
  for (int i = 0; i < K; ++i)
    out[i].ComputeGradient(0, src_out[i]);
  split.ComputeGradient(0, src_split);
  in.ComputeGradient(0, src_in);
  for (int i = 0; i < in.grad(nullptr).count(); ++i) {
    float grad = 0;
    for (int k = 0; k < K; ++k) grad += out[k].grad(nullptr).cpu_data()[i];
    ASSERT_EQ(in.grad(nullptr).cpu_data()[i], grad);
  }
}
