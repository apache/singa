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
#include "singa/core/device.h"
#include "singa/core/scheduler.h"
#include "singa/core/tensor.h"
#include "singa/singa_config.h"

using singa::Blk2InfoMap;
using singa::BlkInfo;
using singa::BlockType;
using singa::BlockVec;
using singa::Context;
using singa::Device;
using singa::Edge;
using singa::EdgeVec;
using singa::Graph;
using singa::Node;
using singa::NodeVec;
using singa::Shape;
using singa::Tensor;

class TestGraph : public testing::Test {
 protected:
  virtual void SetUp();
  virtual void TearDown();

 protected:
  std::vector<std::shared_ptr<Device> > devices;
};

void TestGraph::SetUp() {
  auto cpp_cpu = singa::Platform::GetDefaultDevice();
  devices.push_back(cpp_cpu);

#ifdef USE_CUDA
  auto cuda_gpu = std::make_shared<singa::CudaGPU>();
  devices.push_back(cuda_gpu);
#endif
}

void TestGraph::TearDown() { devices.clear(); }

#define CheckNode(node, id_, in_edges_, out_edges_)         \
  do {                                                      \
    EXPECT_EQ(id_, node->id());                             \
    EXPECT_EQ(in_edges_.size(), node->in_edges().size());   \
    EXPECT_EQ(out_edges_.size(), node->out_edges().size()); \
    for (size_t i = 0; i < in_edges_.size(); ++i) {         \
      EXPECT_EQ(in_edges_[i], node->in_edges()[i])          \
          << "in_edges is wrong at index [" << i << "]";    \
    }                                                       \
    for (size_t i = 0; i < out_edges_.size(); ++i) {        \
      EXPECT_EQ(out_edges_[i], node->out_edges()[i])        \
          << "out_edges is wrong at index [" << i << "]";   \
    }                                                       \
  } while (false)

#define CheckEdge(edge, id_, block_, src_node_, dst_node_) \
  do {                                                     \
    EXPECT_EQ(id_, edge->id());                            \
    EXPECT_EQ(block_, edge->block());                      \
    EXPECT_EQ(src_node_, edge->src_node());                \
    EXPECT_EQ(dst_node_, edge->dst_node());                \
  } while (false)

#define CheckBlock(blkInfo, id_, blk_, type_, ref_, write_node_, last_node_) \
  do {                                                                       \
    EXPECT_EQ(id_, blkInfo->id());                                           \
    EXPECT_EQ(blk_, blkInfo->block());                                       \
    EXPECT_EQ(type_, blkInfo->type());                                       \
    EXPECT_EQ(ref_, blkInfo->graph_ref());                                   \
    EXPECT_EQ(write_node_, blkInfo->write_node());                           \
    EXPECT_EQ(last_node_, blkInfo->last_node());                             \
  } while (false)

#define CheckWriteBlocks(write_blocks, correct_write_blocks)     \
  do {                                                           \
    EXPECT_EQ(correct_write_blocks.size(), write_blocks.size()); \
    for (size_t i = 0; i < write_blocks.size(); ++i) {           \
      EXPECT_EQ(correct_write_blocks[i], write_blocks[i])        \
          << "write_blocks is wrong at index [" << i << "]";     \
    }                                                            \
  } while (false)

TEST_F(TestGraph, AddOp) {
  for (auto &dev : devices) {
    Graph graph(dev.get());

    Tensor in(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    auto op = [in, out](Context *ctx) mutable {};

    graph.AddOperation(op, {in.block()}, {out.block()});

    auto nodes = graph.nodes();
    auto edges = graph.edges();
    auto blocks = graph.blocks();
    auto write_blocks = graph.write_blocks();

    EXPECT_EQ(1u, nodes.size());
    EXPECT_EQ(1u, edges.size());
    EXPECT_EQ(2u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto node = nodes[0];
    auto edge = edges[0];
    auto block1 = blocks[in.block()];
    auto block2 = blocks[out.block()];

    CheckNode(node, 0u, EdgeVec({nullptr}), EdgeVec({}));

    CheckEdge(edge, 0u, in.block(), nullptr, node);

    CheckBlock(block1, 0u, in.block(), BlockType::kInput, 1u, nullptr, node);
    CheckBlock(block2, 1u, out.block(), BlockType::kEnd, 1u, node, node);

    CheckWriteBlocks(write_blocks, BlockVec({out.block()}));

    EXPECT_EQ(true, graph.dirty());
  }
}

TEST_F(TestGraph, AddSyncOp) {
  for (auto &dev : devices) {
    Graph graph(dev.get());

    Tensor in(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    auto op = [in, out](Context *ctx) mutable {};

    graph.AddOperation(op, {in.block()}, {out.block()});

    auto nodes = graph.nodes();
    auto edges = graph.edges();
    auto blocks = graph.blocks();
    auto write_blocks = graph.write_blocks();

    EXPECT_EQ(1u, nodes.size());
    EXPECT_EQ(1u, edges.size());
    EXPECT_EQ(2u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto node = nodes[0];
    auto edge = edges[0];
    auto block1 = blocks[in.block()];
    auto block2 = blocks[out.block()];
    auto write_block = write_blocks[0];

    CheckNode(node, 0u, EdgeVec({edge}), EdgeVec({}));

    CheckEdge(edge, 0u, in.block(), nullptr, node);

    CheckBlock(block1, 0u, in.block(), BlockType::kInput, 1u, nullptr, node);
    CheckBlock(block2, 1u, out.block(), BlockType::kEnd, 1u, node, node);

    CheckWriteBlocks(write_blocks, BlockVec({out.block()}));

    EXPECT_EQ(true, graph.dirty());
  }
}

TEST_F(TestGraph, AddInplaceOp) {}

TEST_F(TestGraph, BlockTypeInput) {}

TEST_F(TestGraph, BlockTypeParam) {}

TEST_F(TestGraph, BlockTypeInter) {}

TEST_F(TestGraph, BlockTypeEnd) {}

TEST_F(TestGraph, RunGraph) {}

TEST_F(TestGraph, RunInSerial) {}

TEST_F(TestGraph, AutoRecycle) {}
