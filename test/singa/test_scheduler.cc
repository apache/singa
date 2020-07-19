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

#include <sstream>
#include <utility>

#include "gtest/gtest.h"
#include "singa/core/device.h"
#include "singa/core/scheduler.h"
#include "singa/core/tensor.h"
#include "singa/singa_config.h"

typedef std::vector<int> IntVec;
using singa::Blk2InfoMap;
using singa::BlkInfo;
using singa::Block;
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

namespace testing {
namespace internal {
enum GTestColor { COLOR_DEFAULT, COLOR_RED, COLOR_GREEN, COLOR_YELLOW };
extern void ColoredPrintf(GTestColor color, const char *fmt, ...);
}  // namespace internal
}  // namespace testing

class Gout : public std::stringstream {
 public:
  ~Gout() {
    testing::internal::ColoredPrintf(testing::internal::COLOR_GREEN,
                                     "[          ] ");
    testing::internal::ColoredPrintf(testing::internal::COLOR_YELLOW,
                                     str().c_str());
  }
};

#define GOUT Gout()

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

#define CheckBlock(blkInfo, id_, blk_, type_, ref_, write_edge_, used_nodes_) \
  do {                                                                        \
    EXPECT_EQ(id_, blkInfo->id());                                            \
    EXPECT_EQ(blk_, blkInfo->block());                                        \
    EXPECT_EQ(type_, blkInfo->type());                                        \
    EXPECT_EQ(ref_, blkInfo->graph_ref());                                    \
    EXPECT_EQ(write_edge_, blkInfo->write_edge());                            \
    EXPECT_EQ(used_nodes_, blkInfo->used_nodes());                            \
    for (size_t i = 0; i < used_nodes_.size(); ++i) {                         \
      EXPECT_EQ(used_nodes_[i], blkInfo->used_node(i))                        \
          << "used_nodes is different at index [" << i << "]";                \
    }                                                                         \
  } while (false)

#define CheckWriteBlocks(write_blocks, correct_write_blocks)     \
  do {                                                           \
    EXPECT_EQ(correct_write_blocks.size(), write_blocks.size()); \
    for (size_t i = 0; i < write_blocks.size(); ++i) {           \
      EXPECT_EQ(correct_write_blocks[i], write_blocks[i])        \
          << "write_blocks is wrong at index [" << i << "]";     \
    }                                                            \
  } while (false)

#define CheckFreeBlocks(node_id, blocks, free_blocks, correct_free_blocks)   \
  do {                                                                       \
    EXPECT_EQ(correct_free_blocks.size(), free_blocks.size());               \
    for (size_t i = 0; i < correct_free_blocks.size(); ++i) {                \
      bool flag = false;                                                     \
      for (size_t j = 0; j < free_blocks.size(); ++j) {                      \
        if (blocks.find(free_blocks[j])->second->id() ==                     \
            correct_free_blocks[i]) {                                        \
          flag = true;                                                       \
          break;                                                             \
        }                                                                    \
      }                                                                      \
      EXPECT_TRUE(flag) << "block [" << correct_free_blocks[i]               \
                        << "] is not recycled properly at node " << node_id; \
    }                                                                        \
  } while (false)

class TestGraph : public testing::Test {
 protected:
  virtual void SetUp();
  virtual void TearDown();

 protected:
  std::vector<std::pair<std::string, std::shared_ptr<Device> > > devices;
};

void TestGraph::SetUp() {
  auto cpp_cpu = singa::Platform::GetDefaultDevice();
  devices.push_back(std::make_pair("cpp_cpu", cpp_cpu));

#ifdef USE_CUDA
  auto cuda_gpu = std::make_shared<singa::CudaGPU>();
  devices.push_back(std::make_pair("cuda_gpu", cuda_gpu));
#endif
}

void TestGraph::TearDown() { devices.clear(); }

TEST_F(TestGraph, AddOp) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    Tensor in(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    auto op = [](Context *ctx) mutable {};

    graph.AddOperation(op, {in.block()}, {out.block()});

    EXPECT_EQ(1u, nodes.size());
    EXPECT_EQ(2u, edges.size());
    EXPECT_EQ(2u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto node = nodes[0];
    auto edge1 = edges[0];
    auto edge2 = edges[1];
    auto block1 = blocks.find(in.block())->second;
    auto block2 = blocks.find(out.block())->second;

    CheckNode(node, 0, EdgeVec({edge1}), EdgeVec({edge2}));
    CheckEdge(edge1, 0, in.block(), nullptr, node);
    CheckEdge(edge2, 1, out.block(), node, nullptr);
    CheckBlock(block1, 0, in.block(), BlockType::kInput, 1, nullptr,
               NodeVec({}));
    CheckBlock(block2, 1, out.block(), BlockType::kEnd, 1, edge2, NodeVec({}));
    CheckWriteBlocks(write_blocks, BlockVec({out.block()}));
    EXPECT_TRUE(graph.dirty());
  }
}

TEST_F(TestGraph, AddSyncOp) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    Tensor in(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    auto op = [](Context *ctx) mutable {};

    graph.AddOperation(op, {in.block()}, {out.block()});
    graph.AddOperation(op, {}, {});

    EXPECT_EQ(2u, nodes.size());
    EXPECT_EQ(3u, edges.size());
    EXPECT_EQ(2u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto node1 = nodes[0];
    auto node2 = nodes[1];
    auto edge1 = edges[0];
    auto edge2 = edges[1];
    auto edge3 = edges[2];
    auto block1 = blocks.find(in.block())->second;
    auto block2 = blocks.find(out.block())->second;

    CheckNode(node1, 0, EdgeVec({edge1}), EdgeVec({edge2}));
    CheckNode(node2, 1, EdgeVec({edge2}), EdgeVec({edge3}));
    CheckEdge(edge1, 0, in.block(), nullptr, node1);
    CheckEdge(edge2, 1, out.block(), node1, node2);
    CheckEdge(edge3, 2, out.block(), node2, nullptr);
    CheckBlock(block1, 0, in.block(), BlockType::kInput, 1, nullptr,
               NodeVec({}));
    CheckBlock(block2, 1, out.block(), BlockType::kInter, 1, edge3,
               NodeVec({}));
    CheckWriteBlocks(write_blocks, BlockVec({out.block()}));
    EXPECT_TRUE(graph.dirty());
  }
}

TEST_F(TestGraph, AddInplaceOp) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    Tensor in(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    auto op = [](Context *ctx) mutable {};

    graph.AddOperation(op, {in.block()}, {in.block()});

    EXPECT_EQ(1u, nodes.size());
    EXPECT_EQ(2u, edges.size());
    EXPECT_EQ(1u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto node1 = nodes[0];
    auto edge1 = edges[0];
    auto edge2 = edges[1];
    auto block1 = blocks.find(in.block())->second;

    CheckNode(node1, 0, EdgeVec({edge1}), EdgeVec({edge2}));
    CheckEdge(edge1, 0, in.block(), nullptr, node1);
    CheckEdge(edge2, 1, in.block(), node1, nullptr);
    CheckBlock(block1, 0, in.block(), BlockType::kParam, 2, edge2, NodeVec({}));
    CheckWriteBlocks(write_blocks, BlockVec({in.block()}));
    EXPECT_TRUE(graph.dirty());

    graph.AddOperation(op, {in.block(), out.block()}, {out.block()});

    EXPECT_EQ(2u, nodes.size());
    EXPECT_EQ(4u, edges.size());
    EXPECT_EQ(2u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto node2 = nodes[1];
    auto edge3 = edges[2];
    auto edge4 = edges[3];
    auto block2 = blocks.find(out.block())->second;

    CheckNode(node1, 0, EdgeVec({edge1}), EdgeVec({edge2}));
    CheckNode(node2, 1, EdgeVec({edge2, edge3}), EdgeVec({edge4}));
    CheckEdge(edge2, 1, in.block(), node1, node2);
    CheckEdge(edge3, 2, out.block(), nullptr, node2);
    CheckEdge(edge4, 3, out.block(), node2, nullptr);
    CheckBlock(block1, 0, in.block(), BlockType::kParam, 3, edge2, NodeVec({}));
    CheckBlock(block2, 1, out.block(), BlockType::kParam, 2, edge4,
               NodeVec({}));
    CheckWriteBlocks(write_blocks, BlockVec({out.block()}));
    EXPECT_TRUE(graph.dirty());
  }
}

TEST_F(TestGraph, BlockTypeInput) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    Tensor in(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    auto op = [](Context *ctx) mutable {};

    graph.AddOperation(op, {in.block()}, {out.block()});

    EXPECT_EQ(1u, nodes.size());
    EXPECT_EQ(2u, edges.size());
    EXPECT_EQ(2u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto block1 = blocks.find(in.block())->second;

    CheckBlock(block1, 0, in.block(), BlockType::kInput, 1, nullptr,
               NodeVec({}));
  }
}

TEST_F(TestGraph, BlockTypeParam) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    Tensor in(Shape{1}, dev);
    Tensor mid(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    auto op = [](Context *ctx) mutable {};

    graph.AddOperation(op, {in.block()}, {in.block()});
    graph.AddOperation(op, {in.block(), mid.block()}, {out.block()});
    graph.AddOperation(op, {out.block()}, {mid.block()});

    EXPECT_EQ(3u, nodes.size());
    EXPECT_EQ(5u, edges.size());
    EXPECT_EQ(3u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto edge2 = edges[1];
    auto edge5 = edges[4];
    auto block1 = blocks.find(in.block())->second;
    auto block2 = blocks.find(mid.block())->second;

    CheckBlock(block1, 0, in.block(), BlockType::kParam, 3, edge2, NodeVec({}));
    CheckBlock(block2, 1, mid.block(), BlockType::kParam, 2, edge5,
               NodeVec({}));
  }
}

TEST_F(TestGraph, BlockTypeInter) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    Tensor in(Shape{1}, dev);
    Tensor mid(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    auto op = [](Context *ctx) mutable {};

    graph.AddOperation(op, {in.block()}, {mid.block(), out.block()});
    graph.AddOperation(op, {mid.block()}, {});
    graph.AddOperation(op, {out.block()}, {out.block()});

    EXPECT_EQ(3u, nodes.size());
    EXPECT_EQ(4u, edges.size());
    EXPECT_EQ(3u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto edge2 = edges[1];
    auto edge4 = edges[3];
    auto block2 = blocks.find(mid.block())->second;
    auto block3 = blocks.find(out.block())->second;

    CheckBlock(block2, 1, mid.block(), BlockType::kInter, 2, edge2,
               NodeVec({}));
    CheckBlock(block3, 2, out.block(), BlockType::kInter, 3, edge4,
               NodeVec({}));
  }
}

TEST_F(TestGraph, BlockTypeEnd) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    Tensor in(Shape{1}, dev);
    Tensor out1(Shape{1}, dev);
    Tensor out2(Shape{1}, dev);
    auto op = [](Context *ctx) mutable {};

    graph.AddOperation(op, {in.block()}, {out1.block()});
    graph.AddOperation(op, {}, {out2.block()});

    EXPECT_EQ(2u, nodes.size());
    EXPECT_EQ(3u, edges.size());
    EXPECT_EQ(3u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    auto edge2 = edges[1];
    auto edge3 = edges[2];
    auto block2 = blocks.find(out1.block())->second;
    auto block3 = blocks.find(out2.block())->second;

    CheckBlock(block2, 1, out1.block(), BlockType::kEnd, 1, edge2, NodeVec({}));
    CheckBlock(block3, 2, out2.block(), BlockType::kEnd, 1, edge3, NodeVec({}));
  }
}

TEST_F(TestGraph, RunGraph) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    Tensor in(Shape{1}, dev);
    Tensor mid(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    Tensor b1(Shape{1}, dev);
    Tensor b2(Shape{1}, dev);
    Tensor dx(Shape{1}, dev);
    Tensor dx1(Shape{1}, dev);
    Tensor dx2(Shape{1}, dev);
    Tensor dy1(Shape{1}, dev);
    Tensor dy2(Shape{1}, dev);
    Tensor db1(Shape{1}, dev);
    Tensor db2(Shape{1}, dev);

    // function: (in + b1) * in + b2
    auto op1 = [in, b1, mid](Context *ctx) mutable {
      singa::Add(in, b1, &mid);
    };
    auto op2 = [mid, in, out](Context *ctx) mutable {
      singa::EltwiseMult(mid, in, &out);
    };
    auto op3 = [out, b2](Context *ctx) mutable { singa::Add(out, b2, &out); };
    auto op4 = [out, dy1, db2](Context *ctx) mutable {
      dy1.CopyData(out);
      db2.CopyData(out);
    };
    auto op5 = [in, mid, dy1, dy2, dx1](Context *ctx) mutable {
      singa::EltwiseMult(dy1, in, &dy2);
      singa::EltwiseMult(dy1, mid, &dx1);
    };
    auto op6 = [dy2, dx2, db1](Context *ctx) mutable {
      dx2.CopyData(dy2);
      db1.CopyData(dy2);
    };
    auto op7 = [dx1, dx2, dx](Context *ctx) mutable {
      singa::Add(dx1, dx2, &dx);
    };

    graph.AddOperation(op1, {in.block(), b1.block()}, {mid.block()});
    graph.AddOperation(op2, {mid.block(), in.block()}, {out.block()});
    graph.AddOperation(op3, {out.block(), b2.block()}, {out.block()});
    graph.AddOperation(op4, {out.block()}, {dy1.block(), db2.block()});
    graph.AddOperation(op5, {dy1.block()}, {dy2.block(), dx1.block()});
    graph.AddOperation(op6, {dy2.block()}, {dx2.block(), db1.block()});
    graph.AddOperation(op7, {dx1.block(), dx2.block()}, {dx.block()});

    EXPECT_EQ(7u, nodes.size());
    EXPECT_EQ(14u, edges.size());
    EXPECT_EQ(12u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    in.SetValue(0);
    b1.SetValue(-1);
    b2.SetValue(2);
    graph.RunGraph();

    float dx_, db1_, db2_;
    dx.ToHost().get_value(&dx_, 1);
    db1.ToHost().get_value(&db1_, 1);
    db2.ToHost().get_value(&db2_, 1);

    EXPECT_EQ(-2, dx_);
    EXPECT_EQ(0, db1_);
    EXPECT_EQ(2, db2_);
  }
}

TEST_F(TestGraph, RunInSerial) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    Tensor in(Shape{1}, dev);
    Tensor mid(Shape{1}, dev);
    Tensor out(Shape{1}, dev);
    Tensor b1(Shape{1}, dev);
    Tensor b2(Shape{1}, dev);
    Tensor dx(Shape{1}, dev);
    Tensor dx1(Shape{1}, dev);
    Tensor dx2(Shape{1}, dev);
    Tensor dy1(Shape{1}, dev);
    Tensor dy2(Shape{1}, dev);
    Tensor db1(Shape{1}, dev);
    Tensor db2(Shape{1}, dev);

    auto op1 = [in, b1, mid](Context *ctx) mutable {
      singa::Add(in, b1, &mid);
    };
    auto op2 = [mid, in, out](Context *ctx) mutable {
      singa::EltwiseMult(mid, in, &out);
    };
    auto op3 = [out, b2](Context *ctx) mutable { singa::Add(out, b2, &out); };
    auto op4 = [out, dy1, db2](Context *ctx) mutable {
      dy1.CopyData(out);
      db2.CopyData(out);
    };
    auto op5 = [in, mid, dy1, dy2, dx1](Context *ctx) mutable {
      singa::EltwiseMult(dy1, in, &dy2);
      singa::EltwiseMult(dy1, mid, &dx1);
    };
    auto op6 = [dy2, dx2, db1](Context *ctx) mutable {
      dx2.CopyData(dy2);
      db1.CopyData(dy2);
    };
    auto op7 = [dx1, dx2, dx](Context *ctx) mutable {
      singa::Add(dx1, dx2, &dx);
    };

    graph.AddOperation(op1, {in.block(), b1.block()}, {mid.block()});
    graph.AddOperation(op2, {mid.block(), in.block()}, {out.block()});
    graph.AddOperation(op3, {out.block(), b2.block()}, {out.block()});
    graph.AddOperation(op4, {out.block()}, {dy1.block(), db2.block()});
    graph.AddOperation(op5, {dy1.block()}, {dy2.block(), dx1.block()});
    graph.AddOperation(op6, {dy2.block()}, {dx2.block(), db1.block()});
    graph.AddOperation(op7, {dx1.block(), dx2.block()}, {dx.block()});

    EXPECT_EQ(7u, nodes.size());
    EXPECT_EQ(14u, edges.size());
    EXPECT_EQ(12u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    in.SetValue(0);
    b1.SetValue(-1);
    b2.SetValue(2);
    graph.RunInSerial();

    float dx_, db1_, db2_;
    dx.ToHost().get_value(&dx_, 1);
    db1.ToHost().get_value(&db1_, 1);
    db2.ToHost().get_value(&db2_, 1);

    EXPECT_EQ(-2, dx_);
    EXPECT_EQ(0, db1_);
    EXPECT_EQ(2, db2_);
  }
}

TEST_F(TestGraph, AutoRecycle) {
  for (auto &it : devices) {
    GOUT << "Test graph on device [" << it.first << "]" << std::endl;

    auto dev = it.second;
    Graph graph(dev.get());

    auto &nodes = graph.nodes();
    auto &edges = graph.edges();
    auto &blocks = graph.blocks();
    auto &write_blocks = graph.write_blocks();

    {
      Tensor in(Shape{1}, dev);
      Tensor mid1(Shape{1}, dev);
      Tensor mid2(Shape{1}, dev);
      Tensor out(Shape{1}, dev);
      Tensor b1(Shape{1}, dev);
      Tensor b2(Shape{1}, dev);
      Tensor dx(Shape{1}, dev);
      Tensor dx1(Shape{1}, dev);
      Tensor dx2(Shape{1}, dev);
      Tensor dx3(Shape{1}, dev);
      Tensor dy1(Shape{1}, dev);
      Tensor dy2(Shape{1}, dev);
      Tensor dy3(Shape{1}, dev);
      Tensor db1(Shape{1}, dev);
      Tensor db2(Shape{1}, dev);

      // function: (in + b1) * in + (in + b2)
      auto op1 = [in, b1, mid1](Context *ctx) mutable {
        singa::Add(in, b1, &mid1);
      };
      auto op2 = [mid1, in, out](Context *ctx) mutable {
        singa::EltwiseMult(mid1, in, &out);
      };
      auto op3 = [in, b2, mid2](Context *ctx) mutable {
        singa::Add(in, b2, &mid2);
      };
      auto op4 = [out, mid2](Context *ctx) mutable {
        singa::Add(out, mid2, &out);
      };
      auto op5 = [out, dy1, dy2](Context *ctx) mutable {
        dy1.CopyData(out);
        dy2.CopyData(out);
      };
      auto op6 = [in, mid1, dy1, dy3, dx1](Context *ctx) mutable {
        singa::EltwiseMult(dy1, in, &dy3);
        singa::EltwiseMult(dy1, mid1, &dx1);
      };
      auto op7 = [dy3, dx2, db1](Context *ctx) mutable {
        dx2.CopyData(dy3);
        db1.CopyData(dy3);
      };
      auto op8 = [dy2, dx3, db2](Context *ctx) mutable {
        dx3.CopyData(dy2);
        db2.CopyData(dy2);
      };
      auto op9 = [dx1, dx2, dx](Context *ctx) mutable {
        singa::Add(dx1, dx2, &dx);
      };
      auto op10 = [dx, dx3](Context *ctx) mutable { singa::Add(dx, dx3, &dx); };

      graph.AddOperation(op1, {in.block(), b1.block()}, {mid1.block()});
      graph.AddOperation(op2, {mid1.block(), in.block()}, {out.block()});
      graph.AddOperation(op3, {in.block(), b2.block()}, {mid2.block()});
      graph.AddOperation(op4, {out.block(), mid2.block()}, {out.block()});
      graph.AddOperation(op5, {out.block()}, {dy1.block(), dy2.block()});
      graph.AddOperation(op6, {in.block(), mid1.block(), dy1.block()},
                         {dy3.block(), dx1.block()});
      graph.AddOperation(op7, {dy3.block()}, {dx2.block(), db1.block()});
      graph.AddOperation(op8, {dy2.block()}, {dx3.block(), db2.block()});
      graph.AddOperation(op9, {dx1.block(), dx2.block()}, {dx.block()});
      graph.AddOperation(op10, {dx.block(), dx3.block()}, {dx.block()});

      in.SetValue(0);
      b1.SetValue(-1);
      b2.SetValue(2);
    }

    EXPECT_EQ(10u, nodes.size());
    EXPECT_EQ(21u, edges.size());
    EXPECT_EQ(15u, blocks.size());
    EXPECT_EQ(1u, write_blocks.size());

    graph.RunGraph();

    auto &begin_nodes = graph.begin_nodes();
    auto &next_nodes = graph.next_nodes();
    auto &free_blocks = graph.free_blocks();

    EXPECT_FALSE(graph.dirty());
    EXPECT_EQ(nodes[0], begin_nodes[0]);
    EXPECT_EQ(nodes[2], begin_nodes[1]);
    EXPECT_EQ(nodes[1], next_nodes[0][0]);
    EXPECT_EQ(nodes[3], next_nodes[1][0]);
    EXPECT_EQ(nodes[4], next_nodes[3][0]);
    EXPECT_EQ(nodes[5], next_nodes[4][0]);
    EXPECT_EQ(nodes[7], next_nodes[4][1]);
    EXPECT_EQ(nodes[6], next_nodes[5][0]);
    EXPECT_EQ(nodes[8], next_nodes[6][0]);
    EXPECT_EQ(nodes[9], next_nodes[8][0]);

    CheckFreeBlocks(0, blocks, free_blocks[0], IntVec({}));
    CheckFreeBlocks(1, blocks, free_blocks[1], IntVec({}));
    CheckFreeBlocks(2, blocks, free_blocks[2], IntVec({}));
    CheckFreeBlocks(3, blocks, free_blocks[3], IntVec({5}));
    CheckFreeBlocks(4, blocks, free_blocks[4], IntVec({3}));
    CheckFreeBlocks(5, blocks, free_blocks[5], IntVec({2, 6}));
    CheckFreeBlocks(6, blocks, free_blocks[6], IntVec({8, 11}));
    CheckFreeBlocks(7, blocks, free_blocks[7], IntVec({7, 13}));
    CheckFreeBlocks(8, blocks, free_blocks[8], IntVec({9, 10}));
    CheckFreeBlocks(9, blocks, free_blocks[9], IntVec({12, 14}));

    // in 0 b1 1 mid1 2 out 3 b2 4
    // mid2 5 dy1 6 dy2 7 dy3 8 dx1 9
    // dx2 10 db1 11 dx3 12 db2 13 dx 14
    bool state[15] = {true,  true,  false, false, true,  false, false, false,
                      false, false, false, false, false, false, false};

    for (auto it : blocks) {
      int id = it.second->id();
      EXPECT_EQ(state[id], it.first->initialized())
          << "The memory of the block[" << id << "] is not properly recycled"
          << std::endl;
    }
  }
}
