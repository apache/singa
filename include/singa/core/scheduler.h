/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SINGA_CORE_SCHEDULER_H_
#define SINGA_CORE_SCHEDULER_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "singa/core/common.h"
#include "singa/utils/safe_queue.h"

using std::function;
using std::string;
using std::unordered_map;
using std::vector;

namespace singa {

class Node;
class Edge;
class Graph;
class Device;
class BlkInfo;

typedef std::vector<Node *> NodeVec;
typedef std::vector<Edge *> EdgeVec;
typedef std::vector<Block *> BlockVec;
typedef std::function<void(Context *)> OpFunc;
typedef std::unordered_map<Block *, BlkInfo *> Blk2InfoMap;

enum BlockType { kUnknow, kInput, kParam, kInter, kEnd };

class Node {
 public:
  Node(int id, OpFunc &&op, string op_name)
      : id_(id), op_(std::move(op)), op_name_(op_name) {}

  void AddInEdge(Edge *in_edge);
  void AddOutEdge(Edge *out_edge);

  // getters of Node
  int id() const { return id_; }
  string op_name() const { return op_name_; }
  const EdgeVec &in_edges() const { return in_edges_; }
  const EdgeVec &out_edges() const { return out_edges_; }
  float time_elapsed() const { return time_elapsed_; }

  // time profiling
  void time_elapsed_inc(float time) { time_elapsed_ += time; }

 private:
  friend Graph;

  int id_;
  OpFunc op_;
  EdgeVec in_edges_;
  EdgeVec out_edges_;

  string op_name_;
  float time_elapsed_ = 0;

#ifdef USE_CUDA
  cudaEvent_t start_;
  cudaEvent_t end_;
  friend class CudaGPU;
#endif  // USE_CUDA
};

class Edge {
 public:
  Edge(int id, Block *blk, Node *src_node, Node *dst_node)
      : id_(id), blk_(blk), src_node_(src_node), dst_node_(dst_node) {}

  void SetBlock(Block *blk);
  void SetSrcNode(Node *src_node);
  void SetDstNode(Node *dst_node);

  // getters of Edge
  int id() const { return id_; }
  Block *block() const { return blk_; }
  Node *src_node() const { return src_node_; }
  Node *dst_node() const { return dst_node_; }

 private:
  friend Graph;

  int id_;
  Block *blk_;
  Node *src_node_;
  Node *dst_node_;
};

class BlkInfo {
 public:
  BlkInfo(int id, Block *blk, BlockType type = BlockType::kUnknow)
      : id_(id), blk_(blk), type_(type), graph_ref_(0), write_edge_(nullptr) {}

  // getters of BlkInfo
  int id() const { return id_; }
  Block *block() const { return blk_; }
  BlockType type() const { return type_; }
  int graph_ref() const { return graph_ref_; }
  Edge *write_edge() const { return write_edge_; }
  const NodeVec &used_nodes() const { return used_nodes_; }
  Node *used_node(const size_t idx) const;

 private:
  friend Graph;

  int id_;
  Block *blk_;
  BlockType type_;
  int graph_ref_;
  Edge *write_edge_;    // the edge of last node that writes data into blk
  NodeVec used_nodes_;  // the nodes that use this block(in order of execution)
};

class Graph {
 public:
  struct CBData {
    Graph *graph_;
    Node *node_;

    CBData(Graph *graph, Node *node) : graph_(graph), node_(node) {}
  };

  ~Graph();
  Graph(Device *device);

  void Reset();
  void Debug();
  void RunGraph();
  void RunInSerial();
  void AddOperation(OpFunc &&op, const BlockVec &read_blocks,
                    const BlockVec &write_blocks, string op_name = "no_name");

  // getters of Graph
  const NodeVec &nodes() const { return nodes_; }
  const EdgeVec &edges() const { return edges_; }
  const Blk2InfoMap &blocks() const { return blocks_; }

  const BlockVec &write_blocks() const { return write_blocks_; }

  bool dirty() const { return dirty_; }
  const NodeVec &begin_nodes() const { return begin_nodes_; }
  const std::vector<NodeVec> &next_nodes() const { return next_nodes_; }
  const std::vector<BlockVec> &free_blocks() const { return free_blocks_; }
  int iteration() const { return iteration_; }

  Node *node(const size_t idx) const;
  Edge *edge(const size_t idx) const;
  BlkInfo *block(Block *blk) const;

  Block *write_block(const size_t idx) const;

  Node *begin_node(const size_t idx) const;
  const NodeVec &next_nodes(const size_t idx) const;
  const BlockVec &free_blocks(const size_t idx) const;

  void PrintTimeProfiling();
  void EvaluateTimeElapsed();

 private:
  void Analyze();
  void FreeLoop();
  void AnalyzeNodes();
  void AnalyzeEdges();
  void AddSyncOp(function<void(Context *)> &&op, string op_name = "no_name");
  void TimeProfilingDoExec(Node *curNode);
  void TakeStartTime();
  void step() { iteration_++; }
  void time_elapsed_inc(float time) { time_elapsed_ += time; }

  // static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status,
  //                                void *data);

 private:
  Device *device_;

  // nodes, edges and blocks included in the calculation graph
  NodeVec nodes_;
  EdgeVec edges_;
  Blk2InfoMap blocks_;

  // Blocks written by the last operation, used for sync op
  BlockVec write_blocks_;

  // Calculation graph analysis
  bool dirty_ = false;
  bool in_serial_ = false;
  NodeVec begin_nodes_;
  std::vector<NodeVec> next_nodes_;
  std::vector<BlockVec> free_blocks_;
  int iteration_ = 0;
  float time_elapsed_ = 0;
  std::chrono::high_resolution_clock::time_point t_start_;

  SafeQueue<int> free_queue_;

};

/// Scheduling Tensor operations with dependency detection.
class Scheduler {};

}  // namespace singa
#endif  // SINGA_CORE_SCHEDULER_H_
