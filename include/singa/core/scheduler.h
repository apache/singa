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

#include <functional>
#include <unordered_map>
#include <vector>

#include "singa/core/common.h"

using std::function;
using std::unordered_map;
using std::vector;

namespace singa {

class Node;
class Edge;
class Graph;
class Device;

enum BlockType { kUnknow, kInput, kParam, kInter, kEnd };

class Node {
 public:
  Node(int id, std::function<void(Context *)> &&op)
      : id_(id), op_(std::move(op)) {}

  void AddInEdge(Edge *in_edge);
  void AddOutEdge(Edge *out_edge);

 private:
  friend Graph;

  int id_;
  std::function<void(Context *)> op_;
  std::vector<Edge *> in_edges_;
  std::vector<Edge *> out_edges_;
};

class Edge {
 public:
  Edge(int id, Block *blk, Node *src_node, Node *dst_node)
      : id_(id), blk_(blk), src_node_(src_node), dst_node_(dst_node) {}

  void SetBlock(Block *blk);
  void SetSrcNode(Node *src_node);
  void SetDstNode(Node *dst_node);

 private:
  friend Graph;

  int id_;
  Block *blk_;
  Node *src_node_;
  Node *dst_node_;
};

class BlockInfo {
 public:
  BlockInfo(int id, Block *blk, BlockType type = BlockType::kUnknow)
      : id_(id),
        blk_(blk),
        type_(type),
        graph_ref_(0),
        write_node_(nullptr),
        last_node_(nullptr) {}

 private:
  friend Graph;

  int id_;
  Block *blk_;
  BlockType type_;
  size_t graph_ref_;
  Node *write_node_;  // last node that writes the block
  Node *last_node_;   // last node that uses the block
};

class Graph {
 public:
  typedef std::vector<Block *> BlockSet;

  ~Graph();
  Graph(Device *device) : device_(device) {}

  void Reset();
  void Debug();
  void RunGraph();
  void RunInSerial();
  void AddOperation(function<void(Context *)> &&op, const BlockSet &read_blocks,
                    const BlockSet &write_blocks);

 private:
  Device *device_;
  std::vector<Node *> nodes_;
  std::vector<Edge *> edges_;
  std::unordered_map<Block *, BlockInfo *> blocks_;
};

/// Scheduling Tensor operations with dependency detection.
class Scheduler {};

}  // namespace singa
#endif  // SINGA_CORE_SCHEDULER_H_
