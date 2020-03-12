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
class EdgeType;
class Device;

class Node {
public:
  Node(std::function<void(Context*)>&& op);

  void AddInEdge(Edge *in_edge);
  void AddOutEdge(Edge *out_edge);

private:
  friend Graph;

  int id_;
  std::function<void(Context*)> op_;
  std::vector<Edge *> in_edges_;
  std::vector<Edge *> out_edges_;
};

class Edge {
public:
  Edge(Block *blk, Node *src_node, Node *dst_node);

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

class Graph {
public:
  typedef std::vector<Block*> BlockSet;

  Graph(Device *device) : device_(device) {}

  void Run();
  void Reset();
  void AddOperation(function<void(Context*)>&& op,
		    const BlockSet &read_blocks, const BlockSet &write_blocks);
  
private:
  Device *device_;
  std::vector<Node *> nodes_;
  std::vector<Edge *> edges_;
  std::unordered_map<Block *, Node *> last_node_;
  std::unordered_map<Block *, Edge *> last_edge_;
  std::unordered_map<Block *, int> blk2index_;
};


/// Scheduling Tensor operations with dependency detection.
class Scheduler {};

}  // namespace singa
#endif  // SINGA_CORE_SCHEDULER_H_
