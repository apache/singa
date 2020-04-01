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

#include "singa/core/scheduler.h"

#include <functional>
#include <thread>
#include <unordered_set>

#include "singa/core/device.h"
#include "singa/utils/safe_queue.h"

namespace singa {

void Node::AddInEdge(Edge *in_edge) { in_edges_.push_back(in_edge); }

void Node::AddOutEdge(Edge *out_edge) { out_edges_.push_back(out_edge); }

void Edge::SetBlock(Block *blk) { blk_ = blk; }

void Edge::SetSrcNode(Node *src_node) { src_node_ = src_node; }

void Edge::SetDstNode(Node *dst_node) { dst_node_ = dst_node; }

Graph::Graph(Device *device) : device_(device) {}

Graph::~Graph() { Reset(); }

Node *Graph::node(const size_t idx) const {
  CHECK_LT(idx, nodes_.size());
  return nodes_[idx];
}

Edge *Graph::edge(const size_t idx) const {
  CHECK_LT(idx, edges_.size());
  return edges_[idx];
}

BlkInfo *Graph::block(Block *blk) const {
  auto it = blocks_.find(blk);
  CHECK(it != blocks_.end());
  return it->second;
}

Block *Graph::write_block(const size_t idx) const {
  CHECK_LT(idx, write_blocks_.size());
  return write_blocks_[idx];
}

Node *Graph::begin_node(const size_t idx) const {
  CHECK_LT(idx, begin_nodes_.size());
  return begin_nodes_[idx];
}

const NodeVec &Graph::next_nodes(const size_t idx) const {
  CHECK_LT(idx, next_nodes_.size());
  return next_nodes_[idx];
}

const BlockVec &Graph::free_blocks(const size_t idx) const {
  CHECK_LT(idx, free_blocks_.size());
  return free_blocks_[idx];
}

void Graph::Reset() {
  for (auto it : nodes_) {
    delete it;
  }
  nodes_.clear();

  for (auto it : edges_) {
    delete it;
  }
  edges_.clear();

  for (auto it : blocks_) {
    delete it.second;
  }
  blocks_.clear();
}

void Graph::Debug() {
  for (size_t i = 0; i < nodes_.size(); ++i) {
    printf("OP[%2lu]: ", i);
    printf("Inputs: ");
    auto node = nodes_[i];
    for (size_t j = 0; j < node->in_edges_.size(); ++j) {
      printf("%d\t", blocks_[node->in_edges_[j]->blk_]->id_);
    }
    for (size_t j = node->in_edges_.size(); j < 3; ++j) {
      printf("\t");
    }
    printf("Outputs: ");
    for (size_t j = 0; j < node->out_edges_.size(); ++j) {
      printf("%d\t", blocks_[node->out_edges_[j]->blk_]->id_);
    }
    printf("\n");
  }

  for (auto it : blocks_) {
    auto blkInfo = it.second;
    printf("Block[%2d]: addr[%p] graph_ref[%d] ref_count[%d] ", blkInfo->id_,
           blkInfo->blk_, blkInfo->graph_ref_, it.first->ref_count());
    switch (blkInfo->type_) {
      case BlockType::kInput:
        printf("type[input] ");
        break;
      case BlockType::kParam:
        printf("type[param] ");
        break;
      case BlockType::kInter:
        printf("type[inter] ");
        break;
      case BlockType::kEnd:
        printf("type[_end_] ");
        break;
      default:
        break;
    }
    int id = -1;
    if (blkInfo->write_node_) {
      id = blkInfo->write_node_->id_;
    }
    printf(" write_node[%2d]", id);
    id = -1;
    if (blkInfo->last_node_) {
      id = blkInfo->last_node_->id_;
    }
    printf(" write_node[%2d]", id);
    printf("\n");
  }
}

void Graph::RunGraph() {
  if (dirty_) Analysis();

  SafeQueue<Node *> node_queue;

  // activate nodes
  for (auto it : begin_nodes_) {
    node_queue.Push(it);
  }

  // run graph
  while (node_queue.Size()) {
    // step 1: pop the first element, get the node corresponding to the index
    Node *curNode = nullptr;
    node_queue.Pop(curNode);
    int curIndex = curNode->id_;

    // step 2: execute the operation
    device_->DoExec(std::move(curNode->op_), 0);

    // step 3: release some blocks' data that won't be used later
    for (auto it : free_blocks_[curIndex]) {
      it->free_data();
    }

    /*
    if (free_blocks_[curIndex].size()) {
      CBData *cb_data = new CBData(this, curNode);
      cudaStreamAddCallback(device_->ctx_.stream, Graph::Callback, (void
    *)(cb_data), 0);
    }
    */

    // step 4: activate the following nodes
    for (auto it : next_nodes_[curIndex]) {
      node_queue.Push(it);
    }
  }
}

void Graph::RunInSerial() {
  if (dirty_) Analysis();

  for (size_t i = 0; i < nodes_.size(); ++i) {
    Node *curNode = nodes_[i];

    // step 1: execute the operation
    device_->DoExec(std::move(curNode->op_), 0);

    // step 2: release some blocks' data that won't be used later
    for (auto it : free_blocks_[i]) {
      it->free_data();
    }

    /*
    // Wait for calculation to complete and then recyle the data
    CBData *cb_data = new CBData(this, curNode);
    CHECK(cudaStreamAddCallback(device_->ctx_.stream, Graph::Callback, (void
    *)(cb_data), 0));
    */
  }
}

void Graph::AddOperation(OpFunc &&op, const BlockVec &read_blocks,
                         const BlockVec &write_blocks) {
  dirty_ = true;

  if (read_blocks.size() == 0 && write_blocks.size() == 0) {
    AddSyncOp(std::move(op));
    return;
  }

  // create new node
  Node *node = new Node(nodes_.size(), std::move(op));

  // create a set to determine if there is a loop
  std::unordered_set<Block *> circle;

  // create edges for read_blocks
  for (size_t i = 0; i < read_blocks.size(); ++i) {
    Block *blk = read_blocks[i];
    Edge *edge = nullptr;
    BlkInfo *blkInfo = nullptr;

    auto it = blocks_.find(blk);
    if (it == blocks_.end()) {
      edge = new Edge(edges_.size(), blk, nullptr, node);
      blkInfo = new BlkInfo(blocks_.size(), blk, BlockType::kInput);
      blocks_[blk] = blkInfo;
    } else {
      blkInfo = it->second;
      if (blkInfo->type_ == BlockType::kEnd) {
        blkInfo->type_ = BlockType::kInter;
      }

      Node *write_node = blkInfo->write_node_;
      edge = new Edge(edges_.size(), blk, write_node, node);
      if (write_node) {
        write_node->AddOutEdge(edge);
      }
    }

    circle.insert(blk);
    blkInfo->graph_ref_ += 1;
    blkInfo->last_node_ = node;

    node->AddInEdge(edge);
    edges_.push_back(edge);
  }

  // update last node for write_blocks
  for (size_t i = 0; i < write_blocks.size(); ++i) {
    Block *blk = write_blocks[i];
    BlkInfo *blkInfo = nullptr;

    auto it = blocks_.find(blk);
    if (it == blocks_.end()) {
      blkInfo = new BlkInfo(blocks_.size(), blk, BlockType::kEnd);
      blocks_[blk] = blkInfo;
    } else {
      blkInfo = it->second;
      if (blkInfo->type_ == BlockType::kInput) {
        blkInfo->type_ = BlockType::kParam;
      }
    }

    if (circle.find(blk) == circle.end()) {
      circle.insert(blk);
      blkInfo->graph_ref_ += 1;
    }

    blkInfo->write_node_ = node;
    blkInfo->last_node_ = node;
  }

  write_blocks_ = write_blocks;

  // add node into nodes
  nodes_.push_back(node);
}

void Graph::Analysis() {
  begin_nodes_.clear();
  next_nodes_.resize(nodes_.size());
  free_blocks_.resize(nodes_.size());

  // init node ref
  std::vector<int> node_ref_;
  node_ref_.resize(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i) {
    node_ref_[i] = nodes_[i]->in_edges_.size();
  }

  // find all input edges and decrease ref count of nodes
  for (size_t i = 0; i < edges_.size(); ++i) {
    Node *src_node = edges_[i]->src_node_;
    if (!src_node) {
      Node *node = edges_[i]->dst_node_;
      int nodeId = node->id_;
      node_ref_[nodeId] -= 1;
    }
  }

  // activate nodes
  SafeQueue<Node *> node_queue;
  for (size_t i = 0; i < node_ref_.size(); ++i) {
    if (node_ref_[i] == 0) {
      begin_nodes_.push_back(nodes_[i]);
      node_queue.Push(nodes_[i]);
    }
  }

  // run graph
  while (node_queue.Size()) {
    // step 1: pop the first element, get the node corresponding to the index
    Node *curNode = nullptr;
    node_queue.Pop(curNode);
    int curIndex = curNode->id_;

    // step 2: release some blocks' data that won't be used later
    free_blocks_[curIndex].clear();
    for (size_t i = 0; i < curNode->in_edges_.size(); ++i) {
      Edge *edge = curNode->in_edges_[i];
      Block *blk = edge->blk_;
      BlkInfo *blkInfo = blocks_[blk];
      if (blkInfo->last_node_ == curNode && blkInfo->write_node_ != curNode) {
        BlockType type = blkInfo->type_;
        if (type == BlockType::kInter &&
            blkInfo->graph_ref_ == blk->ref_count()) {
          free_blocks_[curIndex].push_back(blk);
        }
      }
    }

    // step 3: decrease ref count of nodes and activate nodes
    next_nodes_[curIndex].clear();
    for (size_t i = 0; i < curNode->out_edges_.size(); ++i) {
      Edge *edge = curNode->out_edges_[i];
      Node *nextNode = edge->dst_node_;

      if (nextNode) {
        int nodeId = nextNode->id_;
        node_ref_[nodeId] -= 1;
        if (node_ref_[nodeId] <= 0) {
          node_queue.Push(nextNode);
          next_nodes_[curIndex].push_back(nextNode);
        }
      }
    }
  }

  dirty_ = false;
}

void Graph::FreeLoop() {
  int id = 0;
  for (;;) {
    free_queue_.Pop(id);
    if (id == -1) {
      break;
    } else {
      for (auto it : free_blocks_[id]) {
        it->free_data();
      }
    }
  }
}

void Graph::AddSyncOp(function<void(Context *)> &&op) {
  // create new node
  Node *node = new Node(nodes_.size(), std::move(op));

  for (size_t i = 0; i < write_blocks_.size(); ++i) {
    Block *blk = write_blocks_[i];
    BlkInfo *blkInfo = blocks_[blk];

    if (blkInfo->type_ == BlockType::kEnd) {
      blkInfo->type_ = BlockType::kInter;
    }

    Node *write_node = blkInfo->write_node_;
    Edge *edge = new Edge(edges_.size(), blk, write_node, node);
    if (write_node) {
      write_node->AddOutEdge(edge);
    }

    // fake edges, no need to add the graph ref
    blkInfo->last_node_ = node;
    blkInfo->write_node_ = node;

    node->AddInEdge(edge);
    edges_.push_back(edge);
  }

  // add node into nodes
  nodes_.push_back(node);
}

/*
void CUDART_CB Graph::Callback(cudaStream_t stream, cudaError_t status,
                               void *data) {
  CBData *cb_data = (CBData *)data;
  Graph *graph = cb_data->graph_;
  graph->free_queue_.Push(cb_data->node_->id_);
  delete cb_data;
}
*/

}  // namespace singa
