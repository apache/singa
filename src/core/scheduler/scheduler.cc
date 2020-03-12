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

#include "singa/core/device.h"
#include "singa/utils/safe_queue.h"

namespace singa {

Node::Node(function<void(Context*)>&& op) : op_(std::move(op)) {}

void Node::AddInEdge(Edge *in_edge) {
  in_edges_.push_back(in_edge);
}

void Node::AddOutEdge(Edge *out_edge) {
  out_edges_.push_back(out_edge);
}

Edge::Edge(Block *blk, Node *src_node, Node *dst_node)
  : blk_(blk), src_node_(src_node), dst_node_(dst_node) {}

void Edge::SetBlock(Block* blk) { blk_ = blk; }

void Edge::SetSrcNode(Node* src_node) { src_node_ = src_node; }

void Edge::SetDstNode(Node* dst_node) { dst_node_ = dst_node; }

void Graph::Run() {
  SafeQueue<int> node_queue;
  std::vector<int> node_ref;

  /*
  for (size_t i = 0; i < nodes_.size(); ++i) {
    printf("OP[%2d]: ", i);
    printf("Inputs: ");
    auto node = nodes_[i];
    for (size_t j = 0; j < node->in_edges_.size(); ++j) {
      printf("%d\t", blk2index_[node->in_edges_[j]->blk_]);
    }
    for (size_t j = node->in_edges_.size(); j < 3; ++j) {
      printf("\t");
    }
    printf("Outputs: ");
    for (size_t j = 0; j < node->out_edges_.size(); ++j) {
      printf("%d\t", blk2index_[node->out_edges_[j]->blk_]);
    }
    printf("\n");
  }
  */

  // init node ref
  node_ref.resize(nodes_.size());
  for (int i = 0; i < nodes_.size(); ++i) {
    node_ref[i] = nodes_[i]->in_edges_.size();
  }

  // find all input edges and decrease ref count of nodes
  for (int i = 0; i < edges_.size(); ++i) {
    if (!edges_[i]->src_node_) {
      Node *node = edges_[i]->dst_node_;
      int nodeId = node->id_;
      node_ref[nodeId] -= 1;
    }
  }

  // activate nodes
  for (int i = 0; i < node_ref.size(); ++i) {
    if (node_ref[i] == 0) {
      node_queue.Push(i);
      // printf("push node[%2d]\n", i);
    }
  }

  // run graph
  while (node_queue.Size()) {
    // step 1: pop the first element, get the node corresponding to the index
    int curIndex = -1;
    node_queue.Pop(curIndex);
    Node *curNode = nodes_[curIndex];
    printf("pop node[%2d]\n", curIndex);

    // step 2: execute the operation
    device_->DoExec(std::move(curNode->op_), 0);

    // step 3: release some blocks' data that won't be used later
    for (size_t i = 0; i < curNode->in_edges_.size(); ++i) {
      Edge *edge = curNode->in_edges_[i];
      Block *blk = edge->blk_;
      if (last_edge_[blk] == edge) {
	if (last_node_[blk] != curNode && edge->src_node_) {
	  blk->free_data();
	  printf("free block[%2d]\n", blk2index_[blk]);
	}
      }
    }

    // step 4: decrease ref count of nodes and activate nodes
    for (size_t i = 0; i < curNode->out_edges_.size(); ++i) {
      Edge *edge = curNode->out_edges_[i];
      Node *nextNode = edge->dst_node_;

      if (nextNode) {
	int nodeId = nextNode->id_;
	node_ref[nodeId] -= 1;
	if (node_ref[nodeId] <= 0) {
	  node_queue.Push(nodeId);
	  // printf("push node[%2d]\n", nodeId);
	}
      }
    }
  }

  // printf("end\n");
}

void Graph::AddOperation(function<void(Context*)>&& op, const BlockSet &read_blocks, const BlockSet &write_blocks) {
  // create new node
  Node *node = new Node(std::move(op));

  // create edges for read_blocks
  for (size_t i = 0; i < read_blocks.size(); ++i) {
    Block *blk = read_blocks[i];
    Edge *edge = nullptr;

    auto it = last_node_.find(blk);
    if (it == last_node_.end()) {
      edge = new Edge(blk, nullptr, node);
      last_node_[blk] = nullptr;
    } else {
      edge = new Edge(blk, it->second, node);
      it->second->AddOutEdge(edge);
    }

    edge->id_ = edges_.size();

    node->AddInEdge(edge);
    last_edge_[blk] = edge;
    edges_.push_back(edge);

    if (blk2index_.find(blk) == blk2index_.end()) {
      blk2index_[blk] = blk2index_.size();
      last_node_[blk] = nullptr;
    }
  }

  // update last node for write_blocks
  for (size_t i = 0; i < write_blocks.size(); ++i) {
    Block *blk = write_blocks[i];

    last_node_[blk] = node;

    if (blk2index_.find(blk) == blk2index_.end()) {
      blk2index_[blk] = blk2index_.size();
      last_edge_[blk] = nullptr;
    }

  }

  // add node into nodes
  node->id_ = nodes_.size();
  nodes_.push_back(node);
}

}  // namespace singa
