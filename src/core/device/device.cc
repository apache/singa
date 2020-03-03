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

#include "singa/core/device.h"

namespace singa {
Device::Device(int id, int num_executors)
    : id_(id), num_executors_(num_executors) {
  // TODO(wangwei) create scheduler and vm.
  host_ = defaultDevice;
}

void Device::Exec(function<void(Context*)>&& fn,
                  const vector<Block*> read_blocks,
                  const vector<Block*> write_blocks, bool use_rand_generator) {
  if (buffer_flag_== true)
  {
    size_t op_index = graph_.nodes.size();
    // printf("Exec Op[%2d]\n", op_index);

    OpNode *opNode = new OpNode();
    opNode->op = std::move(fn);
    graph_.node2index[opNode] = op_index;
    graph_.nodes.push_back(opNode);

    auto &blk2index = graph_.blk2index;
    for (size_t i = 0; i < read_blocks.size(); ++i) {
      auto blk = read_blocks[i];

      auto it = blk2index.find(blk);
      if (it != blk2index.end()) {
	Edge *edge = graph_.edges[it->second];
	edge->dst_nodes.push_back(opNode);
	if (edge->type == EdgeType::kEnd) {
	  edge->type = EdgeType::kInter;
	}
      } else {
	Edge * edge = new Edge();
	edge->type = EdgeType::kInput;
	edge->block = blk;
	edge->indegree = 0;
	edge->dst_nodes.push_back(opNode);
	graph_.blk2index[blk] = graph_.edges.size();
	graph_.edges.push_back(edge);
      }
    }

    for (size_t i = 0; i < write_blocks.size(); ++i) {
      auto blk = write_blocks[i];

      auto it = blk2index.find(blk);
      if (it != blk2index.end()) {
	Edge *edge = graph_.edges[it->second];
	edge->indegree += 1;
	if (edge->type == EdgeType::kInput) {
	  edge->type = EdgeType::kParam;
	}
      } else {
	Edge * edge = new Edge();
	edge->type = EdgeType::kEnd;
	edge->block = blk;
	edge->indegree = 1;
	graph_.blk2index[blk] = graph_.edges.size();
	graph_.edges.push_back(edge);
      }
    }

    opNode->read_blocks = std::move(read_blocks);
    opNode->write_blocks = std::move(write_blocks);
  }
  else
  {
    // printf("immediately ops\n");
    DoExec(std::move(fn), 0);
  }
}

void Device::ExecBuffOps() {
  bool previous_state = buffer_flag_;
  buffer_flag_ = false;

  auto &nodes = graph_.nodes;
  auto &edges = graph_.edges;
  auto &blk2index = graph_.blk2index;
  auto &node2index = graph_.node2index;

  /*
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto node = nodes[i];
    printf("OP[%2d]: ", i);
    printf("Inputs: ");
    auto &read_blocks = node->read_blocks;
    for (size_t j = 0; j < read_blocks.size(); ++j) {
      printf("%d\t", blk2index[read_blocks[j]]);
    }
    for (size_t j = read_blocks.size(); j < 3; ++j) {
      printf("\t");
    }
    printf("Outputs: ");
    auto &write_blocks = node->write_blocks;
    for (size_t j = 0; j < write_blocks.size(); ++j) {
      printf("%d\t", blk2index[write_blocks[j]]);
    }
    printf("\n");
  }

  for (size_t i = 0; i < edges.size(); ++i) {
    auto edge = edges[i];
    printf("Edge[%2d]: block[%#x] ", i, edge->block);
    switch (edge->type) {
      case EdgeType::kInput: printf("type[input] "); break;
      case EdgeType::kParam: printf("type[param] "); break;
      case EdgeType::kInter: printf("type[inter] "); break;
      case EdgeType::kEnd: printf("type[_end_] "); break;
      default: break;
    }
    printf("OPs: \t");
    for (size_t j = 0; j < edge->dst_nodes.size(); ++j) {
      printf("%2d\t", node2index[edge->dst_nodes[j]]);
    }
    printf("\n");
  }
  */

  SafeQueue<int> node_queue;
  std::vector<int> node_ref; 
  std::vector<int> edge_ref;
  std::vector<int> edge_wait;
  std::vector<bool> edge_state;

  node_ref.resize(nodes.size());
  edge_ref.resize(edges.size());
  edge_wait.resize(edges.size());
  edge_state.resize(edges.size());

  for (size_t i = 0; i < nodes.size(); ++i) {
    node_ref[i] = nodes[i]->read_blocks.size();
  }

  for (size_t i = 0; i < edges.size(); ++i) {
    edge_ref[i] = edges[i]->indegree;
    edge_wait[i] = edges[i]->dst_nodes.size();
    if (edges[i]->type == EdgeType::kInput || edges[i]->type == EdgeType::kParam) {
      edge_state[i] = true;
      auto &dst_nodes = edges[i]->dst_nodes;
      for (size_t j = 0; j < dst_nodes.size(); ++j) {
	size_t index = node2index[dst_nodes[j]];
	node_ref[index] -= 1;

	bool circle = false;
	for (auto it : dst_nodes[j]->write_blocks) {
	  if (it == edges[i]->block) {
	    edge_state[i] = false;
	    circle = true;
	    break;
	  }
	}

	if (circle) {
	  break;
	}

      }
    } else {
      edge_state[i] = false;
    }
  }

  std::vector<std::vector<int> > anss;
  std::vector<int> ans;
  for (size_t i = 0; i < node_ref.size(); ++i) {
    if (node_ref[i] == 0) {
      node_queue.Push(i);
      ans.push_back(i);
    }
  }
  anss.push_back(ans);

  int de_count = 0;
  while (node_queue.Size()) {
    // step 1: pop the first element and execte the operation
    int curIndex = -1;
    node_queue.Pop(curIndex);
    OpNode *curNode = nodes[curIndex];
    // printf("pop op[%2d]\n", curIndex);
    DoExec(std::move(curNode->op), 0);

    // step 2: decrease the ref count of the input tensors
    for (size_t i = 0; i < curNode->read_blocks.size(); ++i) {
      size_t edge_index = blk2index[curNode->read_blocks[i]];
      edge_wait[edge_index] -= 1;
      if (edge_wait[edge_index] == 0) {
	++de_count;
	// printf("deallocate block[%2d]\n", edge_index);
      }
    }

    // step 3: activate output blocks
    std::vector<int> ans;
    for (size_t i = 0; i < curNode->write_blocks.size(); ++i) {
      Block *write_block = curNode->write_blocks[i];
      size_t edge_index = blk2index[write_block];
      edge_state[edge_index] = true;

      auto &dst_nodes = edges[edge_index]->dst_nodes;
      for (size_t j = 0; j < dst_nodes.size(); ++j) {
	OpNode *dst_node = dst_nodes[j];
	size_t node_index = node2index[dst_node];

	if (node_ref[node_index] <= 0) {
	  continue;
	}

	bool circle = false;
	for (auto it : dst_node->write_blocks) {
	  if (it == write_block) {
	    circle = true;
	    break;
	  }
	}

	node_ref[node_index] -= 1;
	if (node_ref[node_index] != 0) {
	  if (circle) {
	    break;
	  } else {
	    continue;
	  }
	}

	node_queue.Push(node_index);
	ans.push_back(node_index);
	// printf("push op[%2d]\n", node_index);

	if (circle) {
	  edge_ref[edge_index] = false;
	  break;
	}
      }
    }
    if (!ans.empty()) anss.push_back(ans);
  }

  /*
  printf("total released count: %d/%d(without tensor of type _end_)\n", de_count, edges.size());
  for (int i = 0; i < anss.size(); i++) {
    printf("group %d: ", i);
    for (auto it : anss[i]) {
      printf("%d ", it);
    }
    printf("\n");
  }
  */

  buffer_flag_ = previous_state;
}

// Todo(Wangwei) Get Block From The Memory manager
Block* Device::NewBlock(int size) {
  CHECK_GE(size, 0)
      << "size is negative, could be caused by the type cast "
      << "from size_t to int. In that case, the size is too large.";
  if (size > 0) {
    void* ptr = Malloc(size);
    return new Block(ptr, size);
  } else {
    return nullptr;
  }
}

// TODO(wangwei) return Block to the memory manager
void Device::FreeBlock(Block* block) {
  if (block != nullptr) {
    Free(block->mutable_data());
    delete block;
  }
}

void Device::CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                            CopyDirection direct, int dst_offset,
                            int src_offset) {
  this->Exec(
      [this, dst, src, nBytes, direct, dst_offset, src_offset](Context* ctx) {
        this->CopyToFrom(
            reinterpret_cast<char*>(dst->mutable_data()) + dst_offset,
            reinterpret_cast<const char*>(src->data()) + src_offset, nBytes,
            direct, ctx);
      },
      {src}, {dst});
}

void Device::CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes,
                                 size_t dst_offset) {
  auto direct = lang_ == kCpp ? kHostToHost : kHostToDevice;
  void* dstptr = reinterpret_cast<char*>(dst->mutable_data()) + dst_offset;
  Exec([this, dstptr, src, nBytes,
        direct](Context* ctx) { CopyToFrom(dstptr, src, nBytes, direct, ctx); },
       {}, {dst});
}
void Device::Sync() {}
}  // namespace singa
