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

bool Device::lazy_alloc_ = true;

Device::Device(int id, int num_executors)
    : id_(id), num_executors_(num_executors) {
  // TODO(wangwei) create scheduler and vm.
  host_ = defaultDevice;
  graph_ = new Graph(this);
}

Device::~Device() {
  if (graph_) {
    delete graph_;
  }
}

void Device::Reset() {
  // Sync the device to finished the current calculation
  graph_enabled_ = false;
  Sync();

  // Reset Seed
  seed_ = std::chrono::system_clock::now().time_since_epoch().count();
  SetRandSeed(seed_);

  // Reset Graph
  graph_->Reset();

  // Others
  verbosity_ = 0;
  skip_iteration_ = 5;
}

void Device::Exec(function<void(Context*)>&& fn,
                  const vector<Block*> read_blocks,
                  const vector<Block*> write_blocks, string op_name,
                  bool use_rand_generator) {
  if (graph_enabled_ == true) {
    graph_->AddOperation(std::move(fn), read_blocks, write_blocks, op_name);
  } else {
    // printf("immediately ops\n");
    DoExec(std::move(fn), 0);
  }
}

void Device::RunGraph(bool serial) {
  bool previous_state = graph_enabled_;
  graph_enabled_ = false;

  if (serial) {
    // sequential execution
    graph_->RunInSerial();
  } else {
    // execute according to dependencies
    graph_->RunGraph();
  }

  // graph_->Debug();

  graph_enabled_ = previous_state;
}

void Device::PrintTimeProfiling() { graph_->PrintTimeProfiling(); }

// Todo(Wangwei) Get Block From The Memory manager
Block* Device::NewBlock(int size) {
  CHECK_GE(size, 0)
      << "size is negative, could be caused by the type cast "
      << "from size_t to int. In that case, the size is too large.";
  if (size > 0) {
    void* ptr = nullptr;
    if (!lazy_alloc_) {
      ptr = Malloc(size);
    }

    return new Block(ptr, size, this);
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
      {src}, {dst}, "CopyDataToFrom");
}

void Device::CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes,
                                 size_t dst_offset) {
  auto direct = lang_ == kCpp ? kHostToHost : kHostToDevice;
  void* dstptr = reinterpret_cast<char*>(dst->mutable_data()) + dst_offset;
  Exec([this, dstptr, src, nBytes,
        direct](Context* ctx) { CopyToFrom(dstptr, src, nBytes, direct, ctx); },
       {}, {dst}, "CopyDataFromHostPtr");
}
void Device::Sync() {}
}  // namespace singa
