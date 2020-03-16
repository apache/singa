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
  graph_ = new Graph(this);
}

void Device::Exec(function<void(Context*)>&& fn,
                  const vector<Block*> read_blocks,
                  const vector<Block*> write_blocks, bool use_rand_generator) {
  if (buffer_flag_== true) {
    graph_->AddOperation(std::move(fn), read_blocks, write_blocks);
  } else {
    // printf("immediately ops\n");
    DoExec(std::move(fn), 0);
  }
}

void Device::ExecBuffOps() {
  bool previous_state = buffer_flag_;
  buffer_flag_ = false;

  graph_->Run();

  buffer_flag_ = previous_state;
}

// Todo(Wangwei) Get Block From The Memory manager
Block* Device::NewBlock(int size) {
  CHECK_GE(size, 0)
      << "size is negative, could be caused by the type cast "
      << "from size_t to int. In that case, the size is too large.";
  if (size > 0) {
    // void* ptr = Malloc(size);
    // return new Block(ptr, size, this);
    // support lazy allocation
    return new Block(nullptr, size, this);
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
