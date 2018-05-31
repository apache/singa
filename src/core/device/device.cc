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
#include <chrono>
#include <iostream>
#include <fstream>

namespace singa {
Device::Device(int id, int num_executors)
    : id_(id), num_executors_(num_executors) {
  // TODO(wangwei) create scheduler and vm.
  host_ = defaultDevice;
}

void Device::Exec(function<void(Context*)>&& fn, const vector<Block*> read_blocks,
                    const vector<Block*> write_blocks, bool use_rand_generator) {
  // TODO(wangwei) execute operations scheduled by the scheduler.
  DoExec(std::move(fn), 0);
}

// TODO(wangwei) get Block from the memory manager
Block* Device::NewBlock(int size) {
  CHECK_GE(size, 0) << "size is negative, could be caused by the type cast "
    << "from size_t to int. In that case, the size is too large.";
  if (size > 0) {
    void* ptr = Malloc(size);
    Block* block_ = new Block(ptr, size,0,this);
    //std::cout<<"(reference) from device.cc after, data_, block_ device: "<<ptr<<" "<<block_<<' '<<this<<std::endl;
    MakeMetaTable(block_,ptr,size); // make table and append vec_block.
    //cout<<"NewBlock: "<<block_<<' '<<ptr<<endl;
    return block_;
  } else {
    return nullptr;
  }
}

// TODO(wangwei) return Block to the memory manager
void Device::FreeBlock(Block* block) {
  if (block != nullptr) {
    //TODO(junzhe) to merge it
    auto tempPtr = block->mutable_data();
    //cout<<"FreeBlock: "<<block<<' '<<tempPtr<<endl;
    Free(tempPtr);
    //cout<<"SwapGPU::Free() returned"<<endl;
    //Free(block->mutable_data());
    
    //Add Append for free here.
    stringstream strm1;
    strm1<<block;
    string tempStr1 = strm1.str();
    stringstream strm4;
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
    strm4<<t2;
    string tempStr4 = strm4.str();
    string blockInfo ="Free "+tempStr1+" "+tempStr4;
    Append(blockInfo);

    delete block;
  }
}

void Device::AppendInfo(string blockInfo){
  Append(blockInfo);
}

void* Device::GetRealGpuPtrInfo(const Block* block_){
  return GetRealGpuPtr(block_);
}

void Device::SwapOutInfo(const Block* block_){
  SwapOut(block_);
}

void Device::SwapInInfo(const Block* block_){
  SwapIn(block_);
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
