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
Device::Device(int id, int num_executors, string scheduler, string vm)
    : id_(id), num_executors_(num_executors) {
      // TODO(wangwei) create scheduler and vm.
  host_ = &defaultDevice;
}

void Device::Exec(function<void(Context*)>&& fn, const vector<Blob*> read_blobs,
                    const vector<Blob*> write_blobs, bool use_rand_generator) {
  // TODO(wangwei) execute operations scheduled by the scheduler.
  DoExec(std::move(fn), 0);
}

// TODO(wangwei) get Blob from the memory manager
Blob* Device::NewBlob(int size) {
  if (size > 0) {
    void* ptr = Malloc(size);
    // memset(ptr, 0, size);
    return new Blob(ptr, size);
  } else {
    return nullptr;
  }
}

// TODO(wangwei) return Blob to the memory manager
void Device::FreeBlob(Blob* blob) {
  if (blob != nullptr) {
    Free(blob->mutable_data());
    delete blob;
  }
}

void Device::CopyDataToFrom(Blob* dst, Blob* src, size_t nBytes,
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

void Device::CopyDataFromHostPtr(Blob* dst, const void* src, size_t nBytes,
                                 size_t dst_offset) {
  auto direct = lang_ == kCpp ? kHostToHost : kHostToDevice;
  void* dstptr = reinterpret_cast<char*>(dst->mutable_data()) + dst_offset;
  Exec([this, dstptr, src, nBytes,
        direct](Context* ctx) { CopyToFrom(dstptr, src, nBytes, direct, ctx); },
       {}, {dst});
}
void Device::Sync() {}
}  // namespace singa
