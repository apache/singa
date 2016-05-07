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
    : id_(id) {
  scheduler_ = nullptr;
  vm_ = nullptr;
}

void Device::Submit(function<void(Context*)> fn, const vector<Blob*> read_blobs,
                    const vector<Blob*> write_blobs) {
  fn(nullptr);
}

Blob* Device::NewBlob(int size) {
  if (size > 0) {
    void* ptr = malloc(size);
    memset(ptr, 0, size);
    return new Blob(ptr, size);
  } else {
    return nullptr;
  }
}

void Device::FreeBlob(Blob* blob) {
  if (blob != nullptr) {
    free(blob->mutable_data());
    delete blob;
  }
}

void Device::CopyData(Blob* dst, const Blob& src, int len, int dst_offset,
                      int src_offset) {
  memcpy(reinterpret_cast<Byte*>(dst->mutable_data()) + dst_offset,
         (const Byte*)src.data() + src_offset, len);
}

void Device::CopyDataFromHostPtr(Blob* dst, const void* src, size_t size) {
  memcpy(dst->mutable_data(), src, size);
}
void Device::Sync() {}
}  // namespace singa
