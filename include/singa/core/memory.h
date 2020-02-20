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

#ifndef SINGA_CORE_MEMORY_H_
#define SINGA_CORE_MEMORY_H_

#include <atomic>
#include <mutex>

#include "singa/proto/core.pb.h"
#include "singa/singa_config.h"

#ifdef USE_CUDA
#include "cnmem.h"
#endif

namespace singa {

/// Manage device memory pool including garbage collection, memory opt.
class VirtualMemory {};

class DeviceMemPool {
 public:
  virtual void Malloc(void** ptr, const size_t size) = 0;
  virtual void Free(void* ptr) = 0;

  /// Return a pair for free and total memory managed by this pool.
  virtual std::pair<size_t, size_t> GetMemUsage() {
    return std::make_pair(0u, 0u);
  }
  virtual std::pair<size_t, size_t> GetMemUsage(int id) {
    return std::make_pair(0u, 0u);
  }
  virtual ~DeviceMemPool(){};

 protected:
  size_t usage_;
  //  size_t init_size_ = 0, max_size_ = 0;
};

#ifdef USE_CUDA
class CnMemPool : public DeviceMemPool {
 public:
  // Create the mem pool by setting the devices [0, numDevices), and
  // initial pool size (MB), and max pool size (no effect currently).
  CnMemPool(int numDevices = 1, size_t init_size = 256, size_t max_size = 0);
  CnMemPool(const MemPoolConf& conf);

  void Malloc(void** ptr, const size_t size);
  void Free(void* ptr);

  std::pair<size_t, size_t> GetMemUsage() override;
  std::pair<size_t, size_t> GetMemUsage(int id) override;

  // release all memory and set cnmem manager to unintialized
  ~CnMemPool();

 protected:
  void Init();

 private:
  MemPoolConf conf_;
  // whether the (global) memory pool has been initialized
  bool initialized_ = false;
  // lock on the initialized variable
  std::mutex mtx_;
};

class CudaMemPool : public DeviceMemPool {
 public:
  void Malloc(void** ptr, const size_t size) override;
  void Free(void* ptr) override;
};
#endif
}  // namespace singa
#endif  // SINGA_CORE_MEMORY_H_
