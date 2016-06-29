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

#include "singa/singa_config.h"

#ifdef USE_CUDA
#include "cnmem.h"
#endif

#include <mutex>

namespace singa {

/// Manage device memory pool including garbage collection, memory opt.
class VirtualMemory {};

class DeviceMemPool {
 public:
  virtual void InitPool() = 0;
  virtual void Malloc(void** ptr, const size_t size) = 0;
  virtual void Free(void* ptr) = 0;
  virtual ~DeviceMemPool(){};
};

#ifdef USE_CUDA
class CnMemPool : public DeviceMemPool {
 public:
  int status = 1;

  void InitPool();

  /// numDevices: total number of available GPU cards.
  /// initSize: all devices will be allocated with this size
  /// manager_flags: pool manager flag (one for all devices)
  /// flag = 0; default flag
  /// flag = 1: Prevent the manager from growing its memory consumption
  /// flag = 2; Prevent the manager from stealing memory.
  void InitPool(int numDevices, size_t initSize, unsigned flag);

  void Malloc(void** ptr, const size_t size);
  void Free(void* ptr);

  // release all memory and set cnmem manager to unintialized
  ~CnMemPool();

 private:
  // whether the (global) memory pool has been initialized
  static bool initialized;
  // lock on the initialized variable
  static std::mutex mtx;
};

class CudaMemPool : public DeviceMemPool {
 public:
  void InitPool(){};
  void Malloc(void** ptr, const size_t size);
  void Free(void* ptr);
  ~CudaMemPool(){};
};
#endif
}  // namespace singa
#endif  // SINGA_CORE_MEMORY_H_
