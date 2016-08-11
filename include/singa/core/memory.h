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

#include <mutex>
#include <atomic>
#include "singa/proto/core.pb.h"
#include "singa/singa_config.h"
#include "singa/core/common.h"

#ifdef USE_CUDA
#include "cnmem.h"
#endif


namespace singa {

/// Manage device memory pool including garbage collection, memory opt.
class VirtualMemory {};

class DeviceMemPool {
 public:
  virtual void Malloc(void** ptr, const size_t size)  = 0;
  virtual void Free(void* ptr)  = 0;

  /// Return a pair for free and total memory managed by this pool.
  virtual std::pair<size_t, size_t> GetMemUsage() {
    return std::make_pair(0u, 0u);
  }
  virtual ~DeviceMemPool(){};

 protected:
  size_t usage_;
//  size_t init_size_ = 0, max_size_ = 0;
};

class CppMemPool {
	public:
		// initial pool size (MB), and the size of each memory uint in the memory pool (KB)
		CppMemPool(size_t init_size_mb = 256, size_t uint_size_kb = 1);
		
		// return a new pool based on the current pool
		// once returned, the old pool will be invalid
		// re-initial with pool size (MB), and set the size of each memory uint in the memory pool (KB)
		void RsetMemPool(size_t init_size_mb = 256, size_t uint_size_kb = 1);

		// create the memory requested, if size is larger than memUintSize, malloc from system call
		// is_ptr_null indicate whether the pointer is null and if so we will initialize it in the malloc function,
		// otherwise we will use the ptr directly and access its data and functions.
		// after the malloc, the data pointer of the block will be changed and the orginal data pointer will be lost.
		void Malloc(Block** ptr, const size_t size, bool is_ptr_null = true);
		void Free(Block* ptr);

  	std::pair<size_t, size_t> GetMemUsage();
		size_t GetNumFreeUints(){return numUints - numAllocatedUintsInPool;};	

		// release all memory.
		// all pointers allocated in the pool must be freed before calling the descturctor. 
  	~CppMemPool();

	protected:
	// each structure define a memory uint in the memory pool
	// the structure is a static double linked list
		struct _Uint {
			struct _Uint *pPrev, *pNext;
			Block* pBlk;
		};

		// pointer to the memory pool
		void* pMemPool; 

		// head pointer to allocated memory uint
		struct _Uint* pAllocatedMemUint; 
		// head pointer to free memory uint
		struct _Uint* pFreeMemUint;

		// the size of each memory uint with/out the meta data of the uint 
		size_t memUintSize, memUintSizeNoMeta;

		// the number of memory uints in the pool
		size_t numUints;
		// the number of allocated uints which are resided in the memory pool
		size_t numAllocatedUintsInPool;
		// the number of allocated uints including the ones resided outside the memory pool
		size_t numAllocatedUints; 
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

  static std::atomic<int> pool_count;
};

class CudaMemPool : public DeviceMemPool {
 public:
  void Malloc(void** ptr, const size_t size) override;
  void Free(void* ptr) override;
};
#endif
}  // namespace singa
#endif  // SINGA_CORE_MEMORY_H_
