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

#include "singa/core/memory.h"
#include "singa/utils/logging.h"
#include "singa/proto/core.pb.h"
#include <iostream>

namespace singa {

std::pair<size_t, size_t> CppMemPool::GetMemUsage() {
	size_t total,free;
	total = memUintSize * numUints;
	free = total - memUintSize * numAllocatedUintsInPool;
	return std::make_pair(free,total);
}

CppMemPool::CppMemPool(size_t init_size_mb, size_t uint_size_kb)	{
	pMemPool = NULL ;
	pAllocatedMemUint = pFreeMemUint = NULL;
	memUintSize = memUintSizeNoMeta = 0;
	numUints = numAllocatedUintsInPool = numAllocatedUints = 0;
	RsetMemPool(init_size_mb,uint_size_kb);
}


void CppMemPool::RsetMemPool(size_t init_size_mb, size_t uint_size_kb)	{

	if(numAllocatedUintsInPool == 0) { // in the case the pool is empty
		// setting up the parameters in the memory pool
		const size_t kNBytesPerKB = (1u << 10);
		const size_t kNBytesPerMB = (1u << 20);
		memUintSize = uint_size_kb * kNBytesPerKB;
		memUintSizeNoMeta = memUintSize - sizeof(struct _Uint);
		size_t poolSize = init_size_mb * kNBytesPerMB; 
		bool memAligned = poolSize % memUintSize == 0;
		numUints = memAligned ? (poolSize / memUintSize) : (poolSize / memUintSize + 1);
		CHECK_GE(numUints,1);
		poolSize = memUintSize * numUints;
		
		// intialize the memory pool
		pMemPool = malloc(poolSize);
		CHECK(pMemPool != NULL);
		for(size_t idx = 0; idx < numUints; idx++) {
			struct _Uint *pCurUint = (struct _Uint*)((char *)pMemPool + idx * memUintSize);
			pCurUint->pPrev = NULL;
			pCurUint->pNext = pFreeMemUint;
			if(pFreeMemUint != NULL) {
				pFreeMemUint->pPrev = pCurUint;
			}
			pFreeMemUint = pCurUint;
			pCurUint->pBlk = NULL;
		}
	} else { // the pool is not empty, create a new one and copy the old to the new one
		CppMemPool* pNewPool = new CppMemPool(init_size_mb, uint_size_kb);
		struct _Uint* pCurUint = pAllocatedMemUint;
		for(size_t idx = 0; idx < numAllocatedUintsInPool; idx++) {
			Block* pOldBlk = pCurUint->pBlk;
			void* pData = pOldBlk->mutable_data();
			pNewPool->Malloc(&pOldBlk, pOldBlk->size(), false);
			size_t copySize = pOldBlk->size() - pOldBlk->offset();
			memcpy(pOldBlk->mutable_data(),pData,copySize);
			pCurUint = pCurUint->pNext;
		}
		// swap the new pool with the current
		std::swap(pNewPool->pMemPool,pMemPool);
		std::swap(pNewPool->pAllocatedMemUint,pAllocatedMemUint);
		std::swap(pNewPool->pFreeMemUint,pFreeMemUint);
		std::swap(pNewPool->memUintSize,memUintSize);
		std::swap(pNewPool->memUintSizeNoMeta,memUintSizeNoMeta);
		std::swap(pNewPool->numUints,numUints);	
		std::swap(pNewPool->numAllocatedUintsInPool,numAllocatedUintsInPool);	
		pNewPool->numAllocatedUints = 0;
		delete pNewPool;
	}
}

void CppMemPool::Malloc(Block** ptr, const size_t size, bool is_ptr_null) {
	numAllocatedUints++;
	// the size is larger than the memory uint size
	if(size > memUintSizeNoMeta || pFreeMemUint == NULL) { 
		void* pData = malloc(size);
		if(is_ptr_null) {
			*ptr = new Block(pData,size);
		} else {
			CHECK_EQ((*ptr)->size(),size);
			(*ptr)->set_data(pData);
		}
		return;
	}

	// otherwise retrieve from one of the memory uint
	numAllocatedUintsInPool++;
	struct _Uint *pCurUint = pFreeMemUint;
	pFreeMemUint = pCurUint->pNext;
	if(pFreeMemUint != NULL) {
		pFreeMemUint->pPrev = NULL;
	}
	
	pCurUint->pNext = pAllocatedMemUint;
	if(pAllocatedMemUint != NULL) {
		pAllocatedMemUint->pPrev = pCurUint;
	}

	pAllocatedMemUint = pCurUint;
	void* pData = (void*)((char *)pCurUint + sizeof(struct _Uint));
	if(is_ptr_null) {
		*ptr = new Block(pData,size);
	} else {
		CHECK_EQ((*ptr)->size(),size);
		(*ptr)->set_data(pData);
	}
	CHECK(pCurUint->pBlk == NULL);
	pCurUint->pBlk = *ptr;
}

void CppMemPool::Free(Block* ptr) {
	void* pData = ptr->mutable_data();
	if(pMemPool < pData && pData < (void*)((char*)pMemPool + numUints * memUintSize)) {
		struct _Uint *pCurUint = (struct _Uint*)((char*)pData-sizeof(struct _Uint));
		CHECK(ptr == pCurUint->pBlk);

		if(pCurUint == pAllocatedMemUint) {
				pAllocatedMemUint = pCurUint->pNext;
				if(pAllocatedMemUint != NULL) {
					pAllocatedMemUint->pPrev = NULL;
				}		
		} else {
				struct _Uint *pCurPrevUint = pCurUint->pPrev;
				pCurUint->pPrev = NULL;
				pCurPrevUint->pNext = pCurUint->pNext;
				if(pCurUint->pNext != NULL) {
					pCurUint->pNext->pPrev = pCurPrevUint;
				}
		}

		pCurUint->pNext = pFreeMemUint;
		if(pFreeMemUint != NULL) {
			pFreeMemUint->pPrev = pCurUint;
		}
		
		pFreeMemUint = pCurUint;
		pCurUint->pBlk = NULL;
		numAllocatedUintsInPool--;
	}
	else {
		free(pData);
	}
	numAllocatedUints--;
	delete ptr;
}

CppMemPool::~CppMemPool() {
	CHECK_EQ(numAllocatedUints,0);
	free(pMemPool);
}


#ifdef USE_CUDA
std::atomic<int> CnMemPool::pool_count(0);
std::pair<size_t, size_t> CnMemPool::GetMemUsage() {
  size_t free, total;
  auto status = cnmemMemGetInfo(&free, &total, NULL);
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
    << cnmemGetErrorString(status);
  return std::make_pair(free, total);
}

CnMemPool::CnMemPool(int numDevices, size_t init_size, size_t max_size) {
  for (int i = 0; i < numDevices; i++)
    conf_.add_device(i);
  conf_.set_init_size(init_size);
  conf_.set_max_size(max_size);
  CHECK_LT(++pool_count, 2) << "CnMemPool must be used as a singleton.";
}

CnMemPool::CnMemPool(const MemPoolConf &conf) {
  conf_ = conf;
  CHECK_LT(++pool_count, 2) << "CnMemPool must be used as a singleton.";
}

void CnMemPool::Init() {
  mtx_.lock();
  if (!initialized_) {
    const size_t kNBytesPerMB = (1u << 20);
    CHECK_GE(conf_.device_size(), 1);
    cnmemDevice_t *settingPtr = new cnmemDevice_t[conf_.device_size()];
    CHECK_GT(conf_.init_size(), 0u);
    int i = 0;
    for (auto device : conf_.device()) {
      settingPtr[i].device = device;
      settingPtr[i].size = conf_.init_size() * kNBytesPerMB;
      settingPtr[i].numStreams = 0;
      settingPtr[i].streams = NULL;
      settingPtr[i].streamSizes = 0;
      i++;
    }
    auto status = cnmemInit(conf_.device_size(), settingPtr, conf_.flag());
    CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
        << " " << cnmemGetErrorString(status);
    delete[] settingPtr;
    initialized_ = true;
  }
  mtx_.unlock();
}

CnMemPool::~CnMemPool() {
  mtx_.lock();
  if (initialized_) {
    cnmemStatus_t status = cnmemFinalize();
    CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
        << " " << cnmemGetErrorString(status);
    initialized_ = false;
    --pool_count;
  }
  mtx_.unlock();
}

void CnMemPool::Malloc(void **ptr, const size_t size) {
  if (!initialized_)
    Init();
  cnmemStatus_t status = cnmemMalloc(ptr, size, NULL);
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
      << " " << cnmemGetErrorString(status);
}

void CnMemPool::Free(void *ptr) {
  CHECK(initialized_) << "Cannot free the memory as the pool is not initialzied";
  cnmemStatus_t status = cnmemFree(ptr, NULL);
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
      << " " << cnmemGetErrorString(status);
}

// ===========================================================================
void CudaMemPool::Malloc(void **ptr, const size_t size) {
  cudaError_t status = cudaMalloc(ptr, size);
  CHECK_EQ(status, cudaError_t::cudaSuccess);
}

void CudaMemPool::Free(void *ptr) {
  cudaError_t status = cudaFree(ptr);
  CHECK_EQ(status, cudaError_t::cudaSuccess);
}
}
#endif
