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
/*
int get_pos(size_t size) {
	int result = 0;
	while(size > 1) {
		result++;
		size = size/2;
	}
	return result;
}
*/
namespace singa {

CppMemPool::CppMemPool()	{
	memPoolSize = 0;
	freeSize = 0;
	ppAllocUints = (struct _Uint**)malloc(64*sizeof(struct _Uint*));
	ppFreeUints = (struct _Uint**)malloc(64*sizeof(struct _Uint*));
	for(int i = 0; i < 64; i++) {
		ppAllocUints[i] = NULL;
		ppFreeUints[i] = NULL;
	}
}


Block* CppMemPool::Malloc(const size_t size) {	
	CHECK(size > 0);
	Block *pAllocBlk = NULL;
	int pos = 63 - __builtin_clzll(size);
	
	struct _Uint*& pAllocUint = ppAllocUints[pos];
	struct _Uint*& pFreeUint = ppFreeUints[pos];
	struct _Uint* pCurUint = NULL;
	size_t memSize = pow(2,pos);
	size_t blkSize = (size % memSize == 0) ? memSize : memSize*2;
	blkSize += sizeof(struct _Uint);
	
	if(pFreeUint == NULL) { // if no available free blocks
		memPoolSize += blkSize;
		pCurUint = (struct _Uint*)malloc(blkSize);
		pCurUint->pPrev = NULL;
		pCurUint->pNext = pAllocUint; 
		if(pAllocUint != NULL) {
			pAllocUint->pPrev = pCurUint;
		}
		pAllocUint = pCurUint;
		pAllocBlk = new Block((char*)(pCurUint) + sizeof(struct _Uint), size);
		pCurUint->pBlk = pAllocBlk;
	} else {
		freeSize -= blkSize;
		pCurUint = pFreeUint;
		pFreeUint = pCurUint->pNext;
		if(pFreeUint != NULL) {
			pFreeUint->pPrev = NULL;
		}
		
		pCurUint->pNext = pAllocUint;
		if(pAllocUint != NULL) {
			pAllocUint->pPrev = pCurUint;
		}
		pAllocUint = pCurUint;
		pAllocBlk = pCurUint->pBlk;
		pAllocBlk->set_size(size);
	}
	return pAllocBlk;
}

void CppMemPool::Free(Block* ptr) {
	void* pData = ptr->mutable_data();	
	struct _Uint *pCurUint = (struct _Uint*)((char*)pData-sizeof(struct _Uint));
	int pos = 63 - __builtin_clzll(ptr->size());
	struct _Uint*& pAllocUint = ppAllocUints[pos];
	struct _Uint*& pFreeUint = ppFreeUints[pos];
	size_t memSize = pow(2,pos); 
	size_t blkSize = (ptr->size() % memSize == 0) ? memSize : memSize*2;
	blkSize += sizeof(struct _Uint);
	freeSize += blkSize;

	if(pCurUint == pAllocUint) {
		pAllocUint = pCurUint->pNext;
		if(pAllocUint != NULL) {
			pAllocUint->pPrev = NULL;
		}		
	} else {
		struct _Uint *pCurPrevUint = pCurUint->pPrev;
		pCurUint->pPrev = NULL;
		pCurPrevUint->pNext = pCurUint->pNext;
		if(pCurUint->pNext != NULL) {
			pCurUint->pNext->pPrev = pCurPrevUint;
		}
	}
	
	pCurUint->pNext = pFreeUint;
	if(pFreeUint != NULL) {
		pFreeUint->pPrev = pCurUint;
	}		
	pFreeUint = pCurUint;
	ptr->set_size(0);
}


CppMemPool::~CppMemPool() {
	// traverse all lists to delete the memory
	for(int pos = 0; pos < 64; pos++) {
		for(int i = 0; i < 2; i++) {
			struct _Uint *pCurUint = i == 0 ? ppAllocUints[pos] : ppFreeUints[pos];
			while(pCurUint != NULL) {
				struct _Uint *pNextUint = pCurUint->pNext;
				free(pCurUint->pBlk);
				free(pCurUint);
				pCurUint = pNextUint;
			}
		}
	}
	free(ppAllocUints);
	free(ppFreeUints);
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
#endif
}
