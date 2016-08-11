/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "gtest/gtest.h"
#include "singa/utils/logging.h"
#include "singa/core/memory.h"
#include "singa/singa_config.h"
#include "singa/utils/timer.h"
#include "singa/utils/cuda_utils.h"
#include <stdlib.h>

// this tests allocated a number of memory blocks in the memory pool
// the pool consists of 1024 uints and each uint has a size of 1000 bytes
// we malloc 1024 blocks where half of the block will reside outside the pool,
// and the other half will be inside the pool
TEST(CppMemPool, Malloc) {
	singa::CppMemPool pool(1,1);
	const int numOfTests = 1024;
	const size_t dataSizeSmall = 1000;
	const size_t dataSizeLarge = 2000;
	singa::Block** pptr = new singa::Block*[numOfTests];

	for(int i = 0; i < numOfTests; i++) {
		const size_t dataSize = (i%2) ? dataSizeSmall : dataSizeLarge;
		pool.Malloc(&(pptr[i]),dataSize);
		int* data = static_cast<int*>(pptr[i]->mutable_data());
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			data[idx] = i;
		}
		data = static_cast<int*>(pptr[i]->mutable_data());
		int sum = 0;
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			sum += data[idx];
		}
		CHECK_EQ(sum,i*dataSize/4);
	}
	CHECK_EQ(512,pool.GetNumFreeUints());

	for(int i = 0; i < numOfTests; i++) {
		pool.Free(pptr[i]);
	}
	CHECK_EQ(1024,pool.GetNumFreeUints());

	delete[] pptr;
}

// this tests intialize a pool with size 2M bytes and each memory unit has a size of 2048 bytes
// we then allocated 1024 memory block with half of the blocks with size 2000 and the other half with size 1000
// then we reset the pool to size 1M bytes and memory uint size to 1000 bytes to test the reset function
TEST(CppMemPool, MallocAndRest) {
	singa::CppMemPool pool(2,2);
	const int numOfTests = 1024;
	const size_t dataSizeSmall = 1000;
	const size_t dataSizeLarge = 2000;
	singa::Block** pptr = new singa::Block*[numOfTests];

	for(int i = 0; i < numOfTests; i++) {
		const size_t dataSize = (i%2) ? dataSizeSmall : dataSizeLarge;
		pool.Malloc(&(pptr[i]),dataSize);
		int* data = static_cast<int*>(pptr[i]->mutable_data());
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			data[idx] = i;
		}
		data = static_cast<int*>(pptr[i]->mutable_data());
		int sum = 0;
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			sum += data[idx];
		}
		CHECK_EQ(sum,i*dataSize/4);
	}
	CHECK_EQ(0,pool.GetNumFreeUints());

	pool.RsetMemPool(1,1);
	CHECK_EQ(512,pool.GetNumFreeUints());
	for(int i = 0; i < numOfTests; i++) {
		const size_t dataSize = (i%2) ? dataSizeSmall : dataSizeLarge;
		int* data = static_cast<int*>(pptr[i]->mutable_data());
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			data[idx] = i;
		}
		data = static_cast<int*>(pptr[i]->mutable_data());
		int sum = 0;
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			sum += data[idx];
		}
		CHECK_EQ(sum,i*dataSize/4);
	}
	
	for(int i = 0; i < numOfTests; i++) {
		pool.Free(pptr[i]);
	}
	CHECK_EQ(1024,pool.GetNumFreeUints());

	delete[] pptr;
}

// this tests initialize a pool with size 1M bytes and uint size of 1024 bytes
// then 1024 memory blocks are allocated, half of them in the pool and the other half outside the pool
// subsequently, we randomly free 512 blocks and after that allocate them back to the pool
// after reset the pool to a size of 2M bytes and uint size of 2048 bytes,
// we free all memory blocks allocated. 
TEST(CppMemPool, RandomFree) {
	singa::CppMemPool pool(1,1);
	const int numOfTests = 1024;
	const size_t dataSizeSmall = 1000;
	const size_t dataSizeLarge = 2000;
	singa::Block** pptr = new singa::Block*[numOfTests];

	for(int i = 0; i < numOfTests; i++) {
		const size_t dataSize = (i%2) ? dataSizeSmall : dataSizeLarge;
		pool.Malloc(&(pptr[i]),dataSize);
		int* data = static_cast<int*>(pptr[i]->mutable_data());
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			data[idx] = i;
		}
		data = static_cast<int*>(pptr[i]->mutable_data());
		int sum = 0;
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			sum += data[idx];
		}
		CHECK_EQ(sum,i*dataSize/4);
	}
	CHECK_EQ(512,pool.GetNumFreeUints());

	// randomized free pointers
	int* randomPool = new int[numOfTests];
	for(int i = 0; i < numOfTests; i++) {
		randomPool[i] = i;
	}
	int iter = 0;
	while(iter != numOfTests/2) { // random free half of the memory blocks
		int pos = std::rand() % (numOfTests-iter);
		int i = randomPool[pos];
		std::swap(randomPool[pos],randomPool[numOfTests-1-iter]);
		
		// check value before deletion
		const size_t dataSize = (i%2) ? dataSizeSmall : dataSizeLarge;
		int* data = static_cast<int*>(pptr[i]->mutable_data());
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			data[idx] = i;
		}
		data = static_cast<int*>(pptr[i]->mutable_data());
		int sum = 0;
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			sum += data[idx];
		}
		CHECK_EQ(sum,i*dataSize/4);

		pool.Free(pptr[i]);
		iter++;
	}
	
	// test the unfreed memory block value
	for(int pos = 0; pos < numOfTests/2; pos++) {
		int i = randomPool[pos];
		const size_t dataSize = (i%2) ? dataSizeSmall : dataSizeLarge;
		int* data = static_cast<int*>(pptr[i]->mutable_data());
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			data[idx] = i;
		}
		data = static_cast<int*>(pptr[i]->mutable_data());
		int sum = 0;
		for(int idx = 0; idx < (int)dataSize/4; idx++) {
			sum += data[idx];
		}
		CHECK_EQ(sum,i*dataSize/4);
	}

	for(int pos = numOfTests/2; pos < numOfTests; pos++) {
		int i = randomPool[pos];
		const size_t dataSize = (i%2) ? dataSizeSmall : dataSizeLarge;
		pool.Malloc(&(pptr[i]),dataSize);
	}

	pool.RsetMemPool(2,2);
	for(int i = 0; i < numOfTests; i++) {
		pool.Free(pptr[i]);
	}
	CHECK_EQ(1024,pool.GetNumFreeUints());

	delete[] randomPool;
	delete[] pptr;
}

#ifdef USE_CUDA
/*
TEST(CnmemPool, PoolInitAll) {
  singa::CnMemPool pool(1);
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  CHECK_GE(nDevices, 1);
}

TEST(CnmemPool, UsePool) {
  singa::CnMemPool pool;
  pool.InitPool();
  int numOfTests = 10;
  int numOfWriteVsRead = 3;
  int allocSize = 32;
  for (int i = 0; i < numOfTests; i++) {
    int** memPtrs = new int* [numOfWriteVsRead];
    for (int j = 0; j < numOfWriteVsRead; j++) {
      pool.Malloc((void**)(&memPtrs[j]), allocSize);
    }
    pool.Free(memPtrs[0]);
    delete[] memPtrs;
  }
}
TEST(CudaMemPool, UsePool) {
  singa::CudaMemPool pool;
  int numOfTests = 10;
  int numOfWriteVsRead = 3;
  int allocSize = 32;
  for (int i = 0; i < numOfTests; i++) {
    int** memPtrs = new int* [numOfWriteVsRead];
    for (int j = 0; j < numOfWriteVsRead; j++) {
      pool.Malloc((void**)(&memPtrs[j]), allocSize);
    }
    pool.Free(memPtrs[0]);
    delete[] memPtrs;
  }
}
*/

TEST(MemPool, CompareCudaCnmem) {
  singa::CudaMemPool cudaPool;
  singa::CnMemPool cnPool;

  int numOfTests = 5000;
  int allocSize = 32;

  singa::DeviceMemPool* pool = NULL;
  pool = &cnPool;

  CUDA_CHECK(cudaSetDevice(0));
  singa::Timer tick;
  for (int i = 0; i < numOfTests; i++) {
    int* memPtrs = NULL;
    pool->Malloc((void**)&memPtrs, allocSize);
    pool->Free(memPtrs);
  }
  tick.Tick();
  int cn_time = tick.Elapsed();

  pool = &cudaPool;
  for (int i = 0; i < numOfTests; i++) {
    int* memPtrs = NULL;
    pool->Malloc((void**)&memPtrs, allocSize);
    pool->Free(memPtrs);
  }
  tick.Tick();
  int cuda_time = tick.Elapsed();
  EXPECT_GE(cuda_time, cn_time);
}
#endif  // USE_CUDA
