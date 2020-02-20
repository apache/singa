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
#include "singa/core/memory.h"
#include "singa/singa_config.h"
#include "singa/utils/cuda_utils.h"
#include "singa/utils/logging.h"
#include "singa/utils/timer.h"

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
