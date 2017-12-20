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
//for SmartMemPool
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <stdlib.h>     /* malloc, free, rand */
#include <map>
using namespace std;

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

//for SmartMemPool
struct lookUpElement{
    /*
     for memory pool Malloc look-up table.
     */
    int r_idx;
    int d_idx;
    size_t size;
    size_t offset;
    void* ptr;
    int Occupied; //0 is free, 1 is occupied.
    int crossItr; 
    int Occupied_backup; 
};

///class mem-pool SmartMemPool
class SmartMemPool: public DeviceMemPool {
public:
    SmartMemPool(const MemPoolConf &conf); //constructor
    //TODO(junzhe) in Singa, void Malloc( void**, size_t); change to cudaMalloc and cudaFree.
    void Malloc(void** ptr, const size_t size);
    void Free(void* ptr);
    ~SmartMemPool();
    void getMaxLoad(void);
    std::pair<size_t, size_t> GetMemUsage() override;
protected:
    void Init();
private:
    MemPoolConf conf_;
    // whether the (global) memory pool has been initialized
    bool initialized_ = false;
    // lock on the initialized variable
    std::mutex mtx_;

    string colorMethod;
    int mallocFlag =0; //0 for cudaMalloc, 1 for coloringMalloc
    int gc =0; //global counter each time Malloc/Free, add 1.
    int globeCounter=-1;
    int loadLogFlag =1; //record when its 1.
    void* ptrPool = NULL;
    int idxRange = 0;
    size_t offset = 0;
    size_t offsetCrossItr=0; //cross iteration offset.
    int maxLen =0;
    int location=0;
    vector<string> vec;
    map<int,int>Table_r2d; //full duration info, cross-iteration duration.
    map<int,int>Table_d2r;
    //map<int,lookUpElement>Table_r2Ver;
    vector<pair<int,lookUpElement>>Vec_r2Ver; //b. replace Table_r2Ver
    map<int, pair<size_t,size_t>>Table_load; //gc, <cudaLoad, colorLoad>
    map<void*,size_t>Table_p2s; //For tracking load in Free. add when allocate, delete when deallocate.
    map<void*,int>Table_p2r; //ptr for arrival idx, for look up Table during free
    int checkPoint=300; //for reduce number of test.
    size_t maxTotalLoad;
    size_t maxMemUsage;
    float memRatio;
};

#endif
}  // namespace singa
#endif  // SINGA_CORE_MEMORY_H_
