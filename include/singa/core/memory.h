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
  virtual void Append(string blockInfo) = 0;

  virtual void PoolOpt(vector<string> &vec_mf) = 0;
  
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
  void Append(string blockInfo){}

  void PoolOpt(vector<string> &vec_mf) override {}
    
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
  void Append(string blockInfo){}

  void PoolOpt(vector<string> &vec_mf) override {}

};

//for SmartMemPool and SwapPool
struct PoolBlockMeta{
  /*
   for memory pool Malloc look-up table.
   */
  int r_idx;
  int d_idx;
  size_t size;
  size_t offset;
  void* ptr;
  int occupied; //0 is free, 1 is occupied.
  int cross_iteration; 
  int occupied_backup; 
};

///struct Vertex
struct Vertex{
  int name;
  size_t size;
  int r; //arrive
  int d; //depart
  int cross_iteration =0;
  pair<size_t, size_t> color_range;
  vector<pair<size_t, size_t>> vec_color_preoccupied;
  Vertex(int n, size_t s, int r1, int d1):name(n),size(s),r(r1),d(d1){}

};


///SmartMemPool
class SmartMemPool: public DeviceMemPool {
public:
  SmartMemPool(const MemPoolConf &conf); //constructor
  void Malloc(void** ptr, const size_t size);
  void Free(void* ptr);
  ~SmartMemPool();
  std::pair<size_t, size_t> GetMemUsage() override;
  void GetMaxLoad(void);
  void Append(string blockInfo);
  vector<Vertex> Plan(vector<string>vec, int &idx_range, size_t &offset, size_t &offset_cross_iteration,string color_method);
  int Detection(vector<string>vec_string_test, int &iteration_length, int &location_2nd_iteration);

  void PoolOpt(vector<string> &vec_mf) override {}

protected:
  void Init();
private:
  MemPoolConf conf_;
  // whether the (global) memory pool has been initialized
  bool initialized_ = false;
  // lock on the initialized variable
  std::mutex mtx_;

  string color_method;
  int malloc_flag = 0; //0 for cudaMalloc, 1 for coloringMalloc
  int global_index = 0; //global counter each time Malloc/Free, add 1.
  int global_index_threshold = -1;
  int load_flag = 1; //record load at 1
  void* ptr_pool = NULL;
  int idx_range = 0;
  size_t offset = 0;
  size_t offset_cross_iteration = 0; //cross iteration offset.
  int iteration_length = 0;
  int location_2nd_iteration = 0;
  vector<string> vec;
  vector<string> vec_block_rw; //read write only opt info
  vector<string> vec_block_rw_mf; //read write, malloc, free opt info
  map<int,int>table_ridx_to_didx; //table match from r_idx to d_idx
  map<int,int>table_didx_to_ridx; //table match from d_idx to r_idx

  vector<pair<int,PoolBlockMeta>>vec_block_meta; //vec of block meta, index in the vector refering to the r_idx
  map<int, pair<size_t,size_t>>table_load; //global_index, <cudaLoad, colorLoad>
  map<void*,size_t>table_ptr_to_size; //for tracking load in Free. add when allocate, delete when deallocate.
  map<void*,int>table_ptr_to_ridx; //ptr for arrival idx, for look up Table during free
  int check_point = 300; //for reduce number of test.
  size_t max_total_load;
  size_t max_mem_usage;
};


///SwapPool 
class SwapPool : public DeviceMemPool {
public:
  SwapPool(const MemPoolConf &conf); //constructor
  void Malloc(void** ptr, const size_t size);
  void Free(void* ptr);
  ~SwapPool();
  std::pair<size_t, size_t> GetMemUsage() override;
  void Append(string blockInfo);

  //PoolOpt() construct pool based on MF info after Swap constructed.
  void PoolOpt(vector<string> &vec_mf);
protected:
  void Init();
private:
  MemPoolConf conf_;
  // whether the (global) memory pool has been initialized
  bool initialized_ = false;
  // lock on the initialized variable
  std::mutex mtx_; 

  vector<string> vec_block;
  int pool_flag = 0;
  int pool_index = 0; //like global counter in device class
  int iteration_length_mf = 0; //max length of malloc free operation sequences.
  void* ptr_pool = nullptr;
  map<void*,int>table_ptr_to_ridx; //map ptr to arrival idx, for look up Table during free
  map<int,PoolBlockMeta>table_pool_meta; //table of pool block meta, key with r_idx
};

#endif
}  // namespace singa
#endif  // SINGA_CORE_MEMORY_H_