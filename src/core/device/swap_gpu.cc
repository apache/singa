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
#include "singa/singa_config.h"
#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include "singa/core/device.h"
#include "singa/utils/cuda_utils.h"

using namespace std;
namespace singa {

const cudaMemcpyKind copyKind[] = {cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
                                   cudaMemcpyDeviceToHost,
                                   cudaMemcpyDeviceToDevice};

SwapGPU::~SwapGPU() {
  //print out push-info
  fstream file_block1("blockInfo.text", ios::in|ios::out|ios::app);
  for (int i=0; i< vec_block.size();i++){
      file_block1<<vec_block[i]<<endl;
  }
  //main body
  if (ctx_.cublas_handle) CUBLAS_CHECK(cublasDestroy(ctx_.cublas_handle));
  if (ctx_.curand_generator)
    CURAND_CHECK(curandDestroyGenerator(ctx_.curand_generator));
#ifdef USE_CUDNN
  if (ctx_.cudnn_handle) {
    auto status = cudnnDestroy(ctx_.cudnn_handle);
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
  }
#endif
}
const int kNumCudaStream = 1;

SwapGPU::SwapGPU(int id) : Device(id, kNumCudaStream) {
  MemPoolConf conf;
  conf.add_device(id);
  pool_ = std::make_shared<Swap>(conf);
  Setup();
}

SwapGPU::SwapGPU(int id, std::shared_ptr<DeviceMemPool> pool)
    : Device(id, kNumCudaStream) {
  CHECK(pool != nullptr);
  pool_ = pool;
  Setup();
}

void SwapGPU::Setup() {
  lang_ = kCuda;
  ctx_.stream = NULL;  // use the default sync stream
  // TODO(wangwei) create one handle for each steam?
  CUDA_CHECK(cudaSetDevice(id_));
  // use curandCreateGeneratorHost for CudaHost device
  CURAND_CHECK(
      curandCreateGenerator(&ctx_.curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  SetRandSeed(seed);
  // TODO(wangwei) if one generator per stream, then need diff offset per gen?
  CURAND_CHECK(curandSetGeneratorOffset(ctx_.curand_generator, 0));
  CUBLAS_CHECK(cublasCreate(&(ctx_.cublas_handle)));

#ifdef USE_CUDNN
  // TODO(wangwei) create one handle for each stream?
  auto status = cudnnCreate(&ctx_.cudnn_handle);
  CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
#endif  // USE_CUDNN
}

void SwapGPU::SetRandSeed(unsigned seed) {
  CHECK(ctx_.curand_generator);
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx_.curand_generator, seed));
}

void SwapGPU::DoExec(function<void(Context*)>&& fn, int executor) { fn(&ctx_); }

void SwapGPU::CopyToFrom(void* dst, const void* src, size_t nBytes,
                         CopyDirection direction, Context* ctx) {
  cudaMemcpy(dst, src, nBytes, copyKind[direction]);
  // TODO(wangwei) use async copy
  // cudaMemcpyAsync(dst, src, nBytes,cudaMemcpyDefault, ctx_.stream);
}

size_t SwapGPU::GetAllocatedMem() {
  if (pool_ != nullptr) {
    auto ret = pool_->GetMemUsage();
    return ret.second - ret.first;
  }
  LOG(ERROR) << "The memory pool is not set";
  return 0u;
}

/// Allocate gpu memory.
void* SwapGPU::Malloc(int size) {
  void* ptr = nullptr;
  //cout<<"hello, SwapGPU."<<endl;
  if (size > 0) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Malloc((void**)&ptr, size);
    // TODO(wangwei) remove the memset.
    CUDA_CHECK(cudaMemset(ptr, 0, size));
  }

  return ptr;
}

/// Free gpu memory.
void SwapGPU::Free(void* ptr) {
  cout<<"test function in memory.cc: "<<endl;
  onePieceMsg tempMsg;
  cout<<"no error calling onePieceMsg"<<endl;
  if (ptr != nullptr) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Free(ptr);
  }
  //push info
  stringstream strm1;
  strm1<<Table_data_block_.find(ptr)->second;;
  string tempStr1 = strm1.str();
  stringstream strm4;
  auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  strm4<<t2;
  string tempStr4 = strm4.str();
  string blockInfo ="Free "+tempStr1+" "+tempStr4;
  vec_block.push_back(blockInfo);
  //clean up Tables
  Table_Meta.erase(Table_data_block_.find(ptr)->second);
  Table_data_block_.erase(ptr);
  
}

void SwapGPU::MakeMetaTable(Block* block_,void* data_,int size){
  //this is only called once, right after Malloc. 
  //Hence the malloc info is pushed here.
  BlockMeta cpu,gpu;
  cpu.size = size;
  gpu.size = size;
  gpu.ptr = data_;
  pair<BlockMeta,BlockMeta>meta = std::make_pair(cpu, gpu);
  //Make tables
  Table_Meta[block_] = meta;
  Table_data_block_[data_]=block_; //table map data_block, for Free(). 
  //TODO(junzhe) update this table once data_ changed.
  //push info
  stringstream strm1;
  strm1<<size;
  string tempStr1 = strm1.str();
  stringstream strm3;
  strm3<<block_;
  string tempStr3 = strm3.str();
  stringstream strm4;
  auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  strm4<<t2;
  string tempStr4 = strm4.str();
  string blockInfo ="Malloc "+tempStr3+" "+tempStr1+" "+tempStr4+" (data_&size)";
  vec_block.push_back(blockInfo);
}

void SwapGPU::Append(string blockInfo){
  vec_block.push_back(blockInfo);
}

void* SwapGPU::GetRealGpuPtr(const Block* block_){
  //void* data_ = Table_Meta.find(block_)->second.second.ptr;
  return Table_Meta.find(block_)->second.second.ptr;
}

void SwapGPU::SwapOut(const Block* block_){
  printf("A. to swapOut\n");
  auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
  size_t swapSize = Table_Meta.find(block_)->second.second.size;
  Table_Meta.find(block_)->second.first.ptr = malloc(swapSize);
  BlockMeta cpu, gpu;
  cpu = Table_Meta.find(block_)->second.first;
  gpu = Table_Meta.find(block_)->second.second;
  cudaError_t err;
  cout<<"to Copy: "<<cpu.ptr<<' '<<gpu.ptr<<' '<<gpu.size<<endl;
  err=cudaMemcpy(cpu.ptr,gpu.ptr,gpu.size,cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    {
    fprintf(stderr, "SwapOut err (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  // fstream file_block3("blockInfo_swapOut.text", ios::in|ios::out|ios::app);
  // file_block3<<t2-t1<<" "<<swapSize<<endl;
  //printf("B. swapOut done.\n");
  // //cout<<"before free: "<<data_<<endl;
  //without free here.
  //cudaFree(gpu.ptr); //TODO(junzhe) not able to free, work on it.
  //Table_Meta.find(block_)->second.second.ptr=nullptr;
  // //cout<<"after free: "<<data_<<endl;
}

void SwapGPU::SwapIn(const Block* block_){
  printf("1. to swapIn.\n");
  auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
  BlockMeta cpu, gpu;
  cpu = Table_Meta.find(block_)->second.first;
  gpu = Table_Meta.find(block_)->second.second;
  //without free here.
  gpu.ptr=nullptr;
  cudaError_t status = cudaMalloc(&gpu.ptr, gpu.size);
  CHECK_EQ(status, cudaError_t::cudaSuccess);
  //update tables
  Table_Meta.find(block_)->second.second.ptr=gpu.ptr;
  //cout<<"after alloc:1 "<<Table_Meta.find(data_)->second.second.ptr<<endl;
  cudaError_t err;
  err=cudaMemcpy(gpu.ptr, cpu.ptr ,cpu.size,cudaMemcpyHostToDevice);
  //printf("2. swapIn done.\n");
  free(cpu.ptr);
  //update tables
  Table_Meta.find(block_)->second.first.ptr=nullptr;
  //
  auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  fstream file_block3("blockInfo_swapIn.text", ios::in|ios::out|ios::app);
  file_block3<<t2-t1<<" "<<gpu.size<<endl;
}

///functions to be used



}  // namespace singa
#endif  // USE_CUDA
