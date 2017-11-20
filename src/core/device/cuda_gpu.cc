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
#include "singa/core/device.h"
#include "singa/utils/cuda_utils.h"

//added for print cudaMalloc info, not all needed.
#include <iostream>
#include <fstream>
#include <stdint.h>
using namespace std;

namespace singa {

const cudaMemcpyKind copyKind[] = {cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
                                   cudaMemcpyDeviceToHost,
                                   cudaMemcpyDeviceToDevice};

CudaGPU::~CudaGPU() {
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

CudaGPU::CudaGPU(int id) : Device(id, kNumCudaStream) {
  MemPoolConf conf;
  conf.add_device(id);
  pool_ = std::make_shared<CnMemPool>(conf);
  Setup();
}

CudaGPU::CudaGPU(int id, std::shared_ptr<DeviceMemPool> pool)
    : Device(id, kNumCudaStream) {
  CHECK(pool != nullptr);
  pool_ = pool;
  Setup();
}

void CudaGPU::Setup() {
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

void CudaGPU::SetRandSeed(unsigned seed) {
  CHECK(ctx_.curand_generator);
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx_.curand_generator, seed));
}

void CudaGPU::DoExec(function<void(Context*)>&& fn, int executor) { fn(&ctx_); }

void CudaGPU::CopyToFrom(void* dst, const void* src, size_t nBytes,
                         CopyDirection direction, Context* ctx) {
  cudaMemcpy(dst, src, nBytes, copyKind[direction]);
  // TODO(wangwei) use async copy
  // cudaMemcpyAsync(dst, src, nBytes,cudaMemcpyDefault, ctx_.stream);
}

size_t CudaGPU::GetAllocatedMem() {
  if (pool_ != nullptr) {
    auto ret = pool_->GetMemUsage();
    return ret.second - ret.first;
  }
  LOG(ERROR) << "The memory pool is not set";
  return 0u;
}

/// Allocate gpu memory.
void* CudaGPU::Malloc(int size) {
  void* ptr = nullptr;
  if (size > 0) {
    CUDA_CHECK(cudaSetDevice(id_));
    //pool_->Malloc((void**)&ptr, size);
    // below are done by cudaMalloc instead of cnmemPool::Malloc,  by junzhe 11/20
    cudaMalloc((void**)&ptr,size);
    fstream file4("cudaMalloc_memInfo.text", ios::in|ios::out|ios::app);
    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    file4<<"Malloc "<<ptr<<' '<<size<<' '<<now<<endl;
    size_t free_byte=0;
    size_t total_byte=0;
    cudaMemGetInfo(&free_byte,&total_byte);
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    fstream file5("cudaMemGetInfo.text", ios::in|ios::out|ios::app);
    file5<<"Malloc "<<used_db/1024.0/1024.0<<' '<<free_db/1024.0/1024.0<<' '<<total_db/1024.0/1024.0<<endl;
    // TODO(wangwei) remove the memset.
    CUDA_CHECK(cudaMemset(ptr, 0, size));
  }
  return ptr;
}

/// Free gpu memory.
void CudaGPU::Free(void* ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaSetDevice(id_));
    //pool_->Free(ptr);
    // below are done by cudaMalloc instead of cnmemPool::Free,  by junzhe 11/20
    cudaFree(ptr);
    fstream file4("cudaMalloc_memInfo.text", ios::in|ios::out|ios::app);
    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    file4<<"Free "<<ptr<<' '<<now<<endl;
    size_t free_byte=0;
    size_t total_byte=0;
    cudaMemGetInfo(&free_byte,&total_byte);
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    fstream file5("cudaMemGetInfo.text", ios::in|ios::out|ios::app);
    file5<<"Free "<<used_db/1024.0/1024.0<<' '<<free_db/1024.0/1024.0<<' '<<total_db/1024.0/1024.0<<endl;
  }
}

}  // namespace singa
#endif  // USE_CUDA
