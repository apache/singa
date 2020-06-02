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

#ifdef USE_DIST
  CUDA_CHECK(cudaStreamDestroy(ctx_.s));
  CUDA_CHECK(cudaStreamDestroy(ctx_.c1));
  CUDA_CHECK(cudaStreamDestroy(ctx_.c2));
#endif  // USE_DIST
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
  // Preserse for future use instead of default sync stream, for concurrency
  // cudaStreamCreate(&ctx_.stream);

#ifdef USE_DIST
  CUDA_CHECK(cudaStreamCreateWithFlags(&ctx_.s, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&ctx_.c1, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&ctx_.c2, cudaStreamNonBlocking));
#endif  // USE_DIST

  CUDA_CHECK(cudaSetDevice(id_));
  // use curandCreateGeneratorHost for CudaHost device
  CURAND_CHECK(
      curandCreateGenerator(&ctx_.curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetStream(ctx_.curand_generator, ctx_.stream));
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  SetRandSeed(seed);
  // TODO(wangwei) if one generator per stream, then need diff offset per gen?
  CURAND_CHECK(curandSetGeneratorOffset(ctx_.curand_generator, 0));
  CUBLAS_CHECK(cublasCreate(&(ctx_.cublas_handle)));
  CUBLAS_CHECK(cublasSetStream(ctx_.cublas_handle, ctx_.stream));

#ifdef USE_CUDNN
  // TODO(wangwei) create one handle for each stream?
  auto status = cudnnCreate(&ctx_.cudnn_handle);
  CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
  cudnnSetStream(ctx_.cudnn_handle, ctx_.stream);
#endif  // USE_CUDNN
}

void CudaGPU::SetRandSeed(unsigned seed) {
  CHECK(ctx_.curand_generator);
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx_.curand_generator, seed));
}

void CudaGPU::DoExec(function<void(Context*)>&& fn, int executor) { fn(&ctx_); }

void CudaGPU::SyncBeforeCountingTime() {
  // synchronization before counting time
  bool previous_state = graph_enabled();
  graph_enabled_ = false;
  Sync();
  graph_enabled_ = previous_state;
}

void CudaGPU::EvaluateTimeElapsed(Node* node) {
  float totalTime;

  cudaEventElapsedTime(&totalTime, node->start_, node->end_);

  cudaEventDestroy(node->start_);
  cudaEventDestroy(node->end_);

  node->time_elapsed_inc(totalTime * 0.001);
}

void CudaGPU::TimeProfilingDoExec(function<void(Context*)>&& fn, int executor,
                                  Node* node) {
  // time profiling using cudaEvent
  cudaEventCreate(&(node->start_));
  cudaEventCreate(&(node->end_));

#ifdef USE_DIST
  if (node->op_name().find("Dist") != std::string::npos) {
    if (node->op_name().find("Dist_s") != std::string::npos)
      cudaEventRecord(node->start_, ctx_.s);
    else if (node->op_name().find("Dist_c1") != std::string::npos)
      cudaEventRecord(node->start_, ctx_.c1);
    else if (node->op_name().find("Dist_c2") != std::string::npos)
      cudaEventRecord(node->start_, ctx_.c2);
    else if (node->op_name().find("Dist_c1c2") != std::string::npos)
      cudaEventRecord(node->start_, ctx_.c1);
  } else {
    cudaEventRecord(node->start_, ctx_.stream);
  }
#else
  cudaEventRecord(node->start_, ctx_.stream);
#endif  // USE_DIST

  fn(&ctx_);

#ifdef USE_DIST
  if (node->op_name().find("Dist") != std::string::npos) {
    if (node->op_name().find("Dist_s") != std::string::npos)
      cudaEventRecord(node->end_, ctx_.s);
    else if (node->op_name().find("Dist_c1") != std::string::npos)
      cudaEventRecord(node->end_, ctx_.c1);
    else if (node->op_name().find("Dist_c2") != std::string::npos)
      cudaEventRecord(node->end_, ctx_.c2);
    else if (node->op_name().find("Dist_c1c2") != std::string::npos)
      cudaEventRecord(node->end_, ctx_.c2);
  } else {
    cudaEventRecord(node->end_, ctx_.stream);
  }
#else
  cudaEventRecord(node->end_, ctx_.stream);
#endif  // USE_DIST
}

void CudaGPU::CopyToFrom(void* dst, const void* src, size_t nBytes,
                         CopyDirection direction, Context* ctx) {
  // cudaMemcpy(dst, src, nBytes, copyKind[direction]);
  cudaMemcpyAsync(dst, src, nBytes, copyKind[direction], ctx_.stream);
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
    pool_->Malloc((void**)&ptr, size);
    // Comment out for future analysis: without cnmem
    // CUDA_CHECK(cudaMemsetAsync(ptr, 0, size, ctx_.stream));
  }
  return ptr;
}

/// Free gpu memory.
void CudaGPU::Free(void* ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Free(ptr);
  }
}

void CudaGPU::Sync() {
  Exec([this](Context* ctx) { CUDA_CHECK(cudaDeviceSynchronize()); }, {}, {},
       "Waiting");
}

}  // namespace singa
#endif  // USE_CUDA
