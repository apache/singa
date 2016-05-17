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
#ifdef USE_CUDA
#include <chrono>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "singa/core/device.h"
#include "singa/utils/cuda.h"
namespace singa {

const cudaMemcpyKind copyKind[] = {cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
                                   cudaMemcpyDeviceToHost,
                                   cudaMemcpyDeviceToDevice};

CudaDevice::~CudaDevice() {
  if (ctx_.cublas_handle)
    CUBLAS_CHECK(cublasDestroy(ctx_.cublas_handle));
  if (ctx_.curand_generator)
    CURAND_CHECK(curandDestroyGenerator(ctx_.curand_generator));
#ifdef USE_CUDNN
  if (ctx_.cudnn_handle) {
    auto status = cudnnDestroy(ctx_.cudnn_handle);
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
  }
#endif
}

CudaDevice::CudaDevice(int id, int num_executors,
                       string scheduler, string vm)
    : Device(id, num_executors, scheduler, vm) {
  device_type_ = kCuda;
  host_ = nullptr; // TODO(wangwei) add host device
  ctx_.stream = NULL; // use the default sync stream
  // TODO(wangwei) create one handle for each steam?
  CUBLAS_CHECK(cublasCreate(&ctx_.cublas_handle));
  // use curandCreateGeneratorHost for CudaHost device
  CURAND_CHECK(
      curandCreateGenerator(&ctx_.curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  SetRandSeed(seed);
  // TODO(wangwei) if one generator per stream, then need diff offset per gen?
  CURAND_CHECK(curandSetGeneratorOffset(ctx_.curand_generator, 0));

#ifdef USE_CUDNN
  // TODO(wangwei) create one handle for each stream?
  auto status = cudnnCreate(&ctx_.cudnn_handle);
  CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
#endif  // USE_CUDNN
}

void CudaDevice::SetRandSeed(unsigned seed) {
  CHECK(ctx_.curand_generator);
  CURAND_CHECK(
      curandSetPseudoRandomGeneratorSeed(ctx_.curand_generator, seed));
}

void CudaDevice::DoExec(function<void(Context*)>&& fn, int executor) {
  fn(&ctx_);
}

void CudaDevice::CopyToFrom(void* dst, const void* src, size_t nBytes,
                            CopyDirection direction, Context* ctx) {
  cudaMemcpy(dst, src, nBytes, copyKind[direction]);
  // TODO(wangwei) use async copy
  // cudaMemcpyAsync(dst, src, nBytes,cudaMemcpyDefault, ctx_.stream);
}

/// Allocate cpu memory.
void* CudaDevice::Malloc(int size) {
  void* ptr = nullptr;
  cudaMalloc(&ptr, size);
  return ptr;
}

  /// Free cpu memory.
void CudaDevice::Free(void* ptr) {
  CHECK_NE(ptr, nullptr);
  cudaFree(ptr);
}


// ==========Following code is from Caffe src/caffe/common.cpp=================

void CudaDevice::DeviceQuery() {
  cudaDeviceProp prop;
  int device;
  if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  LOG(INFO) << "Device id:                     " << device;
  LOG(INFO) << "Major revision number:         " << prop.major;
  LOG(INFO) << "Minor revision number:         " << prop.minor;
  LOG(INFO) << "Name:                          " << prop.name;
  LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
  LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
  LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
  LOG(INFO) << "Warp size:                     " << prop.warpSize;
  LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
  LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
  LOG(INFO) << "Maximum dimension of block:    "
      << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
      << prop.maxThreadsDim[2];
  LOG(INFO) << "Maximum dimension of grid:     "
      << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
      << prop.maxGridSize[2];
  LOG(INFO) << "Clock rate:                    " << prop.clockRate;
  LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
  LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
  LOG(INFO) << "Concurrent copy and execution: "
      << (prop.deviceOverlap ? "Yes" : "No");
  LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
  LOG(INFO) << "Kernel execution timeout:      "
      << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  return;
}

bool CudaDevice::CheckDevice(const int device_id) {
  bool r = ((cudaSuccess == cudaSetDevice(device_id)) &&
            (cudaSuccess == cudaFree(0)));
  // reset any error that may have occurred.
  cudaGetLastError();
  return r;
}

int CudaDevice::FindDevice(const int start_id) {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) return i;
  }
  return -1;
}


}
#endif  // USE_CUDA
