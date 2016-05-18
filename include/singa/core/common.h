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

#ifndef SINGA_CORE_COMMON_H_
#define SINGA_CORE_COMMON_H_
#include <random>
#include <chrono>
#include "singa/utils/logging.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#endif

namespace singa {
namespace lib {
/// To implemente functions using cpp libraries
typedef struct _Cpp { } Cpp;
/// To implemente functions using cuda libraries
typedef struct _Cuda { } Cuda;
/// To implement function using cudnn
typedef struct _Cudnn { } Cudnn;
/// To implement function using opencl libraries
typedef struct _Opencl { } Opencl;
}  // namespace lib

typedef unsigned char Byte;
/// Blob reprent a chunk of memory (on device or host) managed by VirtualMemory.
class Blob {
 public:
  Blob(void* ptr, size_t size) : data_(ptr), size_(size), ref_count_(1) {}
  void* mutable_data() const { return data_; }
  const void* data() const { return data_; }
  size_t size() const { return size_; }
  int IncRefCount() {
    ref_count_++;
    return ref_count_;
  }
  int DecRefCount() {
    ref_count_--;
    CHECK_GE(ref_count_, 0);
    return ref_count_;
  }
  int ref_count() const { return ref_count_; }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
  int ref_count_ = 0;
};

typedef struct _Context {
  std::mt19937 random_generator;
  unsigned long long seed;
#ifdef USE_CUDA
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
#ifdef USE_CUDNN
  cudnnHandle_t cudnn_handle;
#endif
#endif
} Context;

}  // namespace singa
#endif  // SINGA_CORE_COMMON_H_
