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
#include "singa/singa_config.h"
#include <atomic>
#include <memory>
#include "singa/utils/logging.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#endif // USE_CUDA


#ifdef USE_OPENCL
#include "singa/utils/opencl_utils.h"
#endif  // USE_OPENCL

using std::atomic;

namespace singa {

namespace lang {
/// To implemente functions using cpp libraries
typedef struct _Cpp { } Cpp;
/// To implemente functions using cuda libraries
typedef struct _Cuda { } Cuda;
/// To implement function using opencl libraries
typedef struct _Opencl { } Opencl;
}  // namespace lang

/// Block represent a chunk of memory (on device or host).
class Block {
 public:
  Block(void* ptr, size_t size, size_t offset = 0)
      : data_(ptr), size_(size), offset_(offset) {
    ref_count_ = 1;  // std::make_shared<std::atomic<int>>(1);
  }
  // Disabled as it is not used currently.
  // Block(void* ptr, size_t size, size_t offset, std::shared_ptr<atomic<int>>
  //  ref) : data_(ptr), size_(size), offset_(offset), ref_count_(ref) {}
  void* mutable_data() {
    initialized_ = true;
    return static_cast<char*>(data_) + offset_;
  }
  const void* data() const {
    CHECK(initialized_) << "Must initialize data before reading it";
    return static_cast<char*>(data_) + offset_;
  }
  size_t size() const { return size_; }
  size_t offset() const { return offset_; }
  int IncRefCount() {
    return ++ref_count_;  // Note do not use ref_count_++;
  }
  int DecRefCount() {
    return --ref_count_;
  }
  int ref_count() const { return ref_count_.load(); }

  bool initialized() const {
    return initialized_;
  }

 private:
  Block() {}
  void* data_ = nullptr;
  size_t size_ = 0;
  size_t offset_ = 0;
  bool initialized_ = false;
  // Disabled as it is not used currently.
  // std::shared_ptr<std::atomic<int>> ref_count_ = nullptr;
  std::atomic<int> ref_count_;
};

typedef struct _Context {
  std::mt19937 random_generator;
#ifdef USE_CUDA
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  curandGenerator_t curand_generator;
#ifdef USE_CUDNN
  cudnnHandle_t cudnn_handle;
#endif
#endif // USE_CUDA

#ifdef USE_OPENCL
  // This stores the context ID of the OpenCL context controlled by ViennaCL.
  long vcl_ctx_id;
#endif

} Context;

}  // namespace singa
#endif  // SINGA_CORE_COMMON_H_
