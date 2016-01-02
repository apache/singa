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

#ifndef SINGA_UTILS_CONTEXT_H_
#define SINGA_UTILS_CONTEXT_H_

#include <glog/logging.h>
#include <chrono>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef USE_GPU
#include "singa/utils/cuda_utils.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#endif

namespace singa {

/**
 * Context is used as a global singleton, which stores the mapping from CPU
 * thread id to GPU device id. If a thread has no GPU, then its associated
 * device id is -1. It manages (e.g., creating) the handlers for GPU
 * devices. It also manages the GPU and CPU random generators, which are created
 * when accessed. One CPU thread has a CPU random generator. A GPU device
 * has a GPU random generator, which is accessible after assigning the GPU
 * device with a CPU thread via SetupDevice.
 */
class Context {
 public:
   /**
    * Destructor, release random generators and handlers.
    */
  ~Context() {
#ifdef USE_GPU
    for (auto& entry : device_id_) {
      if (entry.second != -1) {
        cudaSetDevice(entry.second);
        if (cublas_handle_[entry.second] != nullptr) {
          cublasDestroy(cublas_handle_[entry.second]);
          cublas_handle_[entry.second] = nullptr;
        }
        if (curand_generator_[entry.second] != nullptr) {
          curandDestroyGenerator(curand_generator_[entry.second]);
          curand_generator_[entry.second] = nullptr;
        }
      }
    }
#ifdef USE_CUDNN
    for (auto& handle : cudnn_handle_) {
      if (handle != nullptr)
        CHECK_EQ(cudnnDestroy(handle), CUDNN_STATUS_SUCCESS);
      handle = nullptr;
    }
#endif
#endif
    for (auto& entry : rand_generator_) {
      if (entry.second != nullptr) {
        delete entry.second;
        entry.second = nullptr;
      }
    }
  }
  /**
   * Constructor, init handlers and GPU rand generators to nullptr.
   */
  Context() {
    for (int i = 0; i < kMaxNumGPU; i++) {
#ifdef USE_GPU
      cublas_handle_.push_back(nullptr);
      curand_generator_.push_back(nullptr);
#ifdef USE_CUDNN
      cudnn_handle_.push_back(nullptr);
#endif
#endif
    }
  }

  /**
   * @return the device ID of the current thread.
   */
  int device_id() {
    return device_id(std::this_thread::get_id());
  }
  /**
   * @return the ID of the device attached to a given CPU thread, or -1 if this
   * thread has not been attached GPU device.
   */
  int device_id(const std::thread::id& tid) {
    if (device_id_.find(tid) != device_id_.end())
      return device_id_[tid];
    else
      return -1;
  }
  /**
   * Setup the CPU thread, which may be assigned a GPU device.
   * If there is no GPU device, then set did to -1.
   * Set the random seed to -1.
   * @param[in] thread::id CPU thread ID
   * @param[in] device_id GPU device ID
   */
  void SetupDevice(const std::thread::id& tid, const int did) {
    SetupDevice(tid, did, -1);
  }
  /**
   * @copy SetupDevice(const int, const int);
   * @param[in] seed random seed
   */
  void SetupDevice(const std::thread::id& tid, const int did, const int seed) {
    device_id_[tid] = did;
    seed_[tid] = seed;
  }

  /**
   * Activate the GPU device by calling cudaSetDevice.
   */
  void ActivateDevice(const int device_id) {
    CHECK_GE(device_id, 0);
#ifdef USE_GPU
    cudaSetDevice(device_id);
#endif
  }

  /**
   * \copybreif rand_generator(const std::thread::id&);
   * @return the CPU random generator for the calling thread.
   */
  std::mt19937* rand_generator() {
    return rand_generator(std::this_thread::get_id());
  }
  /**
   * Get the CPU random generator.
   * If the generator does not exist, then create it now.
   * If the seed is not set, i.e., seed=-1, then get a seed from system time.
   * @param[in] thread::id CPU thread ID
   * @return the CPU random generator
   */
  std::mt19937* rand_generator(const std::thread::id& tid) {
    if (rand_generator_.find(tid) == rand_generator_.end()) {
      // CHECK(seed_.find(tid) != seed_.end());
      auto seed = static_cast<unsigned>(seed_[tid]);
      if (seed_.find(tid) == seed_.end() || seed_.at(tid) == -1)
        seed = std::chrono::system_clock::now().time_since_epoch().count();
      rand_generator_[tid] = new std::mt19937(seed);
    }
    return rand_generator_[tid];
  }
#ifdef USE_GPU
  /**
   * \copybreif cublas_handle_(const std::thread::id&);
   * @return cublas handle for the calling thread.
   */
  cublasHandle_t cublas_handle() {
    return cublas_handle(std::this_thread::get_id());
  }
  /**
   * Get the handler of the GPU which is assigned to the given thread.
   * Calls cublas_handle(const int);
   */
  cublasHandle_t cublas_handle(const std::thread::id thread_id) {
    return cublas_handle(device_id(thread_id));
  }
  /**
   * Get the handler of the GPU device given its device ID. The device
   * must be set up via SetupDevice(const std::thread::id, const int) before
   * calling this function.
   * @param[in] device_id GPU device ID
   * @return the GPU handler
   */
  cublasHandle_t cublas_handle(const int device_id) {
    CHECK_GE(device_id, 0);
    if (cublas_handle_.at(device_id) == nullptr) {
      cudaSetDevice(device_id);
      cublasCreate(&cublas_handle_[device_id]);
    }
    return cublas_handle_[device_id];
  }
  /**
   * Get the rand generator of the GPU device assigned to the given thread.
   */
  curandGenerator_t curand_generator(const std::thread::id thread_id) {
    return curand_generator(device_id(thread_id));
  }
  /**
   * Get the random generator of the GPU device given the device id.
   * @param[in] device_id GPU device ID
   * @return random generator. If it does not exist, then create one.
   * The random seed will be set to CURAND_RNG_PSEUDO_DEFAULT if it is not set.
   */
  curandGenerator_t curand_generator(const int device_id) {
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, cudnn_handle_.size());
    if (curand_generator_.at(device_id) == nullptr) {
      // TODO(wangwei) handle user set seed
      /*
      CHECK(seed_.find(tid) != seed_.end());
      auto seed = seed_[tid];
      */
      ActivateDevice(device_id);
      curandCreateGenerator(&curand_generator_[device_id],
          CURAND_RNG_PSEUDO_DEFAULT);
    }
    return curand_generator_[device_id];
  }

#ifdef USE_CUDNN
  cudnnHandle_t cudnn_handle() {
    return cudnn_handle(std::this_thread::get_id());
  }

  cudnnHandle_t cudnn_handle(const std::thread::id thread_id) {
    return cudnn_handle(device_id(thread_id));
  }

  cudnnHandle_t cudnn_handle(const int device_id) {
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, cudnn_handle_.size());
    if (cudnn_handle_.at(device_id) == nullptr) {
      ActivateDevice(device_id);
      // LOG(ERROR) << "create cudnn handle for device " << device_id;
      CHECK_EQ(cudnnCreate(&cudnn_handle_[device_id]), CUDNN_STATUS_SUCCESS);
    }
    // LOG(ERROR) << "use cudnn handle from device " << device_id;
    return cudnn_handle_[device_id];
  }
#endif

#endif

 protected:
  //!< max num of GPUs per process
  const int kMaxNumGPU = 64;
  //!< map from thread id to device id
  std::unordered_map<std::thread::id, int> device_id_;
  //!< map from thread id to cpu rand generator
  std::unordered_map<std::thread::id, std::mt19937 *> rand_generator_;
  //!< map from thread id to cpu rand generator seed
  std::unordered_map<std::thread::id, int> seed_;
#ifdef USE_GPU
  //!< cublas handler indexed by GPU device ID
  std::vector<cublasHandle_t> cublas_handle_;
  //!< cublas rand generator indexed by GPU device ID
  std::vector<curandGenerator_t> curand_generator_;

#ifdef USE_CUDNN
  std::vector<cudnnHandle_t> cudnn_handle_;
#endif
#endif
};

}  // namespace singa

#endif  // SINGA_UTILS_CONTEXT_H_
