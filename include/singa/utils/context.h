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

#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <glog/logging.h>


#ifdef USE_GPU
#include "singa/utils/cuda_utils.h"
#endif


namespace singa {

// max num of threads per process
const int kNumMaxThreads = 1024;

/**
 * Context is used as a global singleton, which stores the mapping from CPU
 * thread id to GPU device id. It manages the handlers for GPU
 * devices. It also manages the GPU and CPU random generators, which are created
 * when accessed. One CPU thread has a CPU random generator. A CPU device
 * has a GPU random generator.
 */
class Context {
 public:
   /**
    * Destructor, release random generators and handlers.
    */
	~Context();
  /**
   * Constructor, init arrays for random generators and handlers.
   */
  Context();

  /**
   * @return the ID of the device attached to a given CPU thread:
   * if the device is a GPU card, then returns the GPU device ID;
   * Else return -1.
   */
	int device_id(const std::thread::id tid) {
    CHECK(device_id_.find(tid) != device_id_.end());
	  return device_id_[tid];
	}

  /**
   * Setup the CPU thread, which may be assigned a GPU device.
   * Set the random seed to -1.
   * A GPU handler will be created for the GPU device.
   * @param[in] thread::id CPU thread ID
   * @param[in] device_id GPU device ID
   */
	void SetupDevice(const std::thread::id tid, const int did);

  /**
   * @copy SetupDevice(const int, const int);
   * @param[in] seed random seed
   */
  void SetupDevice(const std::thread::id tid, const int did, long long seed);

  /**
   * Get the CPU random generator.
   * If the generator does not exist, then create it now.
   * If the seed is not set, i.e., seed=-1, then get a seed from system time.
   * @param[in] thread::id CPU thread ID
   * @return the CPU random generator
   */
  std::mt19937* rand_generator(const std::thread::id tid) {
    if (rand_generator_.find(tid) == rand_generator_.end()) {
      CHECK(seed_.find(tid) != seed_.end());
      auto seed = static_cast<unsigned>(seed_[tid]);
      if (seed_[tid] == -1)
        seed = std::chrono::system_clock::now().time_since_epoch().count();
      rand_generator_[tid] = new std::mt19937(seed);
    }
    return rand_generator_[tid];
  }
#ifdef USE_GPU
  /**
   * Get the handler of the GPU device attached to a CPU thread.
   * @param[in] thread::id
   * @return the GPU handler, or nullptr if this thread does not have any GPU.
   */
	cublasHandle_t cublas_handle(const std::thread::id tid) {
    CHECK(cublas_handle_.find(tid) != cublas_handle_.end());
	  return cublas_handle_[tid];
	}
  /**
   * Get the random generator of the GPU device assigned to the given thread.
   * @param[in] thread::id
   * @return random generator. If it does not exist, then create one.
   * The random seed will be set to CURAND_RNG_PSEUDO_DEFAULT if it is not set.
   */
	curandGenerator_t curand_generator(const std::thread::id tid) {
    if (curand_generator_.find(tid) == curand_generator_.end()) {
      CHECK(seed_.find(tid) != seed_.end());
      auto seed = seed_[tid];
      // TODO handle user set seed
      cudaSetDevice(device_id_[tid]);
      curandCreateGenerator(&curand_generator_[tid], CURAND_RNG_PSEUDO_DEFAULT);
    }
	  return curand_generator_[tid];
	}

  /*
 protected:
	void CreateHandle(const int thread::id);
	void DestoryHandle(const int thread::id);
	void CreateGpuRandGenerator(const int thread::id);
	void DestoryGpuRandGenerator(const int thread::id);
  */

#endif

 protected:

	std::unordered_map<std::thread::id, int> device_id_;
  std::unordered_map<std::thread::id, std::mt19937 *> rand_generator_;
  std::unordered_map<std::thread::id, int> seed_;
#ifdef USE_GPU
	std::unordered_map<std::thread::id, cublasHandle_t> cublas_handle_;
	std::unordered_map<std::thread::id, curandGenerator_t> curand_generator_;
#endif
};

}  // namespace singa

#endif  // SINGA_UTILS_MATH_ADDR_H_
