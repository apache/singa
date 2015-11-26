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
#include "singa/utils/context.h"
#include "singa/utils/factory.h"
#include "singa/utils/singleton.h"

namespace singa {

Context::~Context() {
#ifdef USE_GPU
  for (auto& entry ： device_id_) {
    if (entry.second != -1) {
      cudaSetDevice(entry.second);
      if (cublas_handle_[entry.first] != nullptr) {
        cublasDestroy(cublas_handle_[entry.first]);
        cublas_handle_[entry.first] = nullptr;
      }
      if(curand_generator_[entry.first] != nullptr) {
        curandDestroyGenerator(curand_generator_[entry.first]);
        curand_generator_[entry.first] = nullptr;
      }
    }
  }
#endif
  for (auto& entry : rand_generator_) {
    if (entry.second != nullptr) {
      delete entry.second;
      entry.second = nullptr;
    }
  }
}

Context::Context() { }

void Context::SetupDevice(const std::thread::id thread, const int did) {
  SetupDevice(thread, did, -1);
}

void Context::SetupDevice(const std::thread::id thread, const int did,
    long long seed) {
  device_id_[thread] = did;
#ifdef USE_GPU
  if (did > -1) {
    cudaSetDevice(did);
    cublasCreate(&handle_[thread]);
  }
#endif
  seed_[thread] = seed;
}

/*
#ifdef USE_GPU
void Context::DestoryHandle(const int thread::id) {
  cudaSetDevice(device_id_[thread::id]);
  cublasDestroy(handle_[thread::id]);
  handle_[thread::id] = nullptr;
}

void Context::DestoryGpuRandGenerator(const int thread::id) {
  cudaSetDevice(device_id_[thread::id]);
  curandDestroyGenerator(curand_generator_[thread::id]);
  curand_generator_[thread::id] = nullptr;
}
#endif
*/


}  // namespace singa

