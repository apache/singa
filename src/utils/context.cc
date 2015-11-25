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
  for(int i = 0; i < kDefaultDevice; ++i) {
	SetDevice(i);

	if(handles_[i] != NULL) {
	  cublasDestroy(handles_[i]);
	}

	if(gpu_rand_generators_[i] != NULL) {
      curandDestroyGenerator(gpu_rand_generators_[i]);
	}
  }
#endif
}

void Context::Setup() {

  for(int i = 0; i < kDefaultDevice; ++i) {
	//init device index
	device_ids_.push_back(i);
  }

#ifdef USE_GPU
  for(int i = 0; i < kDefaultDevice; ++i) {
	//init handle
	cublasHandle_t handle = NULL;
	handles_.push_back(handle);

	curandGenerator_t gpu_rand_generator = NULL;
	gpu_rand_generators_.push_back(gpu_rand_generator);
  }
#endif
}

#ifdef USE_GPU
void Context::CreateHandle(const int index) {
  SetDevice(device_ids_[index]);
  cublasCreate(&handles_[index]);
}

void Context::DestoryHandle(const int index) {
  SetDevice(device_ids_[index]);
  cublasDestroy(handles_[index]);
  handles_[index] = NULL;
}

void Context::CreateGpuRandGenerator(const int index) {
  SetDevice(device_ids_[index]);
  curandCreateGenerator(&gpu_rand_generators_[index], CURAND_RNG_PSEUDO_DEFAULT);
}

void Context::DestoryGpuRandGenerator(const int index) {
  SetDevice(device_ids_[index]);
  curandDestroyGenerator(gpu_rand_generators_[index]);
  gpu_rand_generators_[index] = NULL;
}

#endif


}  // namespace singa

