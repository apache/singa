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

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#endif


namespace singa {

const int kDefaultDevice = 20;

class Context {
  public:

	~Context();

	void Setup();

#ifdef USE_GPU
	int DeviceID(const int index) {
	  return device_ids_[index];
	}

	void SetDeviceID(const int index, const int id) {
	  device_ids_[index] = id;
	}

	void SetDevice(const int index) {
	  cudaSetDevice(device_ids_[index]);
	}

	cublasHandle_t Handle(const int index) {
	  return handles_[index];
	}

	void CreateHandle(const int index);

	void DestoryHandle(const int index);

	curandGenerator_t GpuRandGenerator(const int index) {
	  return gpu_rand_generators_[index];
	}

	void CreateGpuRandGenerator(const int index);

	void DestoryGpuRandGenerator(const int index);

#endif

  protected:
	std::vector<int> device_ids_;
#ifdef USE_GPU
	std::vector<cublasHandle_t> handles_;
	std::vector<curandGenerator_t> gpu_rand_generators_;
#endif

};

}  // namespace singa

#endif  // SINGA_UTILS_MATH_ADDR_H_
