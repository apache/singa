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


#include "singa/core/memory.h"
#include "singa/utils/logging.h"
#include <iostream>

namespace singa {

bool singa::CnMemPool::initialized = false;
std::mutex singa::CnMemPool::mtx;

void CnMemPool::InitPool(int numDevices, size_t initSize, unsigned flag) {
	mtx.lock();
	if(!initialized) {
		CHECK_GE(numDevices, 1);
		cnmemDevice_t* settingPtr = new cnmemDevice_t[numDevices];
		for(int i = 0; i < numDevices; i++) {
			settingPtr[i].device = i;
			settingPtr[i].size = initSize;
			settingPtr[i].numStreams = 0;
			settingPtr[i].streams = NULL;
			settingPtr[i].streamSizes = 0;
		}
		cnmemStatus_t status = cnmemInit(numDevices, settingPtr, flag);
		CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS) << " " << cnmemGetErrorString(status);
		delete[] settingPtr;
		initialized = true;
	}
	mtx.unlock();
}

void CnMemPool::InitPool() {
		int defaultNumDevices = 1;
		size_t defaultSize = 1000000U;
		InitPool(defaultNumDevices,defaultSize,cnmemManagerFlags_t::CNMEM_FLAGS_DEFAULT);
}

CnMemPool::~CnMemPool() {
	mtx.lock();
	if(initialized) {
		cnmemStatus_t status = cnmemFinalize();
		CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS) << " " << cnmemGetErrorString(status);
		initialized = false;
	}
	mtx.unlock();
	LOG(INFO) << "cnmem has been freed";
}


void CnMemPool::Malloc(void** ptr, const size_t size) {
	cnmemStatus_t status = cnmemMalloc(ptr,size,NULL);
	CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS) << " " << cnmemGetErrorString(status);
}

void CnMemPool::Free(void* ptr) {
	LOG(INFO) << "cnmem free is called !!!!!!!!!!!";
	cnmemStatus_t status = cnmemFree(ptr,NULL);
	CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS) << " " << cnmemGetErrorString(status);
	LOG(INFO) << "cnmem free is terminated";
}

void CudaMemPool::Malloc(void** ptr, const size_t size) {
	cudaError_t status = cudaMalloc(ptr,size);
	CHECK_EQ(status, cudaError_t::cudaSuccess);
}

void CudaMemPool::Free(void* ptr) {
	cudaError_t status = cudaFree(ptr);
	CHECK_EQ(status, cudaError_t::cudaSuccess);
}

}
