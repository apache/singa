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
#ifndef DISABLE_WARNINGS

#include <iostream>

#include "singa/core/device.h"
#include "singa/singa_config.h"
#include "singa/utils/opencl_utils.h"
using namespace std;
namespace singa {

#ifdef USE_CUDA

std::vector<std::shared_ptr<Device>> Platform::UsedDevice;
std::mutex Platform::mtx_;
int Platform::GetNumGPUs() {
  int count;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

bool Platform::CheckDevice(const int device_id) {
  bool r = ((cudaSuccess == cudaSetDevice(device_id)) &&
            (cudaSuccess == cudaFree(0)));
  // reset any error that may have occurred.
  cudaGetLastError();
  return r;
}

/// Return the total num of free GPUs
const vector<int> Platform::GetGPUIDs() {
  vector<int> gpus;
  int count = Platform::GetNumGPUs();
  for (int i = 0; i < count; i++) {
    if (Platform::CheckDevice(i)) {
      gpus.push_back(i);
    }
  }
  return gpus;
}

const std::pair<size_t, size_t> Platform::GetGPUMemSize(const int device) {
  std::pair<size_t, size_t> ret{0, 0};
  if (Platform::CheckDevice(device)) {
    CUDA_CHECK(cudaSetDevice(device));
    size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    ret = std::make_pair(free, total);
  } else {
    LOG(ERROR) << "The device (ID = " << device << ") is not available";
  }
  return ret;
}

const vector<std::pair<size_t, size_t>> Platform::GetGPUMemSize() {
  vector<std::pair<size_t, size_t>> mem;
  int count = Platform::GetNumGPUs();
  for (int i = 0; i < count; i++) {
    mem.push_back(Platform::GetGPUMemSize(i));
  }
  return mem;
}

const string Platform::DeviceQuery(int device, bool verbose) {
  if (cudaSuccess != cudaGetDevice(&device)) {
    return "The device (ID = " + std::to_string(device) + " is not available";
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::ostringstream out;
  out << "Device id:                     " << device << '\n';
  out << "Total global memory:           " << prop.totalGlobalMem << '\n';
  out << "Total shared memory per block: " << prop.sharedMemPerBlock << '\n';
  out << "Maximum threads per block:     " << prop.maxThreadsPerBlock << '\n';
  out << "Maximum dimension of block:    " << prop.maxThreadsDim[0 << '\n']
      << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << '\n';
  out << "Maximum dimension of grid:     " << prop.maxGridSize[0] << ", "
      << "Concurrent copy and execution: "
      << (prop.deviceOverlap ? "Yes" : "No") << '\n';

  if (verbose) {
    out << "Major revision number:         " << prop.major << '\n';
    out << "Minor revision number:         " << prop.minor << '\n';
    out << "Name:                          " << prop.name << '\n';
    out << "Total registers per block:     " << prop.regsPerBlock << '\n';
    out << "Maximum memory pitch:          " << prop.memPitch << '\n';
    out << "Warp size:                     " << prop.warpSize
        << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << '\n';
    out << "Clock rate:                    " << prop.clockRate << '\n';
    out << "Number of multiprocessors:     " << prop.multiProcessorCount
        << '\n';
    out << "Kernel execution timeout:      "
        << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << '\n';
  }
  return out.str();
}

const vector<shared_ptr<Device>> Platform::CreateCudaGPUs(
    const size_t num_devices, size_t init_size) {
  const vector<int> gpus = GetGPUIDs();
  CHECK_LE(num_devices, gpus.size());
  vector<int> use_gpus(gpus.begin(), gpus.begin() + num_devices);
  return CreateCudaGPUsOn(use_gpus, init_size);
}

const vector<shared_ptr<Device>> Platform::CreateCudaGPUsOn(
    const vector<int>& devices, size_t init_size) {
  MemPoolConf conf;
  if (init_size > 0) conf.set_init_size(init_size);
  size_t bytes = conf.init_size() << 20;
  for (auto device : devices) {
    conf.add_device(device);
    CHECK_LE(bytes, Platform::GetGPUMemSize(device).first);
  }
  mtx_.lock();
  if (UsedDevice.size() == 0) {
    int count = Platform::GetNumGPUs();
    for (int i = 0; i < count; i++) UsedDevice.push_back(nullptr);
  }
  auto pool = std::make_shared<CnMemPool>(conf);
  vector<shared_ptr<Device>> ret;
  for (size_t i = 0; i < devices.size(); i++) {
    if (UsedDevice[devices[i]] == nullptr)
      UsedDevice[devices[i]] = std::make_shared<CudaGPU>(devices[i], pool);
    else
      UsedDevice[devices[i]]->Reset();
    ret.push_back(UsedDevice[devices[i]]);
  }
  mtx_.unlock();
  return ret;
}

#endif  // USE_CUDA

#ifdef USE_OPENCL

const int Platform::GetNumOpenclPlatforms() {
  auto all_platforms = viennacl::ocl::get_platforms();
  return (int)all_platforms.size();
}

const int Platform::GetNumOpenclDevices() {
  auto all_platforms = viennacl::ocl::get_platforms();
  unsigned int total_num_devices = 0;
  for (auto plat : all_platforms) {
    auto all_devices = plat.devices(CL_DEVICE_TYPE_ALL);
    total_num_devices += all_devices.size();
  }
  return (int)total_num_devices;
}

const std::shared_ptr<Device> Platform::GetDefaultOpenclDevice() {
  return std::make_shared<OpenclDevice>();
}
/*
static const std::vector<std::shared_ptr<Device>>
Platform::CreateOpenclDevices(const size_t num_devices) {
  auto all_platforms = viennacl::ocl::get_platforms();
  for (auto plat : all_platforms) {
    auto all_devices = plat.devices(CL_DEVICE_TYPE_ALL);
    total_num_devices += all_devices.size();
  }
  return (int)total_num_devices;
}

static const std::vector<std::shared_ptr<Device>>
Platform::CreateOpenclDevices(const std::vector<int> &id) {

}
*/
#endif  // USE_OPENCL

}  // namespace singa

#endif
