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

#ifndef SINGA_CORE_PLATFORM_H_
#define SINGA_CORE_PLATFORM_H_

#include <memory>
#include <vector>

#include "singa/core/device.h"
#include "singa/singa_config.h"

#ifdef USE_CUDA
#include "singa/utils/cuda_utils.h"
#endif // USE_CUDA

#ifdef USE_OPENCL
#include <cl/cl2.hpp>
#endif // USE_OPENCL

namespace singa {

/// This class queries all available calculating devices on a given machine
/// grouped according to manufacturer or device drivers. All methods should be static.
/// If CUDA or OPENCL are not enabled, then the respective related methods should
/// return something that indicates their absence (for example, 0 devices); 
/// however they should always be available regardless of compile-time switches.
class Platform {
public:

  /// Constructor.
  Platform();

  /// Return the number of total available GPUs
  static int GetNumGPUs();

  /// Return the device IDs of available GPUs.
  /// TODO(wangwei) return the IDs according to free memory in decending order
  static const std::vector<int> GetGPUIDs();

  static const std::pair<size_t, size_t> GetGPUMemSize(const int device);

  /// Return the memory of a GPU <free, total>
  static const std::vector<std::pair<size_t, size_t>> GetGPUMemSize();

  /// Return a string containing all hardware info, e.g., version, memory size.
  static const std::string DeviceQuery(int id, bool verbose = false);

  /// Create a set of CudaGPU Device using 'num_devices' free GPUs.
  static const std::vector<std::shared_ptr<Device>>
  CreateCudaGPUs(const size_t num_devices, size_t init_size = 0);

  /// Create a set of CudaGPU Device using given GPU IDs.
  static const std::vector<std::shared_ptr<Device>>
  CreateCudaGPUs(const std::vector<int> &devices, size_t init_size = 0);

  /// Create a \p num_devices set of valid OpenCL devices, regardless of platforms.
  /// If there are fewer valid devices than requested, then this method will return as many as possible.
  /// If OpenCL is not in use, this method will return an empty array.
  const std::vector<std::shared_ptr<Device>> CreateOpenclDevices(const size_t num_devices);

  /// Create a set of valid OpenCL devices, regardless of platforms, assigning \p id to each device in sequence.
  /// If there are fewer valid devices than requested, then this method will return as many as possible.
  /// If OpenCL is not in use, this method will return an empty array.
  const std::vector<std::shared_ptr<Device>> CreateOpenclDevices(const vector<int>& id);

  /// This function is implementd by Caffe (http://caffe.berkeleyvision.org/).
  /// This function checks the availability of GPU #device_id.
  /// It attempts to create a context on the device by calling cudaFree(0).
  /// cudaSetDevice() alone is not sufficient to check the availability.
  /// It lazily records device_id, however, does not initialize a
  /// context. So it does not know if the host thread has the permission to use
  /// the device or not.
  ///
  /// In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  /// or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  /// even if the device is exclusively occupied by another process or thread.
  /// Cuda operations that initialize the context are needed to check
  /// the permission. cudaFree(0) is one of those with no side effect,
  /// except the context initialization.
  static bool CheckDevice(const int device_id);


private:
  cl::Platform clPlatform;
};

} // namespace singa

#endif // SINGA_CORE_PLATFORM_H_
