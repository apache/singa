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

#ifndef SINGA_CORE_OPENCL_DEVICE_H_
#define SINGA_CORE_OPENCL_DEVICE_H_

#include "device.h"

#ifdef USE_OPENCL

// http://github.khronos.org/OpenCL-CLHPP/
// cl2.hpp includes cl.h, do not re-include.
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <map>
#include <CL/cl2.hpp>

#include "singa/utils/opencl_utils.h"

namespace singa {

// Fixed paths and other constants.
constexpr char* kernel_path = "src/core/math";

// Implement Device using OpenCL libs.
class OpenclDevice : public Device {
public:

  // TODO: Constructor arguments to consider:
  // Path to kernel sources?
  // Select only certain device types?
  OpenclDevice(int id = 0, int num_executors = 1,
			   string scheduler = "sync", string vm = "gc-only");
  ~OpenclDevice();

  /// Get the specified kernel.
  cl::Kernel GetKernel(const std::string& kname);

  /// Get the command queue associated with this device.
  cl::CommandQueue cmdq() { return cmdQueue; }

  /// Prints information about all Devices in each Platform.
  void PrintAllDeviceInfo();

  /// Prints status about CL source code builds.
  void PrintClBuildInfo(const cl::Program &p);

// Inherited methods
  void SetRandSeed(unsigned seed) override;

protected:
  cl::Device this_device;						/// The OpenCL device that this object represents.
  cl::CommandQueue cmdQueue;					/// The CommandQueue that is associated with this device.
  std::map<std::string, cl::Kernel> kernels;	/// A list of kernels that has been compiled on this device.

  /// Initializes the OpenCL device by retrieving all details at once,
  /// and checking for respective errors. Called only by the constructor.
  /// Init will also filter for OpenCL 2.0 platforms, which means that 
  /// NVidia devices that only support 1.2 will be excluded.
  /// Init will also call PrintPlatformInfo.
  void Init();
  
  /// Searches the given paths for all .cl files and builds
  /// OpenCL programs, then stores them in the Kernels map.
  void BuildPrograms(const std::string &kdir = ".");
  

// Inherited methods
//  void DoExec(function<void(Context*)>&& fn, int executor) override;

  /// 
  void* Malloc(int size) override;

  /// Converts the void pointer into a cl::Buffer object, then deletes the object.
  /// This has the effect of freeing up device memory.
  void Free(void* p) override;
};

} // namespace singa

#endif // USE_OPENCL

#endif // SINGA_CORE_OPENCL_DEVICE_H_
