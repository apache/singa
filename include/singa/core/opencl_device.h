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

#include "singa/core/device.h"

#ifdef USE_OPENCL
// http://github.khronos.org/OpenCL-CLHPP/
// cl2.hpp includes cl.h, do not re-include.
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <map>
#include <memory>
#include <CL/cl2.hpp>

#include "singa/utils/opencl_utils.h"

namespace singa {

// Implement Device using OpenCL libs.
class OpenclDevice : public singa::Device {
public:

  // TODO: Constructor arguments to consider:
  // Path to kernel sources?
  // Select only certain device types?
  OpenclDevice(int id = 0, int num_executors = 1);
  ~OpenclDevice();

  /// Get the specified kernel.
  cl::Kernel GetKernel(const std::string& kname, cl_int* status = nullptr);

  /// Get the command queue associated with this device.
  cl::CommandQueue GetCmdQ() { return cmdq; }

  /// Prints information about all Devices in each Platform.
  void PrintAllDeviceInfo();

  /// Prints status about CL source code builds.
  void PrintClBuildInfo(cl::Program &p);

// Overridden, inherited methods
  void SetRandSeed(unsigned seed) override;

  void CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                      CopyDirection direction, int dst_offset = 0, 
                      int src_offset = 0);
/*
  void CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes = 0, 
                           size_t dst_offset = 0) override;*/

protected:
  /// The OpenCL device that this object represents.
  /// Each OpenclDevice contains exactly one cl::Device for the lifetime of the
  /// object.
  cl::Device this_device;

  /// Each OpenclDevice has one OpenCL context. It is created along with the 
  /// creation of this object.
  cl::Context ocl_ctx;

  /// The CommandQueue that is associated with this device.
  /// Since each OpenclDevice contains only one cl::Device and one cl::Context,
  /// it naturally also contains one cl::CommandQueue that is associated
  /// with said Device and Context. 
  cl::CommandQueue cmdq;
  
  /// A list of kernels that has been compiled on this device.
  std::shared_ptr<std::map<std::string, cl::Kernel>> kernels;
  
  /// Searches the given paths for all .cl files and builds
  /// OpenCL programs, then stores them in the Kernels map.
  void BuildPrograms(const std::string &kdir = cl_src_path);

// Overridden, inherited methods.

  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx = nullptr) override;

  /// Allocates memory on this OpenCL device
  /// by creating and returning an empty cl::Buffer object.
  /// with the indicated size.
  void* Malloc(int size) override;

  /// Converts the void pointer into a Buffer object, then deletes the object.
  /// This has the effect of freeing up device memory.
  void Free(void* ptr) override;

private:

  /// Copies a data block from host to device.
  /// src: a pointer to an array of data.
  /// dst: a pointer to a cl::Buffer object.
  void WriteToDevice(cl::Buffer* dst, const void* src, const size_t size);

  /// Reads a data block from device to host.
  /// src: a pointer to an cl::Buffer object.
  /// dst: a pointer to an malloc'ed empty array.
  void ReadFromDevice(void* dst, const cl::Buffer* src, const size_t size);

  /// Duplicates a block of data on the device.
  /// src: a pointer to the original cl::Buffer object.
  /// dst: a pointer to the new cl::Buffer object to copy the data into.
  void CopyDeviceBuffer(cl::Buffer* dst, const cl::Buffer* src, const size_t size);

  static const std::string cl_src_path;
};

} // namespace singa

#endif // USE_OPENCL

#endif // SINGA_CORE_OPENCL_DEVICE_H_
