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

#ifndef SINGA_CORE_DEVICE_H_
#define SINGA_CORE_DEVICE_H_

#include <type_traits>
#include <vector>
#include <string>
#include <functional>
#include <memory>

#include "singa/singa_config.h"
#include "singa/core/common.h"
#include "singa/core/memory.h"
#include "singa/core/scheduler.h"
#include "singa/proto/core.pb.h"

#ifdef USE_CUDA
#include "singa/utils/cuda_utils.h"
#endif // USE_CUDA

#ifdef USE_OPENCL
#include "singa/utils/opencl_utils.h"
#endif // USE_OPENCL

using std::vector;
using std::string;
using std::function;
using std::shared_ptr;

namespace singa {

/// Allocate memory and execute Tensor operations.
/// There are three types of devices distinguished by their programming
/// languages, namely cpp, cuda and opencl.
class Device {
  public:
  // Device() = default;
  virtual ~Device() {}
  /// Constructor with device ID, num of executors (e.g., cuda streams),
  /// max mem size to use (in MB)
  Device(int id, int num_executors);

  virtual void SetRandSeed(unsigned seed) = 0;

  /// Called by Tensor.
  Block* NewBlock(int size);

  /// Called by Tensor.
  void FreeBlock(Block* block);

  /// Return the size (bytes) of memory in use
  /// TODO(wangwei) override this function for all devices.
  virtual size_t GetAllocatedMem() {
    return 0u;
  }

  /// Copy data within or across devices.
  virtual void CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                      CopyDirection direction, int dst_offset, int src_offset);

  void CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes,
                           size_t dst_offset = 0);
  /// Submit the operation to the device, which may execute it right now or
  /// delay it depending on the scheduler.
  void Exec(function<void(Context*)>&& fn, const vector<Block*> read_blocks,
                    const vector<Block*> write_blocks,
                    bool use_rand_generator = false);

  // Wait for one event.
  // void WaitFor();

  /// wait for all operations submitted to this device.
  void Sync();

  /// Return the programming language for this device.
  LangType lang() const {
    return lang_;
  }

  virtual std::shared_ptr<Device> host() const { return host_;}

  Context* context(int k) {
    return &ctx_;
  }

  int id() const { return id_; }

 private:
  Device() {};

 protected:
  /// Execute one operation on one executor.
  virtual void DoExec(function<void(Context*)>&& fn, int executor) = 0;

  virtual void CopyToFrom(void* dst, const void* src, size_t nBytes,
                          CopyDirection direction, Context* ctx) = 0;

  /// Allocate device memory.
  virtual void* Malloc(int size) = 0;

  /// Free device memory.
  virtual void Free(void* ptr) = 0;

 protected:
  int id_ = 0;
  int num_executors_ = 0;
  unsigned seed_ = 0;
  // Scheduler* scheduler_ = nullptr;
  // VirtualMemory* vm_ = nullptr;
  /// Programming language type, could be kCpp, kCuda, kOpencl
  LangType lang_;
  // SafeQueue<Operation> op_queue_;
  // SafeQueue<Operation> op_log_;
  /// The host device
  std::shared_ptr<Device> host_;
  // TODO(wangwei) define multiple contexts, one per executor
  Context ctx_;
};

/// a singleton CppDevice as the host for all devices.
extern std::shared_ptr<Device> defaultDevice;

/// Represent a CPU device which may have multiple threads/executors.
/// It runs cpp code.
class CppCPU : public Device {
 public:
  ~CppCPU() {};
  CppCPU();

  std::shared_ptr<Device> host() const override { return defaultDevice;}
  void SetRandSeed(unsigned seed) override;

 protected:
  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) override;

  /// Allocate cpu memory.
  void* Malloc(int size) override;

  /// Free cpu memory.
  void Free(void* ptr) override;
};


// Implement Device using OpenCL libs.
// class OpenclDevice : public Device { };

#ifdef USE_CUDA
// Represent a Nvidia GPU which runs cuda code.
class CudaGPU : public Device {
 public:
  ~CudaGPU();
  /// Construct the device using default mem pool setting.
  CudaGPU(int id = 0);
  /// Construct the device given the physical device ID and memory pool.
  CudaGPU(int id, std::shared_ptr<DeviceMemPool> pool);

  void SetRandSeed(unsigned seed) override;
  size_t GetAllocatedMem() override;

 protected:
  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) override;

  /// Allocate cpu memory.
  void* Malloc(int size) override;

  /// Free cpu memory.
  void Free(void* ptr) override;

 private:
  void Setup();

 private:
	shared_ptr<DeviceMemPool> pool_;
};

/// CudaCPU which uses cudaMallocHost to allocate pinned memory for host.

#endif  // USE_CUDA

#ifdef USE_OPENCL

// Implement Device using OpenCL libs.
class OpenclDevice : public singa::Device {
public:

  // TODO: Constructor arguments to consider:
  // Path to kernel sources?
  // Select only certain device types?
  OpenclDevice(int id = 0, int num_executors = 1);
  ~OpenclDevice();

// Overridden, inherited methods
  void SetRandSeed(unsigned seed) override;

  virtual void CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                      CopyDirection direction, int dst_offset = 0,
                      int src_offset = 0) override;

protected:
  /// The OpenCL device that this object represents.
  /// Each OpenclDevice contains exactly one cl::Device for the lifetime of the
  /// object.
  viennacl::ocl::device this_device;

  /// Each OpenclDevice has one OpenCL context. It is created along with the
  /// creation of this object.
  viennacl::ocl::context vcl_ctx;

  /// Searches the given paths for all .cl files and builds
  /// OpenCL programs, then stores them in the Kernels map.
  void BuildPrograms();

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

  static const std::string cl_src_path;
};
#endif  // USE_OPENCL

/// This class queries all available calculating devices on a given machine
/// grouped according to manufacturer or device drivers. All methods should be static.
/// If CUDA or OPENCL are not enabled, then the respective related methods should
/// return something that indicates their absence (for example, 0 devices);
/// however they should always be available regardless of compile-time switches.
class Platform {
public:

  /// Return the default host device
  static std::shared_ptr<Device> GetDefaultDevice() {
    return defaultDevice;
  }

#ifdef USE_CUDA
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
  CreateCudaGPUsOn(const std::vector<int> &devices, size_t init_size = 0);
  
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
#endif // USE_CUDA

#ifdef USE_OPENCL

  const int GetNumOpenclPlatforms();
  
  const int GetNumOpenclDevices();
  
  static const std::shared_ptr<Device> GetDefaultOpenclDevice();

  /// Create a \p num_devices set of valid OpenCL devices, regardless of
  /// platforms.  If there are fewer valid devices than requested, then this
  /// method will return as many as possible. If OpenCL is not in use, this
  /// method will return an empty array.
//  static const std::vector<std::shared_ptr<Device>>
//  CreateOpenclDevices(const size_t num_devices);

  /// Create a set of valid OpenCL devices, regardless of platforms, assigning
  /// \p id to each device in sequence.
  /// If there are fewer valid devices than requested, then this method will
  /// return as many as possible.
  /// If OpenCL is not in use, this method will return an empty array.
//  const std::vector<std::shared_ptr<Device>>
//  CreateOpenclDevices(const vector<int> &id);
#endif // USE_OPENCL

};


}  // namespace singa

#endif  // SINGA_CORE_DEVICE_H_
