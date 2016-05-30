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
#include "singa_config.h"
#include "singa/core/common.h"
#include "singa/core/memory.h"
#include "singa/core/scheduler.h"
#include "singa/proto/core.pb.h"

using std::vector;
using std::string;
using std::function;
namespace singa {
/// Allocate memory and execute Tensor operations.
/// There are three types of devices distinguished by their programming
/// languages, namely cpp, cuda and opencl.
class Device {
  public:
  Device() = default;
  /// Constructor with device ID, num of executors (e.g., cuda streams),
  /// max mem size to use (in MB), identifier of scheduler type (default
  /// scheduler run operations synchronously) and virtual memory type (default
  /// vm only provides garbage collection).
  Device(int id, int num_executors, string scheduler, string vm);
  virtual void SetRandSeed(unsigned seed) = 0;

  /// Called by Tensor.
  Blob* NewBlob(int size);

  /// Called by Tensor.
  void FreeBlob(Blob* blob);

  /// Copy data within or across devices.
  void CopyDataToFrom(Blob* dst, Blob* src, size_t nBytes,
                      CopyDirection direction, int dst_offset, int src_offset);

  void CopyDataFromHostPtr(Blob* dst, const void* src, size_t nBytes,
                           size_t dst_offset = 0);
  /// Submit the operation to the device, which may execute it right now or
  /// delay it depending on the scheduler.
  void Exec(function<void(Context*)>&& fn, const vector<Blob*> read_blobs,
                    const vector<Blob*> write_blobs,
                    bool use_rand_generator = false);

  // Wait for one event.
  // void WaitFor();

  /// wait for all operations submitted to this device.
  void Sync();

  /// Return the programming language for this device.
  LangType lang() const {
    return lang_;
  }

  Device* host() const { return host_;}

  Context* context(int k) {
    return &ctx_;
  }

  int id() const { return id_; }

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
  Device* host_;
  // TODO(wangwei) define multiple contexts, one per executor
  Context ctx_;
};

/// Represent a CPU device which may have multiple threads/executors.
/// It runs cpp code.
class CppCPU : public Device {
 public:
  CppCPU(int id = -1, int num_executors = 1,
            string scheduler = "sync", string vm = "gc-only");

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

/// a singleton CppDevice as the host for all devices.
extern CppCPU defaultDevice;


// Implement Device using OpenCL libs.
// class OpenclDevice : public Device { };

#ifdef USE_CUDA
// Represent a Nvidia GPU which runs cuda code.
class CudaGPU : public Device {
 public:
  ~CudaGPU();
  CudaGPU(int id = 0, int num_executors = 1, string scheduler = "sync",
         string vm = "gc-only");

  void SetRandSeed(unsigned seed) override;
  static void DeviceQuery();
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
  /// This function finds the first available device by checking devices with
  /// ordinal from start_id to the highest available value. In the
  /// EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  /// claims the device due to the initialization of the context.
  static int FindDevice(const int start_id);
 protected:
  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) override;

  /// Allocate cpu memory.
  void* Malloc(int size) override;

  /// Free cpu memory.
  void Free(void* ptr) override;
};

/// CudaCPU which uses cudaMallocHost to allocate pinned memory for host.

#endif  // USE_CUDA

// Implement a CudaHost device, which used cuda functions for memory
// malloc/free.
// class CudaHost : public Device {}
//
/// The base type of callback argument structure.
/// The specific arg should inherit from this one.
/*
class CallbackArg {
 public:
  template <typename T>
  T* CastTo() {
    static_assert(std::is_base_of<CallbackArg, T>::value,
                  "The casted type must be a sub-class of CallbackArg");
    return static_cast<T*>(this);
  }
};
/// Type of callback functions for executing tensor ops.
typedef function<void(CallbackArg*)> CallbackFn;
public:
  /// Operation has a function, and read/write blobs.
  typedef struct _Operation {
    function<void(Context*)> fn;
    const vector<Blob*> read_blobs;
    const vector<Blob*> write_blobs;
  } Operation;

*/
}  // namespace singa

#endif  // SINGA_CORE_DEVICE_H_
