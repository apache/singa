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

#include "singa/core/common.h"
#include "singa/core/memory.h"
#include "singa/core/scheduler.h"
#include "singa/proto/core.pb.h"

using std::vector;
using std::string;
using std::function;
namespace singa {
/// The base type of callback argument structure.
/// The specific arg should inherit from this one.
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

/// Allocate memory and execute Tensor operations.
class Device {
 public:
  /// Operation has a function, and read/write blobs.
  typedef struct _Operation {
    function<void(Context*)> fn;
    const vector<Blob*> read_blobs;
    const vector<Blob*> write_blobs;
  } Operation;

 public:
  Device() = default;
  /// Constructor with device ID, num of executors (e.g., cuda streams),
  /// max mem size to use (in MB), identifier of scheduler type (default
  /// scheduler run operations synchronously) and virtual memory type (default
  /// vm only provides garbage collection).
  Device(int id, int num_executors = 16, string scheduler = "sync",
         string vm = "gc-only");

  /// Called by Tensor.
  Blob* NewBlob(int size);

  /// Called by Tensor.
  void FreeBlob(Blob* blob);

  /// Copy data within or across devices.
  void CopyData(Blob* dst, const Blob& src, int len, int dst_offset,
                int src_offset);

  void CopyDataFromHostPtr(Blob* dst, const void* src, size_t size);
  /// Submit the operation to the device, which may execute it right now or
  /// delay it depending on the scheduler.
  void Exec(function<void(Context*)> fn, const vector<Blob*> read_blobs,
              const vector<Blob*> write_blobs, bool use_rand_generator = false);

  // Wait for one event.
  // void WaitFor();

  /// wait for all operations submitted to this device.
  void Sync();

  LibType device_lib() const { return device_lib_; }
  LibType nn_lib() const { return nn_lib_; }

  Device* host() const { return host_; }

 protected:
  /// Execute one operation on one executor.
  virtual void Exec(int operation, int executor) = 0;

  /// Allocate device memory.
  virtual void* Malloc(int size) = 0;

  /// Free device memory.
  virtual void Free(void* ptr) = 0;

 protected:
  int id_ = 0;
  Scheduler* scheduler_ = nullptr;
  VirtualMemory* vm_ = nullptr;
  /// could be kCudnn
  LibType nn_lib_;
  /// could be kCpp, kCuda, kOpencl
  LibType device_lib_;
  // SafeQueue<Operation> op_queue_;
  // SafeQueue<Operation> op_log_;
  /// The host device
  Device* host_;
};
// Implement Device using Cpp libs.
class CppDevice : public Device {
 public:
  CppDevice(int id, int num_executors);

  void Exec(int operation, int executor) override;

 protected:
  /// Allocate cpu memory.
  void* Malloc(int size) override;

  /// Free cpu memory.
  void Free(void* ptr) override;
};

/// a singleton CppDevice as the host for all devices.
extern CppDevice hostDeviceSingleton;

// Implement Device using OpenCL libs.
// class OpenclDevice : public Device { };

// Implement Device using Cuda libs for Nvidia GPUs.
// class CudaDevice : public Device { };

}  // namespace singa

#endif  // SINGA_CORE_DEVICE_H_
