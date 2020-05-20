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

/*interface file for swig */

%module core_device
%include "std_vector.i"
%include "std_string.i"
%include "std_pair.i"
%include "std_shared_ptr.i"

%{
#include "singa/core/device.h"
%}

/* smart pointer to avoid memory leak */
%shared_ptr(singa::Device);

namespace std{
%template(PairSizeT) std::pair<size_t, size_t>;
%template(VecPairSizeT) std::vector<std::pair<size_t, size_t>>;
%template(VecSharedPtrDevice) std::vector<std::shared_ptr<singa::Device>>;
%template(VecInt) std::vector<int>;
}

namespace singa{

class Device {
 public:
  virtual void SetRandSeed(unsigned seed) = 0;
  std::shared_ptr<Device> host();
  int id() const;
  virtual void Sync();
  void ResetGraph();
  void RunGraph(bool serial = false);
  bool graph_enabled() const;
  void EnableGraph(bool enable);
  void PrintTimeProfiling();
  void SetVerbosity(int verbosity);
  static void EnableLazyAlloc(bool enbale);
};

class Platform {
 public:
#if USE_CUDA
  static int GetNumGPUs();
  static const std::vector<int> GetGPUIDs();
  static const std::pair<size_t, size_t> GetGPUMemSize(const int device);
  static const std::vector<std::pair<size_t, size_t>> GetGPUMemSize();
  static const std::string DeviceQuery(int id, bool verbose = false);
  static const std::vector<std::shared_ptr<Device>>
  CreateCudaGPUs(const size_t num_devices, size_t init_size = 0);
  static const std::vector<std::shared_ptr<Device>>
  CreateCudaGPUsOn(const std::vector<int> &devices, size_t init_size = 0);
#endif // USE_CUDA

#if USE_OPENCL

  const int GetNumOpenclPlatforms();
  const int GetNumOpenclDevices();
  static const std::shared_ptr<Device> GetDefaultOpenclDevice();
//  static const std::vector<std::shared_ptr<Device>>
//  CreateOpenclDevices(const size_t num_devices);
//  static const std::vector<std::shared_ptr<Device>>
//  CreateOpenclDevices();

#endif // USE_OPENCL

  static std::shared_ptr<Device> GetDefaultDevice();
};

}
