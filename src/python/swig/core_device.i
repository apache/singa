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
%include "std_shared_ptr.i"

%{
#include "singa/core/device.h"
%}

/* smart pointer to avoid memory leak */
%shared_ptr(singa::Device);
%shared_ptr(singa::CppCPU);
%shared_ptr(singa::CudaGPU);

namespace singa{

  class Device {
   public:
    virtual void SetRandSeed(unsigned seed) = 0;
    std::shared_ptr<Device> host();
    int id() const;
  };

  class CppCPU : public Device {
   public:
    CppCPU();
    void SetRandSeed(unsigned seed) override;
    /* (TODO) add necessary functions of CppCPU class
    */
  };

  class CudaGPU : public Device {
   public:
    CudaGPU();
    void SetRandSeed(unsigned seed) override;
    /* (TODO) add necessary functions of CudaGPU class
    */
  };
}

