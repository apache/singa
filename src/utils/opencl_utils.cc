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

#include "singa/utils/opencl_utils.h"

#ifdef USE_OPENCL

void PrintDeviceInfo(const cl::Device &dev) {
  cl_int status = CL_SUCCESS;

  LOG(INFO) << "\tDevice type: " << dev.getInfo<CL_DEVICE_TYPE>(&status);
  LOG(INFO) << "\tUnified memory: " << dev.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>(&status);
  LOG(INFO) << "\tClock speed (MHz): " << dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>(&status);
  LOG(INFO) << "\tECC memory: " << dev.getInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT>(&status);
  LOG(INFO) << "\tLittle endian: " << dev.getInfo<CL_DEVICE_ENDIAN_LITTLE>(&status);
  LOG(INFO) << "\tCompute units: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&status);
  LOG(INFO) << "\tMax work grp size: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&status);
//LOG(INFO) << "\tMax work item size: " << dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>(&status);
  LOG(INFO) << "\tMax item dimension: " << dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(&status);
  LOG(INFO) << "\tQueue properties: " << dev.getInfo<CL_DEVICE_QUEUE_PROPERTIES>(&status);
  LOG(INFO) << "\tExecution capabilities: " << dev.getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>(&status);
  LOG(INFO) << "\tMax mem alloc size: " << dev.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(&status);
  LOG(INFO) << "\tGlobal mem size: " << dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&status);
  LOG(INFO) << "\tLocal mem size: " << dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&status);
  LOG(INFO) << "\n";

  OCL_CHECK(status, "Failed to retrieve device information!");
}


void PrintPlatformInfo(const cl::Platform &p) {
  cl_int status = CL_SUCCESS;

  LOG(INFO) << "\tName: 	 " << p.getInfo<CL_PLATFORM_NAME>(&status);
  LOG(INFO) << "\tProfile: " << p.getInfo<CL_PLATFORM_PROFILE>(&status);
  LOG(INFO) << "\tVersion: " << p.getInfo<CL_PLATFORM_VERSION>(&status);
  LOG(INFO) << "\tVendor:  " << p.getInfo<CL_PLATFORM_VENDOR>(&status);
  LOG(INFO) << "\tExtensions: " << p.getInfo<CL_PLATFORM_EXTENSIONS>(&status);
  LOG(INFO) << "\n";

  OCL_CHECK(status, "Failed to retrieve platform information!");
}


#endif // USE_OPENCL
