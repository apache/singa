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

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "singa/core/device.h"

namespace singa {

#ifdef USE_OPENCL

OpenclDevice::OpenclDevice() {
  Init();
  BuildPrograms();
}


OpenclDevice::~OpenclDevice() {
  
  // Flush and finish the command queue.
  cmdQueue.flush();
  cmdQueue.finish();
}


void OpenclDevice::Init() {
  cl_int status = CL_SUCCESS;

  // Get a list of all OpenCL platforms.
  status = cl::Platform::get(&platforms);
  CheckError(status, "Failed to find any OpenCL platforms!");

  PrintPlatformInformation();

  // Filter for the first OpenCL 2.x platform and set it as the default.
  // If not found, then just set the first platform as the default.
  // TODO: Filter out NVidia devices here for OpenCL 2.0 compatibility?
  cl::Platform defaultPlat;
  bool defaultSet = false;
  for (auto p : platforms) {
    std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
    if (platver.find("OpenCL 2.") != std::string::npos) {
      status = cl::Platform::setDefault(p);
      defaultSet = true;
      break;
    }
  }
  if (!defaultSet) status = cl::Platform::setDefault(platforms[0]);
  CheckError(status, "Failed to set a default platform!");

  // Get a list of all available OpenCL devices.
  for (auto p : platforms) {
    std::vector<Device> pDev;
    status = p.getDevices(CL_DEVICE_TYPE_ALL, &pDev);
    CheckError(status, "Failed to retrieve devices from platform!");
    for (auto d : pDev) {
      devices.push_back(d);
    }
  }

  // Put all devices into a context.
  context = cl::Context(devices, CL_CONTEXT_PLATFORM, nullptr, nullptr, &status);
  CheckError(status, "Failed to create an OpenCL context!");

  // Create the command queue.
  cmdQueue = cl::CommandQueue(context, CL_QUEUE_PROFILING_ENABLE, &status);
  CheckError(status, "Failed to create an OpenCL command queue!");
}


void OpenclDevice::PrintAllPlatformInfo() {
  cl_int status = CL_SUCCESS;

  LOG(INFO) << platforms.size() << " platforms found." << std::endl;

  for (auto p : platforms) {
    LOG(INFO) << "\tName: 	 " << p.getInfo<CL_PLATFORM_NAME>(&status);
    LOG(INFO) << "\tProfile: " << p.getInfo<CL_PLATFORM_PROFILE>(&status);
    LOG(INFO) << "\tVersion: " << p.getInfo<CL_PLATFORM_VERSION>(&status);
    LOG(INFO) << "\tVendor:  " << p.getInfo<CL_PLATFORM_VENDOR>(&status);
    LOG(INFO) << "\tExtensions: " << p.getInfo<CL_PLATFORM_EXTENSIONS>(&status);
    LOG(INFO) << "\n";

    CheckError(status, "Failed to retrieve platform information!");
  }
}


void OpenclDevice::PrintAllDeviceInfo() {
  cl_int status = CL_SUCCESS;

  for (auto dev : devices) {
    PrintDeviceInfo(d);
  }
}


void OpenclDevice::PrintDeviceInfo(const cl::Device &dev) {
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

  CheckError(status, "Failed to retrieve device information!");
}


void PrintClBuildInfo(const cl::Program &p) {
  cl_int status = CL_SUCCESS;

  auto buildLog = p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&status);
  for (auto pair : buildInfo) {
	LOG(ERROR) << pair.second << std::endl;
  }
}


cl::Kernel OpenclDevice::GetKernel(const std::string& kname) {
  if (kernels.find(kname) == kernels.end()) {
    // TODO: Not found
    LOG(ERROR) << "Error: Kernel " << kname << " could not be found!";
  }
  return kernels[kname];
}


void OpenclDevice::BuildPrograms(const std::string &kdir) {
  cl_int status = CL_SUCCESS;

  tinydir_dir dir;
  tinydir_open_sorted(&dir, kdir.c_str());
  for (size_t i = 0; i < dir.n_files; i++) {
	tinydir_file file;
	tinydir_readfile_n(&dir, &file, i);
	string fileext(file.extension);
	if (fileext.compare("cl") != 0) continue;
	
	std::ifstream clFile(file.path, std::ios_base::binary);
	std::stringstream bufer;
	buffer << clFile.rdbuf();
	std::string clSrc(buffer.str());

	cl::Program program(clSrc, false, &status);
	if (status == CL_SUCCESS) {
	  // TODO: Handle success
	} else {
	  PrintClBuildInfo(program);
	}
  }

// Using Boost::Filesystem
/*
  if (!bfs::is_directory(kpath)) return;
  for (auto entry : boost::make_iterator_range(bfs::recursive_directory_iterator(kpath), {})) {
	if (bfs::is_directory(entry) && recursive) BuildPrograms(entry);
	if (entry.path().extension() != ".cl") continue;

	bfs::ifstream clFile(entry.path(), std::ios_base::binary);
	std::stringstream buffer;
	buffer << clFile.rdbuf();
	std::string clSrc(buffer.str());

	cl::Program program(clSrc, false, &status);
	if (status == CL_SUCCESS) {
	  // TODO: Handle success
	} else {
	  PrintClBuildInfo(program);
	}
  }*/
}


static bool OpenclDevice::CheckError(const cl_int status, const std::string& what) {
  if (status == CL_SUCCESS) return true; // Nothing wrong.
  LOG(ERROR) << "CL ERROR " << cl_int << ": " << what << std::endl;
  return false;
}

#endif // USE_OPENCL

} // namespace singa
