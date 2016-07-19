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

#include "singa/core/opencl_device.h"
#include "singa/utils/tinydir.h"

#ifdef USE_OPENCL

using std::string;

namespace singa {

const string OpenclDevice::cl_src_path = "../src/core/tensor";

OpenclDevice::OpenclDevice(int id, int num_executors) 
	: Device(id, num_executors) {
  lang_ = kOpencl;
  this->kernels = std::make_shared<std::map<std::string, cl::Kernel>>();
  
  // Create the OpenCL Device, Context, and CommandQueue.
  /// TODO: This merely chooses the first device on the first platform.
  cl_int status = CL_SUCCESS;

  std::vector<cl::Platform> platforms;
  status = cl::Platform::get(&platforms);
  OCL_CHECK(status, "Failed to find any OpenCL platforms!");
  
  std::vector<cl::Device> devices;
  status = platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
  OCL_CHECK(status, "Failed to get list of devices from platform!");

  this->this_device = cl::Device(devices[0]);
  this->ocl_ctx = cl::Context(this_device, nullptr, nullptr, nullptr, &status);
  OCL_CHECK(status, "Failed to create context!");

  this->cmdq = cl::CommandQueue(ocl_ctx, this_device, CL_QUEUE_PROFILING_ENABLE, &status);
  OCL_CHECK(status, "Failed to create a command queue!");

  BuildPrograms();
  
  ctx_.kernels = kernels;
  ctx_.ocl_cmdq = cmdq;
  ctx_.ocl_ctx = ocl_ctx;
}


OpenclDevice::~OpenclDevice() {
  
  // Flush and finish the command queue.
  cmdq.flush();
  cmdq.finish();
}


cl::Kernel OpenclDevice::GetKernel(const std::string& kname, cl_int* status) {
  if (!status) *status = CL_SUCCESS;
  if (kernels->find(kname) == kernels->end()) {
    // TODO: Not found
    LOG(ERROR) << "Error: Kernel " << kname << " could not be found!";
    if (!status) *status = CL_INVALID_KERNEL;
  }
  return kernels->at(kname);
}

/*
void OpenclDevice::PrintAllDeviceInfo() {
  cl_int status = CL_SUCCESS;

  for (auto dev : devices) {
    PrintDeviceInfo(d);
  }
}
*/


void OpenclDevice::PrintClBuildInfo(cl::Program &p) {
  cl_int status = CL_SUCCESS;

  auto buildStatus = p.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(&status);
  for (auto pair : buildStatus)
	std::cout << clGetBuildInfoString(pair.second) << std::endl;

  auto buildLog = p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&status);
  for (auto pair : buildLog)
	std::cout << pair.second << std::endl;
}


void OpenclDevice::SetRandSeed(unsigned seed) { seed = seed; }


void OpenclDevice::CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                                  CopyDirection direction, int dst_offset, int src_offset) {
  // Pointers must be valid.
  if (!dst || !src) return;

  CopyToFrom(dst->mutable_data(), src->data(), nBytes, direction);
}

/*
void OpenclDevice::CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes, size_t dst_offset) {
  CopyToFrom(dst->mutable_data(), src, 4, kHostToDevice);
}
*/

void OpenclDevice::BuildPrograms(const std::string &kdir) {
  cl_int status = CL_SUCCESS;

  tinydir_dir dir;
  tinydir_open(&dir, kdir.c_str());

  while (dir.has_next) {
	tinydir_file file;
	tinydir_readfile(&dir, &file);
	std::string ext(file.extension);
	if (ext.compare("cl") != 0) {
	  tinydir_next(&dir);
	  continue;
	}

	std::ifstream clFile(file.path, std::ios_base::binary);
	std::stringstream buffer;
	buffer << clFile.rdbuf();
	std::string clSrc(buffer.str());

	cl::Program program(this->ocl_ctx, clSrc, false, &status);
	OCL_CHECK(status, "Program creation failed.");
	status = program.build({this_device}, "-cl-std=CL1.2");
	if (status == CL_SUCCESS) {
	  std::vector<cl::Kernel> built_kernels;
	  status = program.createKernels(&built_kernels);
	  OCL_CHECK(status, "Failed to create kernels in built program.");
	  
	  for (auto k : built_kernels) {
		std::string name = k.getInfo<CL_KERNEL_FUNCTION_NAME>(&status);
		this->kernels->insert(std::make_pair(name, k));
	  }
	} else {
	  OCL_CHECK(status, "Build failed on source path");
	  LOG(ERROR) << file.path << std::endl;
	  PrintClBuildInfo(program);
	}

	tinydir_next(&dir);
  }
}

// Device IO functions.
// TODO:
// Research - MapBuffers can improve performance when the device uses shared memory
// but is more complex to understand. http://stackoverflow.com/questions/22057692/whats-the-difference-between-clenqueuemapbuffer-and-clenqueuewritebuffer
// Intel graphics (and possibly AMD APUs) should use MapBuffers?
// https://software.intel.com/en-us/articles/getting-the-most-from-opencl-12-how-to-increase-performance-by-minimizing-buffer-copies-on-intel-processor-graphics


void OpenclDevice::DoExec(function<void(Context*)>&& fn, int executor) {
  fn(&ctx_);
}

// NOTE: ASSUMES dst AND/OR src POINTERS CAN BE CAST TO cl::Buffer POINTERS!
void OpenclDevice::CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) {
  // Pointers must be valid.
  if (!dst || !src) return;

  switch(direction) {
  case kHostToDevice: {
    WriteToDevice(static_cast<cl::Buffer*>(dst), src, nBytes);
    return;
  }
  case kDeviceToHost: {
    ReadFromDevice(dst, static_cast<const cl::Buffer*>(src), nBytes);
    return;
  }
  case kDeviceToDevice: {
    CopyDeviceBuffer(static_cast<cl::Buffer*>(dst), static_cast<const cl::Buffer*>(src), nBytes);
    return;
  }
  default:
    return;
  }
}


void* OpenclDevice::Malloc(int size) {
  cl_int status = CL_SUCCESS;

  cl::Buffer* buffer = new cl::Buffer(ocl_ctx, CL_MEM_READ_WRITE, size, nullptr, &status);
  OCL_CHECK(status, "Unable to allocate memory in OpenCL device.");

  return static_cast<void*>(buffer);
}


void OpenclDevice::Free(void* p) {
  if (!p) return;
  cl::Buffer* buffer = static_cast<cl::Buffer*>(p);
  delete buffer;
}


void OpenclDevice::WriteToDevice(cl::Buffer* dst, const void* src, const size_t size) {
  cl_int status = CL_SUCCESS;
  
  status = cmdq.enqueueWriteBuffer(*dst, CL_TRUE, 0, size, src);
  OCL_CHECK(status, "Unable to write data to OpenCL device.");
}


void OpenclDevice::ReadFromDevice(void* dst, const cl::Buffer* src, const size_t size) {
  cl_int status = CL_SUCCESS;
  
  status = cmdq.enqueueReadBuffer(*src, CL_TRUE, 0, size, dst);
  OCL_CHECK(status, "Unable to read data from OpenCL device.");
}


// dst: cl::Buffer pointer    src: cl::Buffer pointer
void OpenclDevice::CopyDeviceBuffer(cl::Buffer* dst, const cl::Buffer* src, const size_t size) {
  cl_int status = CL_SUCCESS;

  status = cmdq.enqueueCopyBuffer(*src, *dst, 0, 0, size);
  OCL_CHECK(status, "Unable to copy buffer in OpenCL device.");
}

} // namespace singa

#endif // USE_OPENCL
