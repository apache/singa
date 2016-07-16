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

namespace singa {

#ifdef USE_OPENCL

OpenclDevice(int id = 0, int num_executors = 1, string scheduler = "sync", string vm = "gc-only") 
	: Device(id, num_executors, scheduler, vm) {
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
  std::vector<cl::Platform> platforms;
  status = cl::Platform::get(&platforms);
  OCL_CHECK(status, "Failed to find any OpenCL platforms!");

  PrintPlatformInformation();

  // Filter for the first OpenCL 2.x platform and set it as the default.
  // If not found, then just set the first platform as the default.
  cl::Platform defaultPlat;
  bool defaultSet = false;
  for (auto p : platforms) {
    std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
    if (platver.find("OpenCL 2.") != std::string::npos) {
      status = cl::Platform::setDefault(p);
	  defaultPlat = p;
      defaultSet = true;
      break;
    }
  }

  // If no OpenCL 2.x platform found, then just set any available platform as the default.
  if (!defaultSet) {
	LOG(WARNING) << "No OpenCL 2.x platform found. Falling back to an OpenCL 1.x platform!" << std::endl;
	status = cl::Platform::setDefault(platforms[0]);
	defaultPlat = platforms[0];
  }
  OCL_CHECK(status, "Failed to set a default platform!");

  // Get a list of all available OpenCL devices, and choose one from them.
  // TODO: Enable choice of particular device. Now it only chooses the first one by default.
  std::vector<Device> pDev;
  status = defaultPlat.getDevices(CL_DEVICE_TYPE_ALL, &pDev);
  OCL_CHECK(status, "Failed to retrieve devices from platform!");
  this_device = pDev[0];
  cl::Device::setDefault(pDev[0]);
  

  // Put device into a context.
  auto context = cl::Context(this_device, CL_CONTEXT_PLATFORM, nullptr, nullptr, &status);
  OCL_CHECK(status, "Failed to create an OpenCL context!");
  cl::Context::setDefault(context);

  // Create the command queue.
  cmdQueue = cl::CommandQueue(context, CL_QUEUE_PROFILING_ENABLE, &status);
  OCL_CHECK(status, "Failed to create an OpenCL command queue!");
  cl::CommandQueue::setDefault(cmdQueue);
}


cl::Kernel OpenclDevice::GetKernel(const std::string& kname) {
  if (kernels.find(kname) == kernels.end()) {
    // TODO: Not found
    LOG(ERROR) << "Error: Kernel " << kname << " could not be found!";
  }
  return kernels[kname];
}


void OpenclDevice::PrintAllDeviceInfo() {
  cl_int status = CL_SUCCESS;

  for (auto dev : devices) {
    PrintDeviceInfo(d);
  }
}


void OpenclDevice::PrintClBuildInfo(const cl::Program &p) {
  cl_int status = CL_SUCCESS;

  auto buildLog = p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&status);
  for (auto pair : buildInfo) {
	LOG(ERROR) << pair.second << std::endl;
  }
}


void OpenclDevice::SetRandSeed(unsigned seed) { seed = seed; }


void OpenclDevice::BuildPrograms(const std::string &kdir) {
  cl_int status = CL_SUCCESS;

  tinydir_dir dir;
  tinydir_open_sorted(&dir, kdir.c_str());
  for (size_t i = 0; i < dir.n_files; i++) {
	tinydir_file file;
	tinydir_readfile_n(&dir, &file, i);
	std::string fileext(file.extension);
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
}

// Device IO functions.
// TODO:
// Research - MapBuffers can improve performance when the device uses shared memory
// but is more complex to understand. http://stackoverflow.com/questions/22057692/whats-the-difference-between-clenqueuemapbuffer-and-clenqueuewritebuffer
// Intel graphics (and possibly AMD APUs) should use MapBuffers?
// https://software.intel.com/en-us/articles/getting-the-most-from-opencl-12-how-to-increase-performance-by-minimizing-buffer-copies-on-intel-processor-graphics


void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) {
  dst = WriteToDevice(src, nBytes, ctx);
}


// Create an empty byte array with the specified size. Write it to the device as an OpenCL Buffer.
// Return a pointer to the OpenCL Buffer.
void* OpenclDevice::Malloc(int size, Context ctx) {
  cl_int status = CL_SUCCESS;

  // Create the buffer.
  unsigned char* ptr = new unsigned char[size];
  cl::Buffer* buffer = new cl::Buffer(ctx, CL_MEM_READ_WRITE, size, ptr, &status);
  OCL_CHECK(status, "Unable to allocate memory in OpenCL device.");

  status = ctx.get_dev_cmdq(0).enqueueWriteBuffer(buffer, CL_TRUE, 0, size, ptr);
  OCL_CHECK(status, "Unable to write buffer to OpenCL device.");

  return (void*)buffer;
}


// Cast the void pointer into an OpenCL Buffer object.
// Delete the object and let the destructor take care of the resource release.
void OpenclDevice::Free(void* p) {
  if (!p) return;
  cl::Buffer buffer = *(static_cast<cl::Buffer*>(p));
  delete buffer;
}


// Create a Buffer object allocated with a pointer.
// Write the data pointer into the Buffer.
// With the CommandQueue, push the Buffer into the Device.
// Return the pointer to the Buffer object.
void* OpenclDevice::WriteToDevice(void* data_ptr, const size_t size, Context ctx) {
  if (!data_ptr) return;
  cl_int status = CL_SUCCESS;

  // Create buffer.
  cl::Buffer* buffer = new cl::Buffer(ctx, CL_MEM_READ_WRITE, size, data_ptr, &status);
  OCL_CHECK(status, "Unable to allocate memory in OpenCL device.");

  // Write to device.
  status = ctx.get_dev_cmdq(0).enqueueWriteBuffer(buffer, CL_TRUE, 0, size, data_ptr);
  OCL_CHECK(status, "Unable to write buffer to OpenCL device.");

  return (void*)buffer;
}


// Cast the mutable data in the Block into a OpenCL Buffer pointer.
// Retrieve the data in the Buffer into a byte array.
// Return the pointer to the byte array.
void* OpenclDevice::ReadFromDevice(const size_t size, Block* blk, Context ctx) {
  if (!blk) return;
  cl_int status = CL_STATUS;

  cl::Buffer buffer = *(static_cast<cl::Buffer*>(blk->mutable_data()));
  unsigned char* ptr = new unsigned char[size];
  status = ctx.get_dev_cmdq(0).enqueueReadBuffer(buffer, CL_TRUE, 0, size, ptr);
  return (void*)ptr;
}

#endif // USE_OPENCL

} // namespace singa
