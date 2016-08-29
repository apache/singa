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
#include "singa/utils/tinydir.h"
#include "singa/utils/opencl_utils.h"

#ifdef USE_OPENCL

using namespace viennacl;
using namespace viennacl::backend::opencl;

namespace singa {

const std::string OpenclDevice::cl_src_path = "../src/core/tensor";

OpenclDevice::OpenclDevice(int id, int num_executors)
	: Device(id, num_executors) {
  CHECK_GE(id, 0);
  lang_ = kOpencl;
  
  ocl::current_context().build_options("-cl-std=CL1.2");
  
  ctx_.vcl_ctx = ocl::current_context();
  ctx_.vcl_ocl_ctx_id = 0;
  this->this_device = ocl::current_device();
  
  BuildPrograms(cl_src_path);
}


OpenclDevice::~OpenclDevice() {

  // Flush and finish the command queue.
  auto cmdq = ocl::current_context().get_queue();
  
  cmdq.flush();
  cmdq.finish();
}


void OpenclDevice::SetRandSeed(unsigned seed) { seed = seed; }


void OpenclDevice::CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                                  CopyDirection direction, int dst_offset, int src_offset) {
  // Pointers must be valid.
  if (!dst || !src) return;

  switch(direction) {
  case kHostToDevice: {
    auto dst_handle = WrapHandle(static_cast<cl_mem>(dst->mutable_data()), &ctx_.vcl_ctx);
    memory_write(dst_handle, dst_offset, nBytes, src->data());
    return;
  }
  case kDeviceToHost: {
    auto src_handle = WrapHandle(static_cast<cl_mem>(src->mutable_data()), &ctx_.vcl_ctx);
    memory_read(src_handle, src_offset, nBytes, dst->mutable_data());
    return;
  }
  case kDeviceToDevice: {
    auto src_handle = WrapHandle(static_cast<cl_mem>(src->mutable_data()), &ctx_.vcl_ctx);
    auto dst_handle = WrapHandle(static_cast<cl_mem>(dst->mutable_data()), &ctx_.vcl_ctx);
    memory_copy(src_handle, dst_handle, src_offset, dst_offset, nBytes);
    return;
  }
  default:
    return;
  }
}


void OpenclDevice::BuildPrograms(const std::string &kdir) {
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

	std::string name(file.name);
    ocl::current_context().add_program(clSrc, name);

	tinydir_next(&dir);
  }
}


void OpenclDevice::DoExec(function<void(Context*)>&& fn, int executor) {
  fn(&ctx_);
}


void OpenclDevice::CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) {
  // Pointers must be valid.
  if (!dst || !src) return;

  switch(direction) {
  case kHostToDevice: {
    auto dst_handle = WrapHandle(static_cast<cl_mem>(dst), &ctx_.vcl_ctx);
    memory_write(dst_handle, 0, nBytes, src);
    return;
  }
  case kDeviceToHost: {
    auto src_handle = WrapHandle((const cl_mem)src, &ctx_.vcl_ctx);
    memory_read(src_handle, 0, nBytes, dst);
    return;
  }
  case kDeviceToDevice: {
    auto src_handle = WrapHandle((const cl_mem)src, &ctx_.vcl_ctx);
    auto dst_handle = WrapHandle(static_cast<cl_mem>(dst), &ctx_.vcl_ctx);
    memory_copy(src_handle, dst_handle, 0, 0, nBytes);
    return;
  }
  default:
    return;
  }
}


void* OpenclDevice::Malloc(int size) {
  cl_mem buffer = memory_create(ocl::current_context(), size, nullptr);

  return static_cast<void*>(buffer);
}


void OpenclDevice::Free(void* p) {
  if (!p) return;
  cl_mem buffer = static_cast<cl_mem>(p);
  clReleaseMemObject(buffer);
}

} // namespace singa

#endif // USE_OPENCL
