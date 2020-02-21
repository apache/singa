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

#ifndef SINGA_UTILS_OPENCL_UTILS_H_
#define SINGA_UTILS_OPENCL_UTILS_H_

#ifdef USE_OPENCL

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#ifndef VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_OPENCL
#endif

#ifndef __APPLE__
#include "CL/cl.h"
#else
#include "OpenCL/cl.h"
#endif

#include <viennacl/backend/opencl.hpp>
#include <viennacl/ocl/backend.hpp>
#include <viennacl/ocl/device.hpp>
#include <viennacl/ocl/device_utils.hpp>
#include <viennacl/ocl/kernel.hpp>
#include <viennacl/ocl/platform.hpp>
#include <viennacl/ocl/program.hpp>
#include <viennacl/ocl/utils.hpp>

inline viennacl::ocl::handle<cl_mem> WrapHandle(cl_mem in,
                                                viennacl::ocl::context &ctx) {
  if (in != nullptr) {
    viennacl::ocl::handle<cl_mem> memhandle(in, ctx);
    memhandle.inc();
    return memhandle;
  } else {
    cl_int err;
    cl_mem dummy =
        clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE, 0, nullptr, &err);
    viennacl::ocl::handle<cl_mem> memhandle(dummy, ctx);
    return memhandle;
  }
}

#endif  // USE_OPENCL

#endif  // SINGA_UTILS_OPENCL_UTILS_H_
