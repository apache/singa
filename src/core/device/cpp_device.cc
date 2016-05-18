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
#include "singa/core/device.h"
namespace singa {
CppDevice hostDeviceSingleton(-1, 1);
CppDevice::CppDevice(int id, int num_executors, string scheduler,
         string vm) : Device(id, num_executors, scheduler, vm) {
  device_type_ = kCpp;
  host_ = nullptr;
}

void CppDevice::SetRandSeed(unsigned seed) {
  ctx_.random_generator.seed(seed);
}
void CppDevice::DoExec(function<void(Context*)>&& fn, int executor) {
  CHECK_EQ(executor, 0);
  fn(&ctx_);
}

void* CppDevice::Malloc(int size) {
  return malloc(size);
}

void CppDevice::Free(void* ptr) {
  free(ptr);
}

void CppDevice::CopyToFrom(void* dst, const void* src, size_t nBytes,
                           CopyDirection direction, Context* ctx) {
  memcpy(dst, src, nBytes);
}
}  // namespace singa
