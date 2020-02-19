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

#include "gtest/gtest.h"
#include "singa/core/device.h"
#include "singa/proto/core.pb.h"

#ifdef USE_OPENCL

using singa::Block;
using singa::CppCPU;
using singa::OpenclDevice;

TEST(OpenclDevice, Constructor) {
  OpenclDevice dev;
  EXPECT_EQ(0, dev.id());
}

TEST(OpenclDevice, MemoryMallocFree) {
  OpenclDevice dev;
  Block* b = dev.NewBlock(4);
  EXPECT_NE(nullptr, b);
  EXPECT_EQ(4u, b->size());
  dev.FreeBlock(b);
}

TEST(OpenclDevice, Exec) {
  OpenclDevice dev;
  Block* b = dev.NewBlock(4);
  int x = 1, y = 3, z = 0;
  dev.Exec([x, y, &z](singa::Context* ctx) { z = x + y; }, {b}, {b}, false);
  EXPECT_EQ(x + y, z);
  dev.FreeBlock(b);
}

// Tests for integrity of one round of data transfer to an OpenCL device and
// back.
TEST(OpenclDevice, CopyDataToFrom) {
  OpenclDevice dev;
  CppCPU host;

  Block* a = host.NewBlock(4);
  Block* b = dev.NewBlock(4);
  Block* c = host.NewBlock(4);

  // Allocate the Block object on the host.
  char s[] = {'a', 'b', 'c', 'x'};
  host.CopyDataFromHostPtr(a, s, 4);

  // Copy back and forth.
  dev.CopyDataToFrom(b, a, 4, singa::kHostToDevice);
  dev.CopyDataToFrom(c, b, 4, singa::kDeviceToHost);

  const char* astr = static_cast<const char*>(c->data());
  EXPECT_EQ('a', astr[0]);
  EXPECT_EQ('b', astr[1]);
  EXPECT_EQ('c', astr[2]);
  EXPECT_EQ('x', astr[3]);
}

TEST(OpenclDevice, DuplicateDataOnDevice) {
  OpenclDevice dev;
  CppCPU host;

  Block* a = host.NewBlock(4);
  Block* b = dev.NewBlock(4);
  Block* c = dev.NewBlock(4);
  Block* d = host.NewBlock(4);

  // Allocate the Block object on the host.
  char s[] = {'a', 'b', 'c', 'x'};
  host.CopyDataFromHostPtr(a, s, 4);

  // Copy to device and duplicate.
  dev.CopyDataToFrom(b, a, 4, singa::kHostToDevice);
  dev.CopyDataToFrom(c, b, 4, singa::kDeviceToDevice);
  dev.CopyDataToFrom(d, c, 4, singa::kDeviceToHost);

  const char* astr = static_cast<const char*>(d->data());
  EXPECT_EQ('a', astr[0]);
  EXPECT_EQ('b', astr[1]);
  EXPECT_EQ('c', astr[2]);
  EXPECT_EQ('x', astr[3]);
}

#endif  // USE_OPENCL
