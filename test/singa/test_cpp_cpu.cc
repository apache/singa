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

using singa::Block;
using singa::CppCPU;
TEST(CppCPU, Constructor) {
  CppCPU dev;
  EXPECT_EQ(-1, dev.id());
}

TEST(CppCPU, MemoryMallocFree) {
  CppCPU dev;
  Block* b = dev.NewBlock(4);
  EXPECT_NE(nullptr, b);
  EXPECT_EQ(4u, b->size());
  dev.FreeBlock(b);
}

TEST(CppCPU, Exec) {
  CppCPU dev;
  Block* b = dev.NewBlock(4);
  int x = 1, y = 3, z = 0;
  dev.Exec([x, y, &z](singa::Context* ctx) { z = x + y; }, {b}, {b});
  EXPECT_EQ(x + y, z);
  dev.FreeBlock(b);
}

TEST(CppCPU, CopyData) {
  CppCPU dev;
  Block* b = dev.NewBlock(4);
  char s[] = {'a', 'b', 'c', 'x'};
  dev.CopyDataFromHostPtr(b, s, 4);
  const char* bstr = static_cast<const char*>(b->data());
  EXPECT_EQ('a', bstr[0]);
  EXPECT_EQ('b', bstr[1]);
  EXPECT_EQ('x', bstr[3]);

  Block* c = dev.NewBlock(4);
  dev.CopyDataToFrom(c, b, 4, singa::kHostToHost, 0, 0);
  const char* cstr = static_cast<const char*>(c->data());

  EXPECT_EQ('a', cstr[0]);
  EXPECT_EQ('b', cstr[1]);
  EXPECT_EQ('x', cstr[3]);
  dev.FreeBlock(b);
  dev.FreeBlock(c);
}
