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

#include "gtest/gtest.h"
#include "singa/core/device.h"
#include "singa/core/tensor.h"
using namespace std;
#ifdef USE_CUDA
using singa::Platform;

TEST(Platform, CreateMultDevice) {
  int n = Platform::GetNumGPUs();
  auto devs = Platform::CreateCudaGPUs(n);
  for (size_t i = 0; i < devs.size(); i++) {
    auto b = devs[i]->NewBlock(512 + 512 * (2 - i));
    // for lazy allocation
    b->mutable_data();
    EXPECT_EQ(512 + 512 * (2 - i), devs[i]->GetAllocatedMem());
    devs[i]->FreeBlock(b);
  }
}

TEST(Platform, NumGPUs) {
  int n = Platform::GetNumGPUs();
  EXPECT_GE(n, 0);
  EXPECT_LE(n, 32);
}

TEST(Platform, QueryMem) {
  size_t n = Platform::GetNumGPUs();
  auto ids = Platform::GetGPUIDs();
  EXPECT_EQ(ids.size(), n);
  auto mem = Platform::GetGPUMemSize();
  for (auto x : mem) EXPECT_GT(x.second, x.first);
}

TEST(Platform, CreateDevice) {
  auto dev = Platform::CreateCudaGPUs(1).at(0);
  size_t size[] = {128, 256, 3, 24};
  {
    auto ptr = dev->NewBlock(size[0]);
    // for lazy allocation
    ptr->mutable_data();
    auto allocated = dev->GetAllocatedMem();
    EXPECT_LE(size[0], allocated);
    dev->FreeBlock(ptr);
    allocated = dev->GetAllocatedMem();
  }
  {
    auto ptr0 = dev->NewBlock(size[0]);
    auto ptr1 = dev->NewBlock(size[1]);
    auto ptr2 = dev->NewBlock(size[2]);
    ptr0->mutable_data();
    ptr1->mutable_data();
    ptr2->mutable_data();
    auto allocated = dev->GetAllocatedMem();
    EXPECT_LE(size[0] + size[1] + size[2], allocated);
    auto ptr3 = dev->NewBlock(size[3]);
    ptr3->mutable_data();
    allocated = dev->GetAllocatedMem();
    EXPECT_LE(size[0] + size[1] + size[2] + size[3], allocated);
    dev->FreeBlock(ptr0);
    dev->FreeBlock(ptr1);
    dev->FreeBlock(ptr2);
    //    allocated = dev->GetAllocatedMem();
    //    EXPECT_EQ(size[3], allocated);
    dev->FreeBlock(ptr3);
    //    allocated = dev->GetAllocatedMem();
    //    EXPECT_EQ(0, allocated);
  }
}

TEST(Platform, CreatTensor) {
  auto cuda = Platform::CreateCudaGPUs(1)[0];
  singa::Tensor t(singa::Shape{2, 3, 4}, cuda);
  t.SetValue(2.1f);
  t.ToHost();
  auto tPtr = t.data<float>();
  for (size_t i = 0; i < t.Size(); i++) EXPECT_FLOAT_EQ(tPtr[i], 2.1f);
  t.ToDevice(cuda);
  t = t * 3.0f;
  t.ToHost();
  tPtr = t.data<float>();
  for (size_t i = 0; i < t.Size(); i++) EXPECT_FLOAT_EQ(tPtr[i], 2.1f * 3.0f);
}
#endif
