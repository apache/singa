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
#include "singa/core/opencl_device.h"
#include "singa/core/tensor.h"
#include "singa/proto/core.pb.h"

using singa::OpenclDevice;
using singa::CppCPU;
using singa::Block;
using singa::Shape;
using singa::Tensor;

// Makes a float array and fills it with increasing values from 0.
float* MakeMatrix(const int size) {
  float* mat = new float[size];
  for (int i = 0; i < size; i++)
    mat[i] = i;
  return mat;
}


TEST(OpenclDevice, Constructor) {
  OpenclDevice dev;
  EXPECT_EQ(0, dev.id());
}

TEST(OpenclDevice, MemoryAllocFree) {
  OpenclDevice dev;
  Block* b = dev.NewBlock(4);
  EXPECT_NE(nullptr, b);
  EXPECT_EQ(4u, b->size());
  dev.FreeBlock(b);
}

// Tests for integrity of one round of data transfer to an OpenCL device and back.
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
  EXPECT_EQ('x', astr[3]);
}

// Tensor tests

TEST(OpenCL_TensorMath, TensorMath_CopyDataToDevice) {
  auto ocl_dev = std::make_shared<OpenclDevice>(OpenclDevice());

  Tensor t(Shape{1, 4}, ocl_dev);
  float a[] = {0.0f, 1.0f, 2.0f, 3.0f};
  t.CopyDataFromHostPtr(a, 4);
  
  CppCPU host;
  Block* host_out = host.NewBlock(sizeof(float) * 4);
  ocl_dev->CopyDataToFrom(host_out, t.block(), sizeof(float) * 4, singa::kDeviceToHost);
  
  float* out = static_cast<float*>(host_out->mutable_data());
  EXPECT_EQ(1.0f, out[1]);
  EXPECT_EQ(3.0f, out[3]);
}

TEST(OpenCL_TensorMath, TensorMath_Abs) {
  auto ocl_dev = std::make_shared<OpenclDevice>(OpenclDevice());

  Tensor in(Shape{1, 4}, ocl_dev);
  float a[] = {0.0f, -1.0f, -2.0f, -3.0f};
  in.CopyDataFromHostPtr(a, 4);
  
  in = Abs(in);
  
  CppCPU host;
  Block* host_out = host.NewBlock(sizeof(float) * 4);
  ocl_dev->CopyDataToFrom(host_out, in.block(), sizeof(float) * 4, singa::kDeviceToHost);
  
  float* out = static_cast<float*>(host_out->mutable_data());
  EXPECT_EQ(0.0f, out[0]);
  EXPECT_EQ(1.0f, out[1]);
  EXPECT_EQ(2.0f, out[2]);
  EXPECT_EQ(3.0f, out[3]);
}

TEST(OpenCL_TensorMath, TensorMath_ScalarAdd) {
  auto ocl_dev = std::make_shared<OpenclDevice>(OpenclDevice());

  Tensor in(Shape{1, 4}, ocl_dev);
  float a[] = {0.0f, 1.0f, 2.0f, 3.0f};
  in.CopyDataFromHostPtr(a, 4);
  
  in += 1.0f;
  
  CppCPU host;
  Block* host_out = host.NewBlock(sizeof(float) * 4);
  ocl_dev->CopyDataToFrom(host_out, in.block(), sizeof(float) * 4, singa::kDeviceToHost);
  
  float* out = static_cast<float*>(host_out->mutable_data());
  EXPECT_EQ(1.0f, out[0]);
  EXPECT_EQ(2.0f, out[1]);
  EXPECT_EQ(3.0f, out[2]);
  EXPECT_EQ(4.0f, out[3]);
}

TEST(OpenCL_TensorMath, TensorMath_EltwiseAdd) {
  auto ocl_dev = std::make_shared<OpenclDevice>(OpenclDevice());

  Tensor in_1(Shape{1, 4}, ocl_dev);
  float a[] = {0.0f, 1.0f, 2.0f, 3.0f};
  in_1.CopyDataFromHostPtr(a, 4);
  Tensor in_2 = in_1.Clone();
  
  in_2 += in_1;
  
  CppCPU host;
  Block* host_out = host.NewBlock(sizeof(float) * 4);
  ocl_dev->CopyDataToFrom(host_out, in_2.block(), sizeof(float) * 4, singa::kDeviceToHost);
  
  float* out = static_cast<float*>(host_out->mutable_data());
  EXPECT_EQ(0.0f, out[0]);
  EXPECT_EQ(2.0f, out[1]);
  EXPECT_EQ(4.0f, out[2]);
  EXPECT_EQ(6.0f, out[3]); 
}
