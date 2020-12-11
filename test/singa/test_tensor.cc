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

#include "gtest/gtest.h"
#include "singa/core/tensor.h"
using singa::Device;
using singa::Shape;
using singa::Tensor;

TEST(TensorClass, Constructor) {
  singa::Tensor float_t(singa::Shape{2, 3});
  EXPECT_EQ(6u, float_t.Size());
  EXPECT_EQ(sizeof(float) * 6, float_t.MemSize());
  EXPECT_EQ(singa::kFloat32, float_t.data_type());
  auto s = float_t.shape();
  EXPECT_EQ(s[0], 2u);
  EXPECT_EQ(s[1], 3u);

  EXPECT_NE(float_t.device(), nullptr);

  singa::Tensor float16_t(Shape{2, 3}, singa::kFloat16);
  EXPECT_EQ(singa::kFloat16, float16_t.data_type());
  EXPECT_EQ(6u, float16_t.Size());
  EXPECT_EQ(12u, float16_t.block()->size());

  singa::Tensor x(float16_t);
  EXPECT_EQ(float16_t.Size(), x.Size());
  EXPECT_EQ(float16_t.block(), x.block());
  EXPECT_EQ(float16_t.data_type(), x.data_type());
  EXPECT_EQ(float16_t.device(), x.device());

  singa::Tensor y = float16_t;
  EXPECT_EQ(float16_t.Size(), x.Size());
  EXPECT_EQ(float16_t.block(), x.block());
  EXPECT_EQ(float16_t.data_type(), x.data_type());
  EXPECT_EQ(float16_t.device(), x.device());
}

TEST(TensorClass, Reshape) {
  Tensor t;
  t.Resize(Shape{2, 3});
  EXPECT_TRUE((Shape{2, 3} == t.shape()));

  t.Resize(Shape{3, 3, 4});
  EXPECT_TRUE((Shape{3, 3, 4} == t.shape()));

  t.Resize(Shape{12});
  EXPECT_TRUE((Shape{12} == t.shape()));

  Tensor o;
  EXPECT_TRUE(o.shape() != t.shape());
  o.Resize(Shape{3, 3});
  EXPECT_TRUE(o.shape() != t.shape());
}

#ifdef USE_CUDA

TEST(TensorClass, FloatAsTypeIntCuda) {
  auto cuda = std::make_shared<singa::CudaGPU>();

  Tensor t(Shape{3}, cuda);
  float data[] = {1.0f, 2.0f, 3.0f};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kFloat32, t.data_type());

  t = t.AsType(singa::kInt);

  EXPECT_EQ(singa::kInt, t.data_type());

  t.ToHost();
  const int* dptr2 = static_cast<const int*>(t.block()->data());
  EXPECT_EQ(1, dptr2[0]);
  EXPECT_EQ(2, dptr2[1]);
  EXPECT_EQ(3, dptr2[2]);
}

TEST(TensorClass, IntAsTypeFloatCuda) {
  auto cuda = std::make_shared<singa::CudaGPU>();

  Tensor t(Shape{3}, cuda, singa::kInt);
  int data[] = {1, 2, 3};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kInt, t.data_type());

  t = t.AsType(singa::kFloat32);

  EXPECT_EQ(singa::kFloat32, t.data_type());

  t.ToHost();
  const float* dptr2 = static_cast<const float*>(t.block()->data());
  EXPECT_EQ(1.0f, dptr2[0]);
  EXPECT_EQ(2.0f, dptr2[1]);
  EXPECT_EQ(3.0f, dptr2[2]);
}

#endif  // USE_CUDA

TEST(TensorClass, FloatAsTypeFloatCPU) {
  Tensor t(Shape{3});
  float data[] = {1.0f, 2.0f, 3.0f};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kFloat32, t.data_type());
  const float* dptr = static_cast<const float*>(t.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);

  Tensor t2 = t.AsType(singa::kFloat32);

  EXPECT_EQ(singa::kFloat32, t2.data_type());

  const float* dptr2 = static_cast<const float*>(t2.block()->data());
  EXPECT_EQ(1.0f, dptr2[0]);
  EXPECT_EQ(2.0f, dptr2[1]);
  EXPECT_EQ(3.0f, dptr2[2]);
}

TEST(TensorClass, FloatAsTypeIntCPU) {
  Tensor t(Shape{3});
  float data[] = {1.0f, 2.0f, 3.0f};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kFloat32, t.data_type());
  const float* dptr = static_cast<const float*>(t.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);

  Tensor t2 = t.AsType(singa::kInt);

  EXPECT_EQ(singa::kInt, t2.data_type());
  const int* dptr2 = static_cast<const int*>(t2.block()->data());
  EXPECT_EQ(1, dptr2[0]);
  EXPECT_EQ(2, dptr2[1]);
  EXPECT_EQ(3, dptr2[2]);
}

TEST(TensorClass, IntAsTypeFloatCPU) {
  Tensor t(Shape{3}, singa::kInt);
  int data[] = {1, 2, 3};
  t.CopyDataFromHostPtr(data, 3);
  EXPECT_EQ(singa::kInt, t.data_type());

  auto t2 = t.AsType(singa::kFloat32);

  EXPECT_EQ(singa::kFloat32, t2.data_type());

  const float* dptr2 = static_cast<const float*>(t2.block()->data());
  EXPECT_EQ(1.0f, dptr2[0]);
  EXPECT_EQ(2.0f, dptr2[1]);
  EXPECT_EQ(3.0f, dptr2[2]);
}

TEST(TensorClass, ToDevice) {
  Tensor t(Shape{2, 3});
  EXPECT_EQ(singa::defaultDevice, t.device());
  auto dev = std::make_shared<singa::CppCPU>();
  t.ToDevice(dev);
  EXPECT_NE(singa::defaultDevice, t.device());
}

TEST(TensorClass, CopyDataFromHostPtr) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);
  const float* dptr = static_cast<const float*>(t.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);
}

TEST(TensorClass, CopyData) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o(Shape{3});
  o.CopyData(t);
  const float* dptr = static_cast<const float*>(o.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);
}

TEST(TensorClass, Clone) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o = t.Clone();
  const float* dptr = static_cast<const float*>(o.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);
}

TEST(TensorClass, T) {
  Tensor t(Shape{2, 3});
  EXPECT_FALSE(t.transpose());
  Tensor o = t.T();  // o = t = {3,2}
  t.T();             // t = {2,3}
  EXPECT_EQ(true, o.transpose());
  EXPECT_EQ(t.block(), o.block());
  EXPECT_EQ(t.data_type(), o.data_type());
  EXPECT_EQ(t.shape()[0], o.shape()[1]);
  EXPECT_EQ(t.shape()[1], o.shape()[0]);
}

TEST(TensorClass, Repeat) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o = t.Repeat(vector<size_t>{2}, 9999);
  const float* dptr = static_cast<const float*>(o.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(1.0f, dptr[1]);
  EXPECT_FLOAT_EQ(2.0f, dptr[2]);
  EXPECT_FLOAT_EQ(2.0f, dptr[3]);
  EXPECT_FLOAT_EQ(3.0f, dptr[4]);
  EXPECT_FLOAT_EQ(3.0f, dptr[5]);
}

TEST(TensorClass, RepeatData) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o(Shape{6});
  o.RepeatData({2}, 9999, 2, t);
  const float* dptr = static_cast<const float*>(o.block()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(1.0f, dptr[1]);
  EXPECT_FLOAT_EQ(2.0f, dptr[2]);
  EXPECT_FLOAT_EQ(2.0f, dptr[3]);
  EXPECT_FLOAT_EQ(3.0f, dptr[4]);
  EXPECT_FLOAT_EQ(3.0f, dptr[5]);
}

TEST(TensorClass, Broadcast) {
  {
    Tensor a1(Shape{2, 3, 4, 5}), b1(Shape{5});
    auto c1 = Broadcast(a1, b1.shape()).shape();
    auto c2 = Broadcast(b1, a1.shape()).shape();
    EXPECT_EQ(c1[0], 2);
    EXPECT_EQ(c1[1], 3);
    EXPECT_EQ(c1[2], 4);
    EXPECT_EQ(c1[3], 5);

    EXPECT_EQ(c2[0], 2);
    EXPECT_EQ(c2[1], 3);
    EXPECT_EQ(c2[2], 4);
    EXPECT_EQ(c2[3], 5);
  }
  {
    Tensor a1(Shape{4, 5}), b1(Shape{2, 3, 4, 5});
    auto c1 = Broadcast(a1, b1.shape()).shape();
    auto c2 = Broadcast(b1, a1.shape()).shape();
    EXPECT_EQ(c1[0], 2);
    EXPECT_EQ(c1[1], 3);
    EXPECT_EQ(c1[2], 4);
    EXPECT_EQ(c1[3], 5);

    EXPECT_EQ(c2[0], 2);
    EXPECT_EQ(c2[1], 3);
    EXPECT_EQ(c2[2], 4);
    EXPECT_EQ(c2[3], 5);
  }
  {
    Tensor a1(Shape{1, 4, 5}), b1(Shape{2, 3, 1, 1});
    auto c1 = Broadcast(a1, b1.shape()).shape();
    auto c2 = Broadcast(b1, a1.shape()).shape();

    EXPECT_EQ(c1[0], 2);
    EXPECT_EQ(c1[1], 3);
    EXPECT_EQ(c1[2], 4);
    EXPECT_EQ(c1[3], 5);

    EXPECT_EQ(c2[0], 2);
    EXPECT_EQ(c2[1], 3);
    EXPECT_EQ(c2[2], 4);
    EXPECT_EQ(c2[3], 5);
  }
  {
    Tensor a1(Shape{3, 4, 5}), b1(Shape{2, 1, 1, 1});
    auto c1 = Broadcast(a1, b1.shape()).shape();
    auto c2 = Broadcast(b1, a1.shape()).shape();

    EXPECT_EQ(c1[0], 2);
    EXPECT_EQ(c1[1], 3);
    EXPECT_EQ(c1[2], 4);
    EXPECT_EQ(c1[3], 5);

    EXPECT_EQ(c2[0], 2);
    EXPECT_EQ(c2[1], 3);
    EXPECT_EQ(c2[2], 4);
    EXPECT_EQ(c2[3], 5);
  }
}
