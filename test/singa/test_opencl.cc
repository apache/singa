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
#include "singa/core/tensor.h"
#include "singa/proto/core.pb.h"
using singa::CppCPU;
using singa::Block;
using singa::Shape;
using singa::Tensor;

#ifdef USE_OPENCL
using singa::OpenclDevice;
class OpenCL_TensorMath : public ::testing::Test {
protected:

  OpenCL_TensorMath() {
    for (int i = 0; i < 4; i++) {
      float4[i] = (float)i;
      float4zero[i] = 0.0f;
    }

    for (int i = 0; i < 16; i++) {
      float16[i] = (float)i;
      float16zero[i] = 0.0f;
    }

    auto ocl_dev = std::make_shared<OpenclDevice>();

    tf4in = Tensor(Shape{1, 4}, ocl_dev);
    tf4in.CopyDataFromHostPtr(float4, 4);

    tf4zin = Tensor(Shape{1, 4}, ocl_dev);
    tf4zin.CopyDataFromHostPtr(float4zero, 4);

    tf16in = Tensor(Shape{4, 4}, ocl_dev);
    tf16in.CopyDataFromHostPtr(float16, 16);

    tf16zin = Tensor(Shape{4, 4}, ocl_dev);
    tf16zin.CopyDataFromHostPtr(float16zero, 16);

    float empty[10000] = {};
    empty10k = Tensor(Shape{10000}, ocl_dev);
    empty10k.CopyDataFromHostPtr(empty, 10000);
  }

  float float4[4];
  float float4zero[4];
  float float16[16];
  float float16zero[16];

  Tensor tf4in, tf16in;
  Tensor tf4zin, tf16zin;
  Tensor empty10k;
};


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

// Tensor tests, uses OpenCL_TensorMath class defined above.

TEST_F(OpenCL_TensorMath, CopyDataToDevice) {
  tf4in.ToHost();
  const float* out = tf4in.data<float>();

  EXPECT_EQ(1.0f, out[1]);
  EXPECT_EQ(3.0f, out[3]);
}


TEST_F(OpenCL_TensorMath, MemberAbs) {
  tf4in = Abs(tf4in);

  tf4in.ToHost();
  const float* out = tf4in.data<float>();

  EXPECT_EQ(0.0f, out[0]);
  EXPECT_EQ(1.0f, out[1]);
  EXPECT_EQ(2.0f, out[2]);
  EXPECT_EQ(3.0f, out[3]);
}


TEST_F(OpenCL_TensorMath, MemberExp) {
  tf4in = Exp(tf4in);

  tf4in.ToHost();
  const float* out = tf4in.data<float>();

  EXPECT_NEAR(exp(0.0f), out[0], 1e-5);
  EXPECT_NEAR(exp(1.0f), out[1], 1e-5);
  EXPECT_NEAR(exp(2.0f), out[2], 1e-5);
  EXPECT_NEAR(exp(3.0f), out[3], 1e-5);
}


TEST_F(OpenCL_TensorMath, MemberLog) {
  tf4in = Log(tf4in);

  tf4in.ToHost();
  const float* out = tf4in.data<float>();

//  EXPECT_NEAR(log(0.0f), out[0], 1e-5); // Evaluates to neg infinity.
  EXPECT_NEAR(log(1.0f), out[1], 1e-5);
  EXPECT_NEAR(log(2.0f), out[2], 1e-5);
  EXPECT_NEAR(log(3.0f), out[3], 1e-5);
}


TEST_F(OpenCL_TensorMath, MemberReLU) {
  tf4in -= 2.0f;
  Tensor result = ReLU(tf4in);

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_NEAR(0.0f, out[0], 1e-5);
  EXPECT_NEAR(0.0f, out[1], 1e-5);
  EXPECT_NEAR(0.0f, out[2], 1e-5);
  EXPECT_NEAR(1.0f, out[3], 1e-5);
}


TEST_F(OpenCL_TensorMath, MemberSigmoid) {
  tf4in = Sigmoid(tf4in);

  tf4in.ToHost();
  const float* out = tf4in.data<float>();

  EXPECT_NEAR(1.0f / (1.0f + exp(-0.0f)), out[0], 1e-5);
  EXPECT_NEAR(1.0f / (1.0f + exp(-1.0f)), out[1], 1e-5);
  EXPECT_NEAR(1.0f / (1.0f + exp(-2.0f)), out[2], 1e-5);
}


TEST_F(OpenCL_TensorMath, MemberSign) {
  tf4in -= 1.0f;

  tf4in.ToHost();
  const float* out = tf4in.data<float>();

  EXPECT_NEAR(-1.0f, out[0], 1e-5);
  EXPECT_NEAR(0.0f, out[1], 1e-5);
  EXPECT_NEAR(1.0f, out[2], 1e-5);
  EXPECT_NEAR(2.0f, out[3], 1e-5);
}


TEST_F(OpenCL_TensorMath, MemberSqrt) {
  tf4in = Sqrt(tf4in);

  tf4in.ToHost();
  const float* out = tf4in.data<float>();

  EXPECT_NEAR(0.0f, out[0], 1e-5);
  EXPECT_NEAR(1.0f, out[1], 1e-5);
  EXPECT_NEAR(sqrt(2.0f), out[2], 1e-5);
  EXPECT_NEAR(sqrt(3.0f), out[3], 1e-5);
}


TEST_F(OpenCL_TensorMath, MemberSquare) {
  tf4in = Square(tf4in);

  tf4in.ToHost();
  const float* out = tf4in.data<float>();

  EXPECT_NEAR(0.0f, out[0], 1e-5);
  EXPECT_NEAR(1.0f, out[1], 1e-5);
  EXPECT_NEAR(4.0f, out[2], 1e-5);
  EXPECT_NEAR(9.0f, out[3], 1e-5);
}


TEST_F(OpenCL_TensorMath, MemberTanh) {
  tf4in = Tanh(tf4in);

  tf4in.ToHost();
  const float* out = tf4in.data<float>();

  EXPECT_NEAR(0.0f, out[0], 1e-5);
  EXPECT_NEAR(tanh(1.0f), out[1], 1e-5);
  EXPECT_NEAR(tanh(2.0f), out[2], 1e-5);
  EXPECT_NEAR(tanh(3.0f), out[3], 1e-5);
}


TEST_F(OpenCL_TensorMath, Sum) {
  Tensor result = Sum(tf4in, 0);

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_NEAR(0.0f, out[0], 1e-5);
  EXPECT_NEAR(1.0f, out[1], 1e-5);
  EXPECT_NEAR(2.0f, out[2], 1e-5);
  EXPECT_NEAR(3.0f, out[3], 1e-5);
}

TEST_F(OpenCL_TensorMath, MemberLT) {
  Tensor result = tf4in < 2.0f;

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_FLOAT_EQ(1.0f, out[0]);
  EXPECT_FLOAT_EQ(1.0f, out[1]);
  EXPECT_FLOAT_EQ(0.0f, out[2]);
  EXPECT_FLOAT_EQ(0.0f, out[3]);
}


TEST_F(OpenCL_TensorMath, MemberLE) {
  Tensor result = tf4in <= 2.0f;

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_FLOAT_EQ(1.0f, out[0]);
  EXPECT_FLOAT_EQ(1.0f, out[1]);
  EXPECT_FLOAT_EQ(1.0f, out[2]);
  EXPECT_FLOAT_EQ(0.0f, out[3]);
}


TEST_F(OpenCL_TensorMath, MemberGT) {
  Tensor result = tf4in > 2.0f;

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_FLOAT_EQ(0.0f, out[0]);
  EXPECT_FLOAT_EQ(0.0f, out[1]);
  EXPECT_FLOAT_EQ(0.0f, out[2]);
  EXPECT_FLOAT_EQ(1.0f, out[3]);
}


TEST_F(OpenCL_TensorMath, MemberGE) {
  Tensor result = tf4in >= 2.0f;

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_FLOAT_EQ(0.0f, out[0]);
  EXPECT_FLOAT_EQ(0.0f, out[1]);
  EXPECT_FLOAT_EQ(1.0f, out[2]);
  EXPECT_FLOAT_EQ(1.0f, out[3]);
}


TEST_F(OpenCL_TensorMath, MemberPow) {
  Tensor result = Pow(tf4in, 2.0f);

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_FLOAT_EQ(0.0f, out[0]);
  EXPECT_FLOAT_EQ(1.0f, out[1]);
  EXPECT_FLOAT_EQ(4.0f, out[2]);
  EXPECT_FLOAT_EQ(9.0f, out[3]);

  result = Pow(tf4in, tf4in);

  result.ToHost();
  const float* out1 = result.data<float>();

  EXPECT_FLOAT_EQ(1.0f, out1[0]); // 0 ^ 0 is 1, apparently.
  EXPECT_FLOAT_EQ(1.0f, out1[1]);
  EXPECT_FLOAT_EQ(4.0f, out1[2]);
  EXPECT_FLOAT_EQ(27.0f, out1[3]);
}


TEST_F(OpenCL_TensorMath, MemberSub) {
  Tensor result = tf4in - tf4zin;

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_FLOAT_EQ(0.0f, out[0]);
  EXPECT_FLOAT_EQ(1.0f, out[1]);
  EXPECT_FLOAT_EQ(2.0f, out[2]);
  EXPECT_FLOAT_EQ(3.0f, out[3]);

  result = tf4in - 0.0f;

  result.ToHost();
  const float* out1 = result.data<float>();

  EXPECT_FLOAT_EQ(0.0f, out1[0]);
  EXPECT_FLOAT_EQ(1.0f, out1[1]);
  EXPECT_FLOAT_EQ(2.0f, out1[2]);
  EXPECT_FLOAT_EQ(3.0f, out1[3]);
}


TEST_F(OpenCL_TensorMath, MemberEltwiseMult) {
  Tensor result = tf4in * tf4zin;

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_FLOAT_EQ(0.0f, out[0]);
  EXPECT_FLOAT_EQ(0.0f, out[1]);
  EXPECT_FLOAT_EQ(0.0f, out[2]);
  EXPECT_FLOAT_EQ(0.0f, out[3]);

  result = tf4in * 10.0f;

  result.ToHost();
  const float* out1 = result.data<float>();

  EXPECT_FLOAT_EQ(0.0f, out1[0]);
  EXPECT_FLOAT_EQ(10.0f, out1[1]);
  EXPECT_FLOAT_EQ(20.0f, out1[2]);
  EXPECT_FLOAT_EQ(30.0f, out1[3]);
}


TEST_F(OpenCL_TensorMath, MemberDiv) {
  Tensor result = tf4in / tf4in;

  result.ToHost();
  const float* out = result.data<float>();

//  EXPECT_FLOAT_EQ(0.0f, out[0]); // Divide by zero.
  EXPECT_FLOAT_EQ(1.0f, out[1]);
  EXPECT_FLOAT_EQ(1.0f, out[2]);
  EXPECT_FLOAT_EQ(1.0f, out[3]);

  result = tf4in / 10.0f;

  result.ToHost();
  const float* out1 = result.data<float>();

  EXPECT_FLOAT_EQ(0.0f, out1[0]);
  EXPECT_FLOAT_EQ(0.1f, out1[1]);
  EXPECT_FLOAT_EQ(0.2f, out1[2]);
  EXPECT_FLOAT_EQ(0.3f, out1[3]);

  result = Div(10.0f, tf4in);

  result.ToHost();
  const float* out2 = result.data<float>();

//  EXPECT_FLOAT_EQ(0.0f, out[0]); // Divide by 0.
  EXPECT_FLOAT_EQ(10.0f, out2[1]);
  EXPECT_FLOAT_EQ(5.0f, out2[2]);
  EXPECT_NEAR((10.0f / 3.0f), out2[3], 1e-5);
}

// **************************************
// Random functions
// **************************************

TEST_F(OpenCL_TensorMath, Bernoulli) {
  const float p = 0.3f;

  Bernoulli(p, &empty10k);

  empty10k.ToHost();
  const float* out = empty10k.data<float>();

  float sum = 0.0f;
  for (int i = 0; i < 10000; i++) sum += out[i];

  float mean = sum / 10000;

  EXPECT_NEAR(mean, p, 1e-2);

  sum = 0.0f;
  for (int i = 0; i < 10000; i++) sum += (out[i] - mean) * (out[i] - mean);
  float variance = sum / 9999;

  EXPECT_NEAR(variance, p * (1 - p), 1e-2);
}


TEST_F(OpenCL_TensorMath, Gaussian) {
  Gaussian(0.0f, 1.0f, &empty10k);

  empty10k.ToHost();
  const float* out = empty10k.data<float>();

  float sum = 0.0f;
  for (int i = 0; i < 10000; i++) sum += out[i];
  float mean = sum / 10000;

  EXPECT_NEAR(mean, 0.0f, 1e-2);

  sum = 0.0f;
  for (int i = 0; i < 10000; i++) sum += (out[i] - mean) * (out[i] - mean);
  float variance = sum / 9999;

  EXPECT_NEAR(variance, 1.0f, 1e-2);
}


TEST_F(OpenCL_TensorMath, Uniform) {
  Uniform(0.1f, 0.2f, &empty10k);

  empty10k.ToHost();
  const float* out = empty10k.data<float>();

  float sum = 0.0f;
  for (int i = 0; i < 10000; i++) sum += out[i];
  float mean = sum / 10000;

  EXPECT_NEAR(mean, 0.15f, 1e-2);

  sum = 0.0f;
  for (int i = 0; i < 10000; i++) sum += (out[i] - mean) * (out[i] - mean);
  float variance = sum / 9999;

  EXPECT_NEAR(variance, 0.01f, 1e-2);
}

// *********************************************************
// BLAS functions, ref to http://docs.nvidia.com/cuda/cublas
// *********************************************************


TEST_F(OpenCL_TensorMath, EltwiseAdd) {
  Tensor result = tf4in + tf4in;

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_EQ(0.0f, out[0]);
  EXPECT_EQ(2.0f, out[1]);
  EXPECT_EQ(4.0f, out[2]);
  EXPECT_EQ(6.0f, out[3]);

  result = tf4in + tf4zin;

  result.ToHost();
  const float* out1 = result.data<float>();

  EXPECT_EQ(0.0f, out1[0]);
  EXPECT_EQ(1.0f, out1[1]);
  EXPECT_EQ(2.0f, out1[2]);
  EXPECT_EQ(3.0f, out1[3]);

  result = Tensor(tf4in.shape(), tf4in.device(), tf4in.data_type());
  Add(tf4in, tf4in, &result);

  result.ToHost();
  const float* out2 = result.data<float>();

  EXPECT_EQ(0.0f, out2[0]);
  EXPECT_EQ(2.0f, out2[1]);
  EXPECT_EQ(4.0f, out2[2]);
  EXPECT_EQ(6.0f, out2[3]);

  result = tf4in + 1.0f;

  result.ToHost();
  const float* out3 = result.data<float>();

  EXPECT_EQ(1.0f, out3[0]);
  EXPECT_EQ(2.0f, out3[1]);
  EXPECT_EQ(3.0f, out3[2]);
  EXPECT_EQ(4.0f, out3[3]);
}


TEST_F(OpenCL_TensorMath, SetValue) {
  const float one_third = 1.0f / 3.0f;
  empty10k.SetValue(one_third);

  empty10k.ToHost();
  const float* out = empty10k.data<float>();

  EXPECT_EQ(one_third, out[0]);
  EXPECT_EQ(one_third, out[1]);
  EXPECT_EQ(one_third, out[1024]);
  EXPECT_EQ(one_third, out[4096]);
  EXPECT_EQ(one_third, out[9998]);
  EXPECT_EQ(one_third, out[9999]);
}


TEST_F(OpenCL_TensorMath, Axpy) {
  Axpy(10.0f, tf4in, &tf4in);

  tf4in.ToHost();
  const float* out = tf4in.data<float>();

  EXPECT_EQ(0.0f, out[0]);  // 0 * 10 + 0 = 0
  EXPECT_EQ(11.0f, out[1]); // 1 * 10 + 1 = 11
  EXPECT_EQ(22.0f, out[2]); // 2 * 10 + 2 = 22
  EXPECT_EQ(33.0f, out[3]); // 3 * 10 + 3 = 33
}

TEST_F(OpenCL_TensorMath, Mult) {
  Tensor result = Mult(tf4in, tf4zin.T()); // Multiply with zero.

  result.ToHost();
  const float* out = result.data<float>();

  EXPECT_EQ(0.0f, out[0]); // 1x4 * 4x1 = 1x1.

  result = Mult(tf4in, tf4in.T());

  result.ToHost();
  const float* out0 = result.data<float>();

  EXPECT_EQ(14.0f, out0[0]); // 1x4 * 4x1 = 1x1.

  tf16zin.SetValue(10.0f); // Multiply with 10.0.
  result = Mult(tf16in, tf16zin); // 4x4 * 4x4 = 4x4.

  result.ToHost();
  const float* out1 = result.data<float>();
  EXPECT_EQ(240.0f, out1[0]);
  EXPECT_EQ(280.0f, out1[1]);
  EXPECT_EQ(320.0f, out1[2]);
  EXPECT_EQ(360.0f, out1[3]);

  EXPECT_EQ(240.0f, out1[4]);
  EXPECT_EQ(280.0f, out1[5]);
  EXPECT_EQ(320.0f, out1[6]);
  EXPECT_EQ(360.0f, out1[7]);

  EXPECT_EQ(240.0f, out1[8]);
  EXPECT_EQ(280.0f, out1[9]);
  EXPECT_EQ(320.0f, out1[10]);
  EXPECT_EQ(360.0f, out1[11]);

  EXPECT_EQ(240.0f, out1[12]);
  EXPECT_EQ(280.0f, out1[13]);
  EXPECT_EQ(320.0f, out1[14]);
  EXPECT_EQ(360.0f, out1[15]);
}



// TODO: ComputeCrossEntropy, SoftmaxCrossEntropy
//
#endif  // USE_OPENCL
