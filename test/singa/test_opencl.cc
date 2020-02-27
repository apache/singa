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

using singa::Block;
using singa::CppCPU;
using singa::Shape;
using singa::Tensor;

#ifdef USE_OPENCL

using singa::OpenclDevice;

class OpenCL_TensorMath : public ::testing::Test {
 protected:
  OpenCL_TensorMath() {
    auto ocl_dev = std::make_shared<OpenclDevice>();

    a = Tensor(Shape{6}, ocl_dev);
    b = Tensor(Shape{6}, ocl_dev);
    c = Tensor(Shape{6, 1}, ocl_dev);
    d = Tensor(Shape{3, 2}, ocl_dev);
    e = Tensor(Shape{3, 2}, ocl_dev);
    empty10k = Tensor(Shape{10000}, ocl_dev);

    a.CopyDataFromHostPtr<float>(dat1, 6);
    b.CopyDataFromHostPtr<float>(dat2, 6);
    e.CopyDataFromHostPtr<float>(dat1, 6);
  }

  Tensor a, b, c, d, e;
  Tensor empty10k;
  const float dat1[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float dat2[6] = {1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f};
};

TEST_F(OpenCL_TensorMath, MemberAbs) {
  Tensor aa = a.Clone();
  Tensor bb = b.Clone();
  Tensor cc = aa - bb;

  cc.ToHost();
  const float *dptr = cc.data<float>();
  EXPECT_NEAR(-0.1, dptr[0], 1e-5);
  EXPECT_NEAR(-0.1, dptr[1], 1e-5);
  EXPECT_NEAR(-0.1, dptr[2], 1e-5);

  Tensor p = Abs(cc);
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(0.1, dptr1[0], 1e-5);
  EXPECT_NEAR(0.1, dptr1[1], 1e-5);
  EXPECT_NEAR(0.1, dptr1[2], 1e-5);
}

// TEST_F(OpenCL_TensorMath, MemberClamp) { }

TEST_F(OpenCL_TensorMath, MemberExp) {
  Tensor p = Exp(a);

  p.ToHost();
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(exp(1.0f), dptr1[0], 1e-5);
  EXPECT_NEAR(exp(2.0f), dptr1[1], 1e-5);
  EXPECT_NEAR(exp(3.0f), dptr1[2], 1e-5);
}

TEST_F(OpenCL_TensorMath, MemberLog) {
  Tensor p = Log(a);

  p.ToHost();
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(log(1.0f), dptr1[0], 1e-5);
  EXPECT_NEAR(log(2.0f), dptr1[1], 1e-5);
  EXPECT_NEAR(log(3.0f), dptr1[2], 1e-5);
}

TEST_F(OpenCL_TensorMath, MemberReLU) {
  Tensor aa = a.Clone();
  Tensor cc = aa - 2.0f;

  cc.ToHost();
  const float *dptr = cc.data<float>();
  EXPECT_NEAR(-1.0f, dptr[0], 1e-5);
  EXPECT_NEAR(0.0f, dptr[1], 1e-5);
  EXPECT_NEAR(1.0f, dptr[2], 1e-5);

  Tensor p = ReLU(cc);
  p.ToHost();
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(0.0f, dptr1[0], 1e-5);
  EXPECT_NEAR(0.0f, dptr1[1], 1e-5);
  EXPECT_NEAR(1.0f, dptr1[2], 1e-5);
}

TEST_F(OpenCL_TensorMath, MemberSigmoid) {
  Tensor p = Sigmoid(a);
  p.ToHost();
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(1.0f / (1.0f + exp(-1.0f)), dptr1[0], 1e-5);
  EXPECT_NEAR(1.0f / (1.0f + exp(-2.0f)), dptr1[1], 1e-5);
  EXPECT_NEAR(1.0f / (1.0f + exp(-3.0f)), dptr1[2], 1e-5);
}

TEST_F(OpenCL_TensorMath, MemberSign) {
  Tensor aa = a.Clone();
  Tensor cc = aa - 2.0f;
  cc.ToHost();
  const float *dptr = cc.data<float>();
  EXPECT_NEAR(-1.0f, dptr[0], 1e-5);
  EXPECT_NEAR(0.0f, dptr[1], 1e-5);
  EXPECT_NEAR(1.0f, dptr[2], 1e-5);

  Tensor p = Sign(cc);
  p.ToHost();
  const float *dptr1 = p.data<float>();
  EXPECT_EQ(-1.0f, dptr1[0]);
  EXPECT_EQ(0.0f, dptr1[1]);
  EXPECT_EQ(1.0f, dptr1[2]);
}

TEST_F(OpenCL_TensorMath, MemberSqrt) {
  Tensor p = Sqrt(a);
  p.ToHost();
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(sqrt(1.0), dptr1[0], 1e-5);
  EXPECT_NEAR(sqrt(2.0), dptr1[1], 1e-5);
  EXPECT_NEAR(sqrt(3.0), dptr1[2], 1e-5);
}

TEST_F(OpenCL_TensorMath, MemberSquare) {
  Tensor p = Square(a);
  p.ToHost();
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(1.0, dptr1[0], 1e-5);
  EXPECT_NEAR(4.0, dptr1[1], 1e-5);
  EXPECT_NEAR(9.0, dptr1[2], 1e-5);
}

TEST_F(OpenCL_TensorMath, MemberTanh) {
  Tensor p = Tanh(a);
  p.ToHost();
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(tanh(1.0), dptr1[0], 1e-5);
  EXPECT_NEAR(tanh(2.0), dptr1[1], 1e-5);
  EXPECT_NEAR(tanh(3.0), dptr1[2], 1e-5);
}

TEST_F(OpenCL_TensorMath, Sum) {
  float result = Sum(a);
  EXPECT_EQ(21.0f, result);
}

/*
TEST_F(OpenCL_TensorMath, SoftMax) {
  Tensor p1 = SoftMax(Reshape(e, Shape{1, 6}));
  p1.ToHost();
  const float *dptr1 = p1.data<float>();
  float sum = 0;
  for (int i = 0; i < 6; i++) sum += exp(i + 1);
  EXPECT_NEAR(exp(1) / sum, dptr1[0], 1e-5);
  EXPECT_NEAR(exp(3) / sum, dptr1[2], 1e-5);
  EXPECT_NEAR(exp(5) / sum, dptr1[4], 1e-5);
  EXPECT_NEAR(exp(2) / sum, dptr1[1], 1e-5);
  EXPECT_NEAR(exp(4) / sum, dptr1[3], 1e-5);
  EXPECT_NEAR(exp(6) / sum, dptr1[5], 1e-5);

  Tensor p2 = SoftMax(e);
  p2.ToHost();
  const float *dptr2 = p2.data<float>();
  EXPECT_NEAR(exp(1) / (exp(1) + exp(2)), dptr2[0], 1e-5);
  EXPECT_NEAR(exp(2) / (exp(1) + exp(2)), dptr2[1], 1e-5);
}
*/

TEST_F(OpenCL_TensorMath, MemberLT) {
  Tensor p1 = a < 2.0f;
  p1.ToHost();
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(1.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(0.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(0.0f, dptr1[2]);
}

TEST_F(OpenCL_TensorMath, MemberLE) {
  Tensor p1 = a <= 2.0f;
  p1.ToHost();
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(1.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(1.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(0.0f, dptr1[2]);
}

TEST_F(OpenCL_TensorMath, MemberGT) {
  Tensor p1 = a > 2.0f;
  p1.ToHost();
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(0.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(0.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(1.0f, dptr1[2]);
}

TEST_F(OpenCL_TensorMath, MemberGE) {
  Tensor p1 = a >= 2.0f;
  p1.ToHost();
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(0.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(1.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(1.0f, dptr1[2]);
}

TEST_F(OpenCL_TensorMath, MemberPow) {
  Tensor p1 = Pow(b, 3.0f);
  p1.ToHost();
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(pow(1.1f, 3.0f), dptr1[0]);
  EXPECT_FLOAT_EQ(pow(2.1f, 3.0f), dptr1[1]);
  EXPECT_FLOAT_EQ(pow(3.1f, 3.0f), dptr1[2]);

  Tensor p2 = Pow(a, b);
  p2.ToHost();
  const float *dptr2 = p2.data<float>();
  EXPECT_FLOAT_EQ(pow(1.0f, 1.1f), dptr2[0]);
  EXPECT_FLOAT_EQ(pow(2.0f, 2.1f), dptr2[1]);
  EXPECT_FLOAT_EQ(pow(3.0f, 3.1f), dptr2[2]);
}

TEST_F(OpenCL_TensorMath, MemberSub) {
  Tensor p1 = a - b;
  p1.ToHost();
  const float *dptr1 = p1.data<float>();
  EXPECT_NEAR(-0.1, dptr1[0], 1e-5);
  EXPECT_NEAR(-0.1, dptr1[1], 1e-5);
  EXPECT_NEAR(-0.1, dptr1[2], 1e-5);
}

TEST_F(OpenCL_TensorMath, MemberEltwiseMult) {
  Tensor p1 = a * b;
  p1.ToHost();
  const float *dptr1 = p1.data<float>();
  EXPECT_NEAR(1.0 * 1.1, dptr1[0], 1e-5);
  EXPECT_NEAR(2.0 * 2.1, dptr1[1], 1e-5);
  EXPECT_NEAR(3.0 * 3.1, dptr1[2], 1e-5);
}

TEST_F(OpenCL_TensorMath, MemberDiv) {
  Tensor p1 = a / b;
  p1.ToHost();
  const float *dptr1 = p1.data<float>();
  EXPECT_NEAR(1.0 / 1.1, dptr1[0], 1e-5);
  EXPECT_NEAR(2.0 / 2.1, dptr1[1], 1e-5);
  EXPECT_NEAR(3.0 / 3.1, dptr1[2], 1e-5);

  Tensor p2 = Div(10.0f, b);
  p2.ToHost();
  const float *dptr2 = p2.data<float>();
  EXPECT_NEAR(10.0 / 1.1, dptr2[0], 1e-5);
  EXPECT_NEAR(10.0 / 2.1, dptr2[1], 1e-5);
  EXPECT_NEAR(10.0 / 3.1, dptr2[2], 1e-5);

  Tensor p3 = a / 8.0f;
  p3.ToHost();
  const float *dptr3 = p3.data<float>();
  EXPECT_NEAR(1.0 / 8.0, dptr3[0], 1e-5);
  EXPECT_NEAR(2.0 / 8.0, dptr3[1], 1e-5);
  EXPECT_NEAR(3.0 / 8.0, dptr3[2], 1e-5);
}

// **************************************
// Random functions
// **************************************

TEST_F(OpenCL_TensorMath, Bernoulli) {
  const float p = 0.3f;
  Bernoulli(p, &empty10k);
  empty10k.ToHost();
  const float *out = empty10k.data<float>();
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
  const float *out = empty10k.data<float>();
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
  const float *out = empty10k.data<float>();
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

TEST_F(OpenCL_TensorMath, MemberAddTensor) {
  Tensor aa = a.Clone();
  aa += a;
  aa.ToHost();
  const float *dptr = aa.data<float>();
  EXPECT_FLOAT_EQ(2.0f, dptr[0]);
  EXPECT_FLOAT_EQ(4.0f, dptr[1]);
  EXPECT_FLOAT_EQ(6.0f, dptr[2]);

  // check p is initialized to 0
  Tensor p(Shape{6});
  p += aa;
  p.ToHost();
  const float *dptr1 = p.data<float>();
  EXPECT_FLOAT_EQ(2.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(4.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(6.0f, dptr1[2]);

  a += b;
  a.ToHost();
  const float *dptr2 = a.data<float>();
  EXPECT_FLOAT_EQ(2.1f, dptr2[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr2[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr2[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr2[5]);
}

TEST_F(OpenCL_TensorMath, AddTensors) {
  Tensor ret(a.shape(), a.device(), a.data_type());
  Add(a, b, &ret);
  ret.ToHost();
  const float *dptr = ret.data<float>();
  EXPECT_FLOAT_EQ(2.1f, dptr[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr[5]);

  Tensor d = a + b;
  d.ToHost();
  const float *dptr2 = d.data<float>();
  EXPECT_FLOAT_EQ(2.1f, dptr2[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr2[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr2[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr2[5]);

  Add(a, b, &a);
  a.ToHost();
  const float *dptr1 = a.data<float>();
  EXPECT_FLOAT_EQ(2.1f, dptr1[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr1[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr1[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr1[5]);
}

TEST_F(OpenCL_TensorMath, SetValue) {
  Tensor t(Shape{4});
  t.SetValue(0.3f);
  t.ToHost();
  const float *ptr = t.data<float>();
  for (int i = 0; i < 4; i++) EXPECT_FLOAT_EQ(ptr[i], 0.3f);
}

TEST_F(OpenCL_TensorMath, Axpy) {
  Tensor ret(b.shape(), b.device(), b.data_type());
  const float zero[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  ret.CopyDataFromHostPtr<float>(zero, 6);
  Axpy(10.0f, b, &ret);
  ret.ToHost();
  const float *out = ret.data<float>();

  EXPECT_EQ(11.0f, out[0]);  // 1.1 * 10 + 0 = 11
  EXPECT_EQ(21.0f, out[1]);  // 2.1 * 10 + 1 = 22
  EXPECT_EQ(31.0f, out[2]);  // 3.1 * 10 + 2 = 33
  EXPECT_EQ(41.0f, out[3]);  // 4.1 * 10 + 3 = 44
}

TEST_F(OpenCL_TensorMath, GEMM) {
  a.Reshape(Shape{6, 1});
  Tensor result = Mult(a.T(), a);
  result.ToHost();
  const float *out = result.data<float>();

  EXPECT_EQ(91.0f, out[0]);
}

// TODO: ComputeCrossEntropy, SoftmaxCrossEntropy
//
#endif  // USE_OPENCL
