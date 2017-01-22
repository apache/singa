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
using singa::Tensor;
using singa::Shape;
using singa::Device;

class TestTensorMath : public ::testing::Test {
 protected:
  virtual void SetUp() {
    a.Reshape(singa::Shape{6});
    b.Reshape(singa::Shape{6});
    c.Reshape(singa::Shape{6, 1});
    d.Reshape(singa::Shape{3, 2});
    e.Reshape(singa::Shape{3, 2});

    a.CopyDataFromHostPtr<float>(dat1, 6);
    b.CopyDataFromHostPtr<float>(dat2, 6);
    e.CopyDataFromHostPtr<float>(dat1, 6);
  }
  Tensor a, b, c, d, e;
  const float dat1[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float dat2[6] = {1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f};
};

TEST_F(TestTensorMath, MemberAbs) {
  Tensor aa = a.Clone();
  Tensor bb = b.Clone();
  Tensor cc = aa - bb;
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

TEST_F(TestTensorMath, MemberExp) {
  Tensor p = Exp(a);
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(exp(1.0f), dptr1[0], 1e-5);
  EXPECT_NEAR(exp(2.0f), dptr1[1], 1e-5);
  EXPECT_NEAR(exp(3.0f), dptr1[2], 1e-5);
}

TEST_F(TestTensorMath, MemberLog) {
  Tensor p = Log(a);
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(log(1.0f), dptr1[0], 1e-5);
  EXPECT_NEAR(log(2.0f), dptr1[1], 1e-5);
  EXPECT_NEAR(log(3.0f), dptr1[2], 1e-5);
}

TEST_F(TestTensorMath, MemberReLU) {
  Tensor aa = a.Clone();
  Tensor cc = aa - 2.0f;
  const float *dptr = cc.data<float>();
  EXPECT_NEAR(-1.0f, dptr[0], 1e-5);
  EXPECT_NEAR(0.0f, dptr[1], 1e-5);
  EXPECT_NEAR(1.0f, dptr[2], 1e-5);

  Tensor p = ReLU(cc);
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(0.0f, dptr1[0], 1e-5);
  EXPECT_NEAR(0.0f, dptr1[1], 1e-5);
  EXPECT_NEAR(1.0f, dptr1[2], 1e-5);
}

TEST_F(TestTensorMath, MemberSigmoid) {
  Tensor p = Sigmoid(a);
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(1.0f / (1.0f + exp(-1.0f)), dptr1[0], 1e-5);
  EXPECT_NEAR(1.0f / (1.0f + exp(-2.0f)), dptr1[1], 1e-5);
  EXPECT_NEAR(1.0f / (1.0f + exp(-3.0f)), dptr1[2], 1e-5);
}

TEST_F(TestTensorMath, MemberSign) {
  Tensor aa = a.Clone();
  Tensor cc = aa - 2.0f;
  const float *dptr = cc.data<float>();
  EXPECT_NEAR(-1.0f, dptr[0], 1e-5);
  EXPECT_NEAR(0.0f, dptr[1], 1e-5);
  EXPECT_NEAR(1.0f, dptr[2], 1e-5);

  Tensor p = Sign(cc);
  const float *dptr1 = p.data<float>();
  EXPECT_EQ(-1.0f, dptr1[0]);
  EXPECT_EQ(0.0f, dptr1[1]);
  EXPECT_EQ(1.0f, dptr1[2]);
}

TEST_F(TestTensorMath, MemberSqrt) {
  Tensor p = Sqrt(a);
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(sqrt(1.0), dptr1[0], 1e-5);
  EXPECT_NEAR(sqrt(2.0), dptr1[1], 1e-5);
  EXPECT_NEAR(sqrt(3.0), dptr1[2], 1e-5);
}

TEST_F(TestTensorMath, MemberSquare) {
  Tensor p = Square(a);
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(1.0, dptr1[0], 1e-5);
  EXPECT_NEAR(4.0, dptr1[1], 1e-5);
  EXPECT_NEAR(9.0, dptr1[2], 1e-5);
}

TEST_F(TestTensorMath, MemberTanh) {
  Tensor p = Tanh(a);
  const float *dptr1 = p.data<float>();
  EXPECT_NEAR(tanh(1.0), dptr1[0], 1e-5);
  EXPECT_NEAR(tanh(2.0), dptr1[1], 1e-5);
  EXPECT_NEAR(tanh(3.0), dptr1[2], 1e-5);
}

TEST_F(TestTensorMath, Sum) {
  Tensor p1 = Sum(e, 0);
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(9.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(12.0f, dptr1[1]);

  Tensor p2(Shape{3, 1});
  p2 = Sum(e, 1);
  const float *dptr2 = p2.data<float>();
  EXPECT_FLOAT_EQ(3.0f, dptr2[0]);
  EXPECT_FLOAT_EQ(7.0f, dptr2[1]);
  EXPECT_FLOAT_EQ(11.0f, dptr2[2]);
}

TEST_F(TestTensorMath, SoftMax) {
  Tensor p1 = SoftMax(Reshape(e, Shape{1, 6}));
  const float *dptr1 = p1.data<float>();
  float sum = 0;
  for (int i = 0; i < 6; i++) sum += (float)exp(i + 1);
  EXPECT_NEAR(exp(1) / sum, dptr1[0], 1e-5);
  EXPECT_NEAR(exp(3) / sum, dptr1[2], 1e-5);
  EXPECT_NEAR(exp(5) / sum, dptr1[4], 1e-5);
  EXPECT_NEAR(exp(2) / sum, dptr1[1], 1e-5);
  EXPECT_NEAR(exp(4) / sum, dptr1[3], 1e-5);
  EXPECT_NEAR(exp(6) / sum, dptr1[5], 1e-5);

  Tensor p2 = SoftMax(e);
  const float *dptr2 = p2.data<float>();
  EXPECT_NEAR(exp(1) / (exp(1) + exp(2)), dptr2[0], 1e-5);
  EXPECT_NEAR(exp(2) / (exp(1) + exp(2)), dptr2[1], 1e-5);
}

TEST_F(TestTensorMath, MemberLT) {
  Tensor p1 = a < 2.0f;
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(1.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(0.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(0.0f, dptr1[2]);
}

TEST_F(TestTensorMath, MemberLE) {
  Tensor p1 = a <= 2.0f;
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(1.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(1.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(0.0f, dptr1[2]);
}

TEST_F(TestTensorMath, MemberGT) {
  Tensor p1 = a > 2.0f;
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(0.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(0.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(1.0f, dptr1[2]);
}

TEST_F(TestTensorMath, MemberGE) {
  Tensor p1 = a >= 2.0f;
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(0.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(1.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(1.0f, dptr1[2]);
}

TEST_F(TestTensorMath, MemberPow) {
  Tensor p1 = Pow(b, 3.0f);
  const float *dptr1 = p1.data<float>();
  EXPECT_FLOAT_EQ(pow(1.1f, 3.0f), dptr1[0]);
  EXPECT_FLOAT_EQ(pow(2.1f, 3.0f), dptr1[1]);
  EXPECT_FLOAT_EQ(pow(3.1f, 3.0f), dptr1[2]);

  // TODO(Yuchen): check pow(tensor a, tensor b) and add testcase after the
  // function is complete
  // Tensor p2 = Pow(a,b);
  // const float *dptr2 = p2.data<float>();
  // EXPECT_FLOAT_EQ(pow(1.0f,1.1f), dptr2[0]);
  // EXPECT_FLOAT_EQ(pow(2.0f,2.1f), dptr2[1]);
  // EXPECT_FLOAT_EQ(pow(3.0f,3.1f), dptr2[2]);
}

TEST_F(TestTensorMath, MemberSub) {
  Tensor p1 = a - b;
  const float *dptr1 = p1.data<float>();
  EXPECT_NEAR(-0.1, dptr1[0], 1e-5);
  EXPECT_NEAR(-0.1, dptr1[1], 1e-5);
  EXPECT_NEAR(-0.1, dptr1[2], 1e-5);
}

TEST_F(TestTensorMath, MemberEltwiseMult) {
  Tensor p1 = a * b;
  const float *dptr1 = p1.data<float>();
  EXPECT_NEAR(1.0 * 1.1, dptr1[0], 1e-5);
  EXPECT_NEAR(2.0 * 2.1, dptr1[1], 1e-5);
  EXPECT_NEAR(3.0 * 3.1, dptr1[2], 1e-5);
}

TEST_F(TestTensorMath, MemberDiv) {
  Tensor p1 = a / b;
  const float *dptr1 = p1.data<float>();
  EXPECT_NEAR(1.0 / 1.1, dptr1[0], 1e-5);
  EXPECT_NEAR(2.0 / 2.1, dptr1[1], 1e-5);
  EXPECT_NEAR(3.0 / 3.1, dptr1[2], 1e-5);

  Tensor p2 = Div(10.0f, b);
  const float *dptr2 = p2.data<float>();
  EXPECT_NEAR(10.0 / 1.1, dptr2[0], 1e-5);
  EXPECT_NEAR(10.0 / 2.1, dptr2[1], 1e-5);
  EXPECT_NEAR(10.0 / 3.1, dptr2[2], 1e-5);

  Tensor p3 = a / 8.0f;
  const float *dptr3 = p3.data<float>();
  EXPECT_NEAR(1.0 / 8.0, dptr3[0], 1e-5);
  EXPECT_NEAR(2.0 / 8.0, dptr3[1], 1e-5);
  EXPECT_NEAR(3.0 / 8.0, dptr3[2], 1e-5);
}

TEST_F(TestTensorMath, MemberBernoulli) {
  Tensor p1(Shape{10000});
  Bernoulli(0.3f, &p1);
  const float *dptr1 = p1.data<float>();
  float sum = 0;
  for (int i = 0; i < 10000; i++) sum += dptr1[i];
  float mean = sum / 10000;
  EXPECT_NEAR(mean, 0.3f, 1e-2);

  sum = 0;
  for (int i = 0; i < 10000; i++) sum += (dptr1[i] - mean) * (dptr1[i] - mean);
  float variance = sum / 9999;
  EXPECT_NEAR(variance, 0.3 * 0.7, 1e-2);
}

TEST_F(TestTensorMath, MemberUniform) {
  Tensor p1(Shape{10000});
  Uniform(0.1f, 0.2f, &p1);
  const float *dptr1 = p1.data<float>();
  float sum = 0;
  for (int i = 0; i < 10000; i++) sum += dptr1[i];
  float mean = sum / 10000;
  EXPECT_NEAR(mean, 0.15f, 1e-3);

  sum = 0;
  for (int i = 0; i < 10000; i++) sum += (dptr1[i] - mean) * (dptr1[i] - mean);
  float variance = sum / 9999;
  EXPECT_NEAR(variance, 0.01f / 12, 1e-3);
}

TEST_F(TestTensorMath, MemberGaussian) {
  Tensor p1(Shape{50000});
  Gaussian(0.0f, 1.0f, &p1);
  const float *dptr1 = p1.data<float>();
  float sum = 0;
  for (int i = 0; i < 50000; i++) sum += dptr1[i];
  float mean = sum / 50000;
  EXPECT_NEAR(mean, 0.0, 1e-2);

  sum = 0;
  for (int i = 0; i < 50000; i++) sum += (dptr1[i] - mean) * (dptr1[i] - mean);
  float variance = sum / 49999;
  EXPECT_NEAR(variance, 1.0, 1e-2);
}

TEST_F(TestTensorMath, MemberAddTensor) {
  Tensor aa = a.Clone();
  aa += a;
  const float *dptr = aa.data<float>();
  EXPECT_FLOAT_EQ(2.0f, dptr[0]);
  EXPECT_FLOAT_EQ(4.0f, dptr[1]);
  EXPECT_FLOAT_EQ(6.0f, dptr[2]);

  // check p is initialized to 0
  Tensor p(Shape{6});
  p += aa;
  const float *dptr1 = p.data<float>();
  EXPECT_FLOAT_EQ(2.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(4.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(6.0f, dptr1[2]);

  a += b;
  const float *dptr2 = a.data<float>();
  EXPECT_FLOAT_EQ(2.1f, dptr2[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr2[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr2[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr2[5]);
}

TEST_F(TestTensorMath, AddTensors) {
  Tensor ret(a.shape(), a.device(), a.data_type());
  Add(a, b, &ret);
  const float *dptr = ret.data<float>();
  EXPECT_FLOAT_EQ(2.1f, dptr[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr[5]);

  const Tensor d = a + b;
  const float *dptr2 = d.data<float>();
  EXPECT_FLOAT_EQ(2.1f, dptr2[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr2[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr2[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr2[5]);

  Add(a, b, &a);
  const float *dptr1 = a.data<float>();
  EXPECT_FLOAT_EQ(2.1f, dptr1[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr1[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr1[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr1[5]);
}

TEST_F(TestTensorMath, SetValue) {
  Tensor t(Shape{4});
  t.SetValue(0.3f);
  const float *ptr = t.data<float>();
  for (int i = 0; i < 4; i++) EXPECT_FLOAT_EQ(ptr[i], 0.3f);
}

TEST_F(TestTensorMath, Reshape) {
  Tensor t(Shape{4});
  t.SetValue(0.3f);
  Tensor p = Reshape(t, Shape{4, 1});
  const float *ptr = t.data<float>();
  EXPECT_EQ(p.shape(0), 4u);
  EXPECT_EQ(p.shape(1), 1u);
  for (int i = 0; i < 4; i++) EXPECT_FLOAT_EQ(ptr[i], 0.3f);
}
#ifdef USE_CBLAS
TEST_F(TestTensorMath, L2Cpp) {
  float l2 = a.L2();
  float target = 0.0f;
  for (size_t i = 0; i < a.Size(); i++) target += dat1[i] * dat1[i];
  EXPECT_FLOAT_EQ(l2, sqrt(target) / a.Size());
}
TEST_F(TestTensorMath, MultCpp) {
  const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor t(Shape{2, 2});
  t.CopyDataFromHostPtr(x, 4);
  d.CopyDataFromHostPtr(dat1, 6);
  Tensor C = Mult(d, t);
  const float *xptr = C.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      float tmp = 0;
      for (int k = 0; k < 2; k++) {
        tmp += dat1[i * 2 + k] * x[k * 2 + j];
      }
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], tmp);
    }
  }
  const float y[8] = {1.0f, 2.0f, 3.0f, 4.0f, 1.1f, 2.1f, 3.1f, 4.1f};
  Tensor s(Shape{4, 2});
  s.CopyDataFromHostPtr(y, 8);
  const float *sPtr = s.data<float>();
  for (int i = 0; i < 8; i++) EXPECT_FLOAT_EQ(sPtr[i], y[i]);
  Tensor D = Mult(d, s.T());
  const float *DPtr = D.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      float tmp = 0;
      for (int k = 0; k < 2; k++) {
        tmp += dat1[i * 2 + k] * y[j * 2 + k];
      }
      EXPECT_FLOAT_EQ(DPtr[i * 4 + j], tmp);
    }
  }
  Tensor p(Shape{4, 1});
  p.CopyDataFromHostPtr(x, 4);
  Tensor q(Shape{1, 4});
  q.SetValue(1.0f);
  Tensor o(Shape{4, 4});

  Mult(p, q, &o);
  const float *oPtr = o.data<float>();
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_FLOAT_EQ(oPtr[i * 4 + j], x[i]);
    }
  }
}

TEST_F(TestTensorMath, AddColumnCpp) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  AddColumn(t, &d);
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] + x[i]);
    }
  }
}
TEST_F(TestTensorMath, SubColumnCpp) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  SubColumn(t, &d);
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] - x[i]);
    }
  }
}

TEST_F(TestTensorMath, DivColumnCpp) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  DivColumn(t, &d);
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] / x[i]);
    }
  }
}

TEST_F(TestTensorMath, AddRowCpp) {
  const float x[2] = {1.1f, 2.1f};
  Tensor t(Shape{2});
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  AddRow(t, &d);
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] + x[j]);
    }
  }
}

TEST_F(TestTensorMath, SubRowCpp) {
  const float x[2] = {1.1f, 2.1f};
  Tensor t(Shape{2});
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  SubRow(t, &d);
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] - x[j]);
    }
  }
}

TEST_F(TestTensorMath, MultRowCpp) {
  const float x[2] = {1.1f, 2.1f};
  Tensor t(Shape{2});
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  MultRow(t, &d);
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] * x[j]);
    }
  }
}

TEST_F(TestTensorMath, SumRowsCpp) {
  Tensor t(Shape{2});
  float dat[6];
  for (int i = 0; i < 6; i ++)
    dat[i] = (float)rand()/(float)(RAND_MAX/ 10);
  d.CopyDataFromHostPtr(dat, 6);
  SumRows(d, &t);
  const float *tptr = t.data<float>();
  for (int i = 0; i < 2; i++) {
    float tmp = 0;
    for (int j = 0; j < 3; j++) {
      tmp += dat[j * 2 + i];
    }
    EXPECT_FLOAT_EQ(tptr[i], tmp);
  }
}

TEST_F(TestTensorMath, SumColumnsCpp) {
  Tensor t(Shape{3});
  d.CopyDataFromHostPtr(dat1, 6);
  SumColumns(d, &t);
  const float *tptr = t.data<float>();
  for (int i = 0; i < 3; i++) {
    float tmp = 0;
    for (int j = 0; j < 2; j++) {
      tmp += dat1[i * 2 + j];
    }
    EXPECT_FLOAT_EQ(tptr[i], tmp);
  }
}
#endif
#ifdef USE_CUDA
TEST_F(TestTensorMath, L2Cuda) {
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{3, 2}, dev);
  t.CopyDataFromHostPtr(dat1, 6);
  float l2 = t.L2();
  float target = 0.0f;
  for (size_t i = 0; i < t.Size(); i++) target += dat1[i] * dat1[i];
  EXPECT_FLOAT_EQ(l2, sqrt(target) / t.Size());
}
TEST_F(TestTensorMath, MultCuda) {
  const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{2, 2}, dev);
  t.CopyDataFromHostPtr(x, 4);
  d.ToDevice(dev);
  d.CopyDataFromHostPtr(dat1, 6);
  Tensor C = Mult(d, t);
  C.ToHost();
  const float *xptr = C.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      float tmp = 0;
      for (int k = 0; k < 2; k++) {
        tmp += dat1[i * 2 + k] * x[k * 2 + j];
      }
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], tmp);
    }
  }

  const float y[8] = {1.0f, 2.0f, 3.0f, 4.0f, 1.1f, 2.1f, 3.1f, 4.1f};
  Tensor s(Shape{4, 2}, dev);
  s.CopyDataFromHostPtr(y, 8);
  Tensor D = Mult(d, s.T());
  D.ToHost();
  const float *DPtr = D.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      float tmp = 0;
      for (int k = 0; k < 2; k++) {
        tmp += dat1[i * 2 + k] * y[j * 2 + k];
      }
      EXPECT_FLOAT_EQ(DPtr[i * 4 + j], tmp);
    }
  }
  Tensor p(Shape{4, 1}, dev);
  p.CopyDataFromHostPtr(x, 4);
  Tensor q(Shape{1, 4}, dev);
  q.SetValue(1.0f);
  Tensor o(Shape{4, 4}, dev);

  Mult(p, q, &o);
  o.ToHost();
  const float *oPtr = o.data<float>();
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_FLOAT_EQ(oPtr[i * 4 + j], x[i]);
    }
  }
  d.ToHost();
  p.ToHost();
}

TEST_F(TestTensorMath, AddColumnCuda) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{3}, dev);
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  AddColumn(t, &d);
  d.ToHost();
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] + x[i]);
    }
  }
}

TEST_F(TestTensorMath, SubColumnCuda) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{3}, dev);
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  SubColumn(t, &d);
  d.ToHost();
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] - x[i]);
    }
  }
}
#endif
TEST_F(TestTensorMath, MultColumnCpp) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  MultColumn(t, &d);
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] * x[i]);
    }
  }
}
#ifdef USE_CUDA
TEST_F(TestTensorMath, MultColumnCuda) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{3}, dev);
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  MultColumn(t, &d);
  d.ToHost();
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] * x[i]);
    }
  }
}
TEST_F(TestTensorMath, DivColumnCuda) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{3}, dev);
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  DivColumn(t, &d);
  d.ToHost();
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] / x[i]);
    }
  }
}
TEST_F(TestTensorMath, AddRowCuda) {
  const float x[2] = {1.1f, 2.1f};
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{2}, dev);
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  AddRow(t, &d);
  d.ToHost();
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] + x[j]);
    }
  }
}
TEST_F(TestTensorMath, SubRowCuda) {
  const float x[2] = {1.1f, 2.1f};
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{2}, dev);
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  SubRow(t, &d);
  d.ToHost();
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] - x[j]);
    }
  }
}
TEST_F(TestTensorMath, MultRowCuda) {
  const float x[2] = {1.1f, 2.1f};
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{2}, dev);
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  MultRow(t, &d);
  d.ToHost();
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] * x[j]);
    }
  }
}
#endif
TEST_F(TestTensorMath, DivRowCpp) {
  const float x[2] = {1.1f, 2.1f};
  Tensor t(Shape{2});
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  DivRow(t, &d);
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] / x[j]);
    }
  }
}
#ifdef USE_CUDA
TEST_F(TestTensorMath, DivRowCuda) {
  const float x[2] = {1.1f, 2.1f};
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{2}, dev);
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  DivRow(t, &d);
  d.ToHost();
  const float *xptr = d.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] / x[j]);
    }
  }
}
TEST_F(TestTensorMath, SumRowsCuda) {
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{2}, dev);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  SumRows(d, &t);
  t.ToHost();
  const float *tptr = t.data<float>();
  for (int i = 0; i < 2; i++) {
    float tmp = 0;
    for (int j = 0; j < 3; j++) {
      tmp += dat1[j * 2 + i];
    }
    EXPECT_FLOAT_EQ(tptr[i], tmp);
  }
  d.ToHost();
}
TEST_F(TestTensorMath, SumColumnCuda) {
  auto dev = std::make_shared<singa::CudaGPU>();
  Tensor t(Shape{3}, dev);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(dev);
  SumColumns(d, &t);
  t.ToHost();
  const float *tptr = t.data<float>();
  for (int i = 0; i < 3; i++) {
    float tmp = 0;
    for (int j = 0; j < 2; j++) {
      tmp += dat1[i * 2 + j];
    }
    EXPECT_FLOAT_EQ(tptr[i], tmp);
  }
  d.ToHost();
}

#endif

TEST_F(TestTensorMath, ConcatenateRowsCpp) {
  d.CopyDataFromHostPtr<float>(dat1, 6);
  e.CopyDataFromHostPtr<float>(dat2, 6);
  const auto ret = singa::ConcatenateRows(vector<Tensor>{d, e});
  EXPECT_EQ(ret.shape(0), d.shape(0) + e.shape(0));
  EXPECT_EQ(ret.shape(1), d.shape(1));
  const float *retPtr = ret.data<float>();
  for (int i = 0; i < 6; i++) EXPECT_FLOAT_EQ(retPtr[i], dat1[i]);
  for (int i = 0; i < 6; i++) EXPECT_FLOAT_EQ(retPtr[i + 6], dat2[i]);
}

TEST_F(TestTensorMath, ConcatenateColumnsCpp) {
  d.CopyDataFromHostPtr<float>(dat1, 6);
  e.CopyDataFromHostPtr<float>(dat2, 6);
  const auto ret = singa::ConcatenateColumns(vector<Tensor>{d, e});
  EXPECT_EQ(ret.shape(0), d.shape(0));
  EXPECT_EQ(ret.shape(1), d.shape(1) + e.shape(1));

  const float *retPtr = ret.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(retPtr[i * 4 + j], dat1[i * 2 + j]);
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(retPtr[i * 4 + 2 + j], dat2[i * 2 + j]);
  }
}

TEST_F(TestTensorMath, CopyRowsCpp) {
  const auto ret = singa::CopyRows(e, 1, 2);
  EXPECT_EQ(ret.shape(0), 1u);
  EXPECT_EQ(ret.shape(1), e.shape(1));
  const float *retPtr = ret.data<float>();
  for (size_t i = 0; i < ret.Size(); i++)
    EXPECT_FLOAT_EQ(retPtr[i], dat1[1 * 2 + i]);
}

TEST_F(TestTensorMath, CopyColumnsCpp) {
  a.Reshape(Shape{2, 3});
  const auto ret = singa::CopyColumns(a, 1, 3);
  EXPECT_EQ(ret.shape(0), a.shape(0));
  EXPECT_EQ(ret.shape(1), 2u);
  const float *retPtr = ret.data<float>();
  for (size_t i = 0; i < ret.shape(0); i++)
    for (size_t j = 0; j < ret.shape(1); j++)
      EXPECT_FLOAT_EQ(retPtr[i * ret.shape(1) + j],
                      dat1[i * a.shape(1) + j + 1]);
}

#ifdef USE_CUDA

TEST_F(TestTensorMath, ConcatenateRowsCuda) {
  auto dev = std::make_shared<singa::CudaGPU>();
  d.ToDevice(dev);
  e.ToDevice(dev);
  d.CopyDataFromHostPtr<float>(dat1, 6);
  e.CopyDataFromHostPtr<float>(dat2, 6);
  auto ret = singa::ConcatenateRows(vector<Tensor>{d, e});
  EXPECT_EQ(ret.shape(0), d.shape(0) + e.shape(0));
  EXPECT_EQ(ret.shape(1), d.shape(1));
  ret.ToHost();
  const float *retPtr = ret.data<float>();
  for (int i = 0; i < 6; i++) EXPECT_FLOAT_EQ(retPtr[i], dat1[i]);
  for (int i = 0; i < 6; i++) EXPECT_FLOAT_EQ(retPtr[i + 6], dat2[i]);
}

TEST_F(TestTensorMath, ConcatenateColumnsCuda) {
  auto dev = std::make_shared<singa::CudaGPU>();
  d.ToDevice(dev);
  e.ToDevice(dev);
  d.CopyDataFromHostPtr<float>(dat1, 6);
  e.CopyDataFromHostPtr<float>(dat2, 6);
  auto ret = singa::ConcatenateColumns(vector<Tensor>{d, e});
  ret.ToHost();
  EXPECT_EQ(ret.shape(0), d.shape(0));
  EXPECT_EQ(ret.shape(1), d.shape(1) + e.shape(1));

  const float *retPtr = ret.data<float>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(retPtr[i * 4 + j], dat1[i * 2 + j]);
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(retPtr[i * 4 + 2 + j], dat2[i * 2 + j]);
  }
}

TEST_F(TestTensorMath, CopyRowsCuda) {
  auto dev = std::make_shared<singa::CudaGPU>();
  e.ToDevice(dev);
  auto ret = singa::CopyRows(e, 1, 2);
  ret.ToHost();
  EXPECT_EQ(ret.shape(0), 1u);
  EXPECT_EQ(ret.shape(1), e.shape(1));
  const float *retPtr = ret.data<float>();
  for (size_t i = 0; i < ret.Size(); i++)
    EXPECT_FLOAT_EQ(retPtr[i], dat1[1 * 2 + i]);
}

TEST_F(TestTensorMath, CopyColumnsCuda) {
  auto dev = std::make_shared<singa::CudaGPU>();
  a.Reshape(Shape{2, 3});
  a.ToDevice(dev);
  auto ret = singa::CopyColumns(a, 1, 3);
  EXPECT_EQ(ret.shape(0), a.shape(0));
  EXPECT_EQ(ret.shape(1), 2u);
  ret.ToHost();
  const float *retPtr = ret.data<float>();
  for (size_t i = 0; i < ret.shape(0); i++)
    for (size_t j = 0; j < ret.shape(1); j++)
      EXPECT_FLOAT_EQ(retPtr[i * ret.shape(1) + j],
                      dat1[i * a.shape(1) + j + 1]);
}

#endif
