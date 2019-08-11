/*********************************************************
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
************************************************************/

#ifdef USE_TC
#include "../src/model/operation/tc_fn.h"
#include "gtest/gtest.h"
#include <iostream>
#include <chrono>

using namespace singa;
TEST(OperationTCFn, SoftmaxForward) {
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor t1(singa::Shape{1,2}, cuda);

  const float dat1[2] = {1.0f, 2.0f};
  t1.CopyDataFromHostPtr<float>(dat1, 2);

  std::string tc = R"TC(
def softmax(float(N, D) I) -> (O, expsum, maxVal) {
    maxVal(n) max=!     I(n, d)
    expsum(n)   +=! exp(I(n, d) - maxVal(n))
         O(n, d) =  exp(I(n, d) - maxVal(n)) / expsum(n)
}
)TC";
  TcFnHandle tfh(tc, "softmax", {t1});

  std::chrono::steady_clock::time_point beginTC = std::chrono::steady_clock::now();
  Tensor output = tcExecute(tfh, {t1});
  std::chrono::steady_clock::time_point endTC = std::chrono::steady_clock::now();
  std::cout << "\nTime " << std::chrono::duration_cast<std::chrono::microseconds>(endTC - beginTC).count() << "[microseconds]" << std::endl;

  output.ToHost();

  EXPECT_EQ(output.shape(0), 1);
  EXPECT_EQ(output.shape(1), 2);
  const float *dptr1 = output.data<float>();
  EXPECT_NEAR(0.26894142f, dptr1[0], 1e-5);
  EXPECT_NEAR(0.73105858f, dptr1[1], 1e-5);
}

TEST(OperationTCFn, SoftmaxBackward) {
  const float x[] = {1.f, 2.f, 0.f, -2.f, -3.f, -1.f};
  const float grad[] = {2.0f, -3.0f, 1.0f, 3.0f, -1.0f, -2.0f};

  size_t n = sizeof(x) / sizeof(float);
  size_t batch = 2, c = 3;

  singa::Shape shape = {batch, c};
  auto cuda = std::make_shared<singa::CudaGPU>();

  singa::Tensor in(shape, cuda);
  in.CopyDataFromHostPtr<float>(x, n);
  singa::Tensor output_grad(shape, cuda);
  output_grad.CopyDataFromHostPtr<float>(grad, n);


  std::string tc_forward_def = R"TC(
def softmax(float(N, D) I) -> (O, expsum, maxVal) {
    maxVal(n) max=!     I(n, d)
    expsum(n)   +=! exp(I(n, d) - maxVal(n))
         O(n, d) =  exp(I(n, d) - maxVal(n)) / expsum(n)
}
)TC";
  TcFnHandle tfh_forward(tc_forward_def, "softmax", {in});
  Tensor output = tcExecute(tfh_forward, {in});


  std::string tc = R"TC(
def softmax_bwd(float(N, D) output, float(N, D) grad_output) -> (grad_input, sigma)
{
  sigma(n) +=! output(n, d) * grad_output(n ,d)
  grad_input(n, d) = ( grad_output(n, d) - sigma(n) ) * output(n, d)
}
)TC";
  TcFnHandle tfh(tc, "softmax_bwd", {output, output_grad});


  std::chrono::steady_clock::time_point beginTC = std::chrono::steady_clock::now();
  Tensor in_grad = tcExecute(tfh, {output, output_grad});
  std::chrono::steady_clock::time_point endTC = std::chrono::steady_clock::now();
  std::cout << "\nTime " << std::chrono::duration_cast<std::chrono::microseconds>(endTC - beginTC).count() << "[microseconds]" << std::endl;


  in_grad.ToHost();
  const float *xptr = in_grad.data<float>();

  output.ToHost();
  const float* yptr = output.data<float>();

  float* dx = new float[n];
  float* sigma = new float[batch];
  for (size_t i = 0; i < batch; i++)
    sigma[i] = 0.f;
  for (size_t i = 0; i < n; i++)
    sigma[i / c] += grad[i] * yptr[i];
  for (size_t i = 0; i < batch; i++)
    for (size_t j = 0; j < c; j++)
      dx[i * c + j] = (grad[i * c + j] - sigma[i]) * yptr[i * c +j];
  EXPECT_FLOAT_EQ(dx[0], xptr[0]);
  EXPECT_FLOAT_EQ(dx[1], xptr[1]);
  EXPECT_FLOAT_EQ(dx[2], xptr[2]);
  EXPECT_FLOAT_EQ(dx[3], xptr[3]);
  EXPECT_FLOAT_EQ(dx[4], xptr[4]);
  EXPECT_FLOAT_EQ(dx[5], xptr[5]);
}

TEST(OperationTCFn, ReLU) {
  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor t1(singa::Shape{2, 2}, cuda);

  const float dat1[4] = {-1.0f, 1.0f, -2.0f, 3.0f};
  t1.CopyDataFromHostPtr<float>(dat1, 4);


  std::string tc = R"TC(
def relu(float(B,M) I) -> (O1){
  O1(b, m) = fmax(I(b, m), 0)
}
  )TC";
  TcFnHandle tfh(tc, "relu", {t1});

  std::chrono::steady_clock::time_point beginTC = std::chrono::steady_clock::now();
  Tensor o1 = tcExecute(tfh, {t1});
  std::chrono::steady_clock::time_point endTC = std::chrono::steady_clock::now();
  std::cout << "\nTime " << std::chrono::duration_cast<std::chrono::microseconds>(endTC - beginTC).count() << "[microseconds]" << std::endl;
  o1.ToHost();

  EXPECT_EQ(o1.shape(0), 2);
  EXPECT_EQ(o1.shape(1), 2);
  const float *dptr = o1.data<float>();
  EXPECT_FLOAT_EQ(0.0f, dptr[0]);
  EXPECT_FLOAT_EQ(1.0f, dptr[1]);
  EXPECT_FLOAT_EQ(0.0f, dptr[2]);
  EXPECT_FLOAT_EQ(3.0f, dptr[3]);
}

TEST(OperationTCFn, FC) {
  std::string tc = R"TC(
def fc(float(B,M) I, float(N,M) W1, float(N) B1) -> (O1) {
  O1(b, n) +=! I(b, m) * W1(n, m)
  O1(b, n) = O1(b, n) + B1(n)
}
  )TC";

  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor x(singa::Shape{2, 3}, cuda);
  singa::Tensor W(singa::Shape{4, 3}, cuda);
  singa::Tensor b(singa::Shape{4}, cuda);
  x.SetValue(1.1f);
  W.SetValue(1.2f);
  b.SetValue(1.3f);


  TcFnHandle tfh(tc, "fc", {x,W,b});

  std::chrono::steady_clock::time_point beginTC = std::chrono::steady_clock::now();
  Tensor o1 = tcExecute(tfh, {x,W,b});
  std::chrono::steady_clock::time_point endTC = std::chrono::steady_clock::now();
  std::cout << "\nTime " << std::chrono::duration_cast<std::chrono::microseconds>(endTC - beginTC).count() << "[microseconds]" << std::endl;
  o1.ToHost();

  EXPECT_EQ(o1.shape(0), 2);
  EXPECT_EQ(o1.shape(1), 4);
  const float *dptr = o1.data<float>();
  EXPECT_FLOAT_EQ(5.26f, dptr[0]);
  EXPECT_FLOAT_EQ(5.26f, dptr[1]);
  EXPECT_FLOAT_EQ(5.26f, dptr[2]);
  EXPECT_FLOAT_EQ(5.26f, dptr[3]);
  EXPECT_FLOAT_EQ(5.26f, dptr[4]);
  EXPECT_FLOAT_EQ(5.26f, dptr[5]);
  EXPECT_FLOAT_EQ(5.26f, dptr[6]);
  EXPECT_FLOAT_EQ(5.26f, dptr[7]);
}

TEST(OperationTCFn, MatMul) {
  std::string tc = R"TC(
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
  )TC";

  auto cuda = std::make_shared<singa::CudaGPU>();
  singa::Tensor t1(singa::Shape{2, 2}, cuda);
  singa::Tensor t2(singa::Shape{2, 2}, cuda);
  t1.SetValue(1.1f);
  t2.SetValue(1.2f);

  TcFnHandle tfh(tc, "matmul", {t1,t2});

  std::chrono::steady_clock::time_point beginTC = std::chrono::steady_clock::now();
  Tensor o1 = tcExecute(tfh, {t1,t2});
  std::chrono::steady_clock::time_point endTC = std::chrono::steady_clock::now();
  std::cout << "\nTime " << std::chrono::duration_cast<std::chrono::microseconds>(endTC - beginTC).count() << "[microseconds]" << std::endl;
  o1.ToHost();

  EXPECT_EQ(o1.shape(0), 2);
  EXPECT_EQ(o1.shape(1), 2);
  const float *dptr = o1.data<float>();
  EXPECT_FLOAT_EQ(2.64f, dptr[0]);
  EXPECT_FLOAT_EQ(2.64f, dptr[1]);
  EXPECT_FLOAT_EQ(2.64f, dptr[2]);
  EXPECT_FLOAT_EQ(2.64f, dptr[3]);

}
#endif // USE_TC
