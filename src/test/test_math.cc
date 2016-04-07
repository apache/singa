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
#include <thread>
#include "gtest/gtest.h"
#include "singa/utils/blob.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/math_addr.h"
#include "singa/utils/math_kernel.h"
#include "singa/utils/singa_op.h"
#include "singa/utils/context.h"
#include "singa/utils/singleton.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

using namespace singa;
using namespace std;

TEST(MathBlobTest, TestScale) {
  Blob<float> *A = new Blob<float>(10);
  Blob<float> *B = new Blob<float>(10);
  A->SetValue(2);
  B->SetValue(6);
  Scale<float>(3.0, A);
  ASSERT_EQ(A->check_equal(B), true);
}

TEST(MathBlobTest, TestAXPY) {
  Blob<float> * A = new Blob<float>(10);
  Blob<float> * B = new Blob<float>(10);
  Blob<float> * C = new Blob<float>(10);
  Blob<float> * D = new Blob<float>(10);
  A->SetValue(2);
  B->SetValue(3);
  C->SetValue(7);
  D->SetValue(2);
  AXPY<float>(2.0, *A, B);
  ASSERT_EQ(B->check_equal(C), true);
  ASSERT_EQ(A->check_equal(D), true);
}

TEST(MathBlobTest, TestGEMV) {
  float A[5][5] = {};
  float AT[5][5] = {};
  float B[5] = {};
  float Res[5] = {};
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      A[i][j] = i * j + i - j;
      AT[j][i] = i * j + i - j;
    }
    B[i] = 5*i + 3;
    Res[i] = i;
  }

  Blob<float> * BlobA = new Blob<float>(5, 5);
  Blob<float> * BlobAT = new Blob<float>(5, 5);
  Blob<float> * BlobB = new Blob<float>(5);
  Blob<float> * BlobAB = new Blob<float>(5);
  Blob<float> * BlobATB = new Blob<float>(5);
  Blob<float> * BlobRes = new Blob<float>(5);

  BlobA->set_cpu_data(A[0]);
  BlobAT->set_cpu_data(AT[0]);
  BlobAT->set_transpose(true);
  BlobB->set_cpu_data(B);
  BlobAB->set_cpu_data(Res);
  BlobATB->set_cpu_data(Res);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      Res[i] += 2*A[i][j] * B[j];
    }
  }

  BlobRes->set_cpu_data(Res);

  GEMV<float>(2, 1, *BlobA, *BlobB, BlobAB);
  GEMV<float>(2, 1, *BlobAT, *BlobB, BlobATB);

  ASSERT_EQ(BlobAB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobATB->check_equal(BlobRes), true);
}

TEST(MathBlobTest, TestMVDot) {
  float A[5][5] = {};
  float AT[5][5] = {};
  float B[5] = {};
  float Res[5] = {};
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      A[i][j] = i * j + i - j;
      AT[j][i] = i * j + i - j;
    }
    B[i] = 5*i -2;
    Res[i] = 0;
  }

  Blob<float> * BlobA = new Blob<float>(5, 5);
  Blob<float> * BlobAT = new Blob<float>(5, 5);
  Blob<float> * BlobB = new Blob<float>(5);
  Blob<float> * BlobAB = new Blob<float>(5);
  Blob<float> * BlobATB = new Blob<float>(5);
  Blob<float> * BlobRes = new Blob<float>(5);

  BlobA->set_cpu_data(A[0]);
  BlobAT->set_cpu_data(AT[0]);
  BlobAT->set_transpose(true);
  BlobB->set_cpu_data(B);
  BlobAB->set_cpu_data(Res);
  BlobATB->set_cpu_data(Res);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      Res[i] += A[i][j] * B[j];
    }
  }

  BlobRes->set_cpu_data(Res);

  MVDot<float>(*BlobA, *BlobB, BlobAB);
  MVDot<float>(*BlobAT, *BlobB, BlobATB);

  const float * addrRes = BlobAB->cpu_data();
  for (int i = 0; i < 5; i++) {
    ASSERT_EQ(addrRes[i], Res[i]);
  }
  ASSERT_EQ(BlobAB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobAB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobATB->check_equal(BlobRes), true);
}

TEST(MathBlobTest, TestGEMM) {
  float A[5][5] = {};
  float AT[5][5] = {};
  float B[5][5]= {};
  float BT[5][5]= {};
  float Res[5][5]= {};
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      A[i][j] = i * j + i - j;
      AT[j][i] = i * j + i - j;
      B[i][j] = - i * j + i * i - j * j;
      BT[j][i] = - i * j + i * i - j * j;
      Res[i][j] = i * j + i * i + j * j;
    }
  }

  Blob<float> * BlobA = new Blob<float>(5, 5);
  BlobA->set_cpu_data(A[0]);
  Blob<float> * BlobAT = new Blob<float>(5, 5);
  BlobAT->set_cpu_data(AT[0]);
  BlobAT->set_transpose(true);
  Blob<float> * BlobB = new Blob<float>(5, 5);
  BlobB->set_cpu_data(B[0]);
  Blob<float> * BlobBT = new Blob<float>(5, 5);
  BlobBT->set_cpu_data(BT[0]);
  BlobBT->set_transpose(true);
  Blob<float> * BlobAB = new Blob<float>(5, 5);
  BlobAB->set_cpu_data(Res[0]);
  Blob<float> * BlobABT = new Blob<float>(5, 5);
  BlobABT->set_cpu_data(Res[0]);
  Blob<float> * BlobATB = new Blob<float>(5, 5);
  BlobATB->set_cpu_data(Res[0]);
  Blob<float> * BlobATBT = new Blob<float>(5, 5);
  BlobATBT->set_cpu_data(Res[0]);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      Res[i][j] *= 2;
      for (int k = 0; k < 5; k++) {
        Res[i][j] += 3 * A[i][k]*B[k][j];
      }
    }
  }

  Blob<float> * BlobRes = new Blob<float>(5, 5);
  BlobRes->set_cpu_data(Res[0]);

  GEMM<float>(3, 2, *BlobA, *BlobB, BlobAB);
  GEMM<float>(3, 2, *BlobA, *BlobBT, BlobABT);
  GEMM<float>(3, 2, *BlobAT, *BlobB, BlobATB);
  GEMM<float>(3, 2, *BlobAT, *BlobBT, BlobATBT);

  ASSERT_EQ(BlobAB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobATB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobABT->check_equal(BlobRes), true);
  ASSERT_EQ(BlobATBT->check_equal(BlobRes), true);
}

TEST(MathBlobTest, TestMMDot) {
  float A[5][5] = {};
  float AT[5][5] = {};
  float B[5][5]= {};
  float BT[5][5]= {};
  float Res[5][5]= {};
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      A[i][j] = i * j + i - j;
      AT[j][i] = i * j + i - j;
      B[i][j] = - i * j + i * i - j * j;
      BT[j][i] = - i * j + i * i - j * j;
      Res[i][j] = i * j + i * i + j * j;
    }
  }

  Blob<float> * BlobA = new Blob<float>(5, 5);
  BlobA->set_cpu_data(A[0]);
  Blob<float> * BlobAT = new Blob<float>(5, 5);
  BlobAT->set_cpu_data(AT[0]);
  BlobAT->set_transpose(true);
  Blob<float> * BlobB = new Blob<float>(5, 5);
  BlobB->set_cpu_data(B[0]);
  Blob<float> * BlobBT = new Blob<float>(5, 5);
  BlobBT->set_cpu_data(BT[0]);
  BlobBT->set_transpose(true);
  Blob<float> * BlobAB = new Blob<float>(5, 5);
  BlobAB->set_cpu_data(Res[0]);
  Blob<float> * BlobABT = new Blob<float>(5, 5);
  BlobABT->set_cpu_data(Res[0]);
  Blob<float> * BlobATB = new Blob<float>(5, 5);
  BlobATB->set_cpu_data(Res[0]);
  Blob<float> * BlobATBT = new Blob<float>(5, 5);
  BlobATBT->set_cpu_data(Res[0]);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      Res[i][j] = 0;
      for (int k = 0; k < 5; k++) {
        Res[i][j] += A[i][k]*B[k][j];
      }
    }
  }

  Blob<float> * BlobRes = new Blob<float>(5, 5);
  BlobRes->set_cpu_data(Res[0]);

  MMDot<float>(*BlobA, *BlobB, BlobAB);
  MMDot<float>(*BlobA, *BlobBT, BlobABT);
  MMDot<float>(*BlobAT, *BlobB, BlobATB);
  MMDot<float>(*BlobAT, *BlobBT, BlobATBT);

  ASSERT_EQ(BlobAB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobATB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobABT->check_equal(BlobRes), true);
  ASSERT_EQ(BlobATBT->check_equal(BlobRes), true);
}

TEST(MathBlobTest, TestVVDot) {
  float A[10] = {};
  float B[10] = {};
  float prod = 0;
  for (int i = 0; i < 10; i++) {
    A[i] = i * i - 5* (i%2);
    B[i] = 2* i * i - 3* (i%4);
    prod += A[i] * B[i];
  }

  Blob<float> * BlobA = new Blob<float>(10);
  BlobA->set_cpu_data(A);
  Blob<float> * BlobB = new Blob<float>(10);
  BlobB->set_cpu_data(B);
  float blobprod = VVDot<float>(*BlobA, *BlobB);
  ASSERT_EQ(blobprod, prod);
}

TEST(MathBlobTest, TestOuterProduct) {
  float A[10] = {};
  float B[10] = {};
  float AB[10][10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = i * i - 5* (i%2);
    B[i] = 2* i * i - 3* (i%4);
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      AB[i][j] = A[i]*B[j];
    }
  }
  Blob<float> * BlobA = new Blob<float>(10);
  BlobA->set_cpu_data(A);
  Blob<float> * BlobB = new Blob<float>(10);
  BlobB->set_cpu_data(B);
  Blob<float> * BlobAB = new Blob<float>(10, 10);
  // BlobAB->SetValue(3);
  Blob<float> * BlobRes = new Blob<float>(10, 10);
  BlobRes->set_cpu_data(AB[0]);
  OuterProduct<float>(*BlobA, *BlobB, BlobAB);

  ASSERT_EQ(BlobAB->check_equal(BlobRes), true);
}

TEST(MathBlobTest, TestMapAB) {
  float A[10] = {};
  float Res[10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = i * i - 5* (i%2);
    Res[i] = A[i] * A[i];
  }
  Blob<float> * BlobA = new Blob<float>(10);
  BlobA->set_cpu_data(A);
  Blob<float> * BlobB = new Blob<float>(10);
  Blob<float> * BlobRes = new Blob<float>(10);
  BlobRes->set_cpu_data(Res);
  Map<singa::op::Square<float>, float>(*BlobA, BlobB);
  ASSERT_EQ(BlobB->check_equal(BlobRes), true);
}

TEST(MathBlobTest, TestMapABC) {
  float A[10] = {};
  float B[10] = {};
  float Res[10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = i * i - 5* (i%2);
    B[i] = 2* i * i - 3* (i%4);
    Res[i] = A[i] * B[i];
  }
  Blob<float> * BlobA = new Blob<float>(10);
  BlobA->set_cpu_data(A);
  Blob<float> * BlobB = new Blob<float>(10);
  BlobB->set_cpu_data(B);
  Blob<float> * BlobC = new Blob<float>(10);
  Blob<float> * BlobRes = new Blob<float>(10);
  BlobRes->set_cpu_data(Res);
  Map<singa::op::Mult<float>, float>(*BlobA, *BlobB, BlobC);
  ASSERT_EQ(BlobC->check_equal(BlobRes), true);
}

TEST(MathBlobTest, TestCopy) {
  Blob<float> *BlobA = new Blob<float>(10);
  Blob<float> *BlobB = new Blob<float>(10);
  float A[10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = i * i - 5* (i%2);
  }
  BlobA->set_cpu_data(A);
  Copy<float>(*BlobA, BlobB);
  ASSERT_EQ(BlobA->check_equal(BlobB), true);
}

TEST(MathBlobTest, TestAdd) {
  Blob<float> *A = new Blob<float>(10);
  Blob<float> *B = new Blob<float>(10);
  Blob<float> *C = new Blob<float>(10);
  Blob<float> *D = new Blob<float>(10);
  A->SetValue(5);
  B->SetValue(6);
  D->SetValue(11);
  Add<float>(*A, *B, C);
  ASSERT_EQ(C->check_equal(D), true);
}

TEST(MathBlobTest, TestSub) {
  Blob<float> *A = new Blob<float>(10);
  Blob<float> *B = new Blob<float>(10);
  Blob<float> *C = new Blob<float>(10);
  Blob<float> *D = new Blob<float>(10);
  A->SetValue(5);
  B->SetValue(6);
  D->SetValue(-1);
  Sub<float>(*A, *B, C);
  ASSERT_EQ(C->check_equal(D), true);
}

TEST(MathBlobTest, TestMVAddCol) {
  Blob<float> *BlobA = new Blob<float>(10);
  Blob<float> *BlobB = new Blob<float>(10, 10);
  Blob<float> *BlobBT = new Blob<float>(10, 10);
  Blob<float> *BlobRes = new Blob<float>(10, 10);
  Blob<float> *BlobResT = new Blob<float>(10, 10);

  float A[10] = {};
  float B[10][10] = {};
  float BT[10][10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = 5*i -2;
    for (int j = 0; j < 10; j++) {
      B[i][j] = i * j + i - j;
      BT[j][i] = i * j + i - j;
    }
  }

  BlobA->set_cpu_data(A);
  BlobB->set_cpu_data(B[0]);
  BlobBT->set_cpu_data(BT[0]);
  BlobBT->set_transpose(true);

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      B[i][j] = 2.0 * A[i] + 3.0 * B[i][j];
      BT[j][i] = 2.0 * A[i] + 3.0 * BT[j][i];
    }
  }

  BlobRes->set_cpu_data(B[0]);
  BlobResT->set_cpu_data(BT[0]);
  BlobResT->set_transpose(true);

  MVAddCol<float>(2.0, 3.0, *BlobA, BlobB);
  MVAddCol<float>(2.0, 3.0, *BlobA, BlobBT);

  ASSERT_EQ(BlobB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobBT->check_equal(BlobResT), true);
}

TEST(MathBlobTest, TestMVAddRow) {
  Blob<float> *BlobA = new Blob<float>(10);
  Blob<float> *BlobB = new Blob<float>(10, 10);
  Blob<float> *BlobBT = new Blob<float>(10, 10);
  Blob<float> *BlobRes = new Blob<float>(10, 10);
  Blob<float> *BlobResT = new Blob<float>(10, 10);

  float A[10] = {};
  float B[10][10] = {};
  float BT[10][10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = 5*i -2;
    for (int j = 0; j < 10; j++) {
      B[i][j] = i * j + i - j;
      BT[j][i] = i * j + i - j;
    }
  }

  BlobA->set_cpu_data(A);
  BlobB->set_cpu_data(B[0]);
  BlobBT->set_cpu_data(BT[0]);
  BlobBT->set_transpose(true);

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      B[j][i] = 2.0 * A[i] + 3.0 * B[j][i];
      BT[i][j] = 2.0 * A[i] + 3.0 * BT[i][j];
    }
  }

  BlobRes->set_cpu_data(B[0]);
  BlobResT->set_cpu_data(BT[0]);
  BlobResT->set_transpose(true);

  MVAddRow<float>(2.0, 3.0, *BlobA, BlobB);
  MVAddRow<float>(2.0, 3.0, *BlobA, BlobBT);

  ASSERT_EQ(BlobB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobBT->check_equal(BlobResT), true);
}

TEST(MathBlobTest, TestRepmatCol) {
  Blob<float> *BlobA = new Blob<float>(10);
  Blob<float> *BlobB = new Blob<float>(10, 10);
  Blob<float> *BlobBT = new Blob<float>(10, 10);
  Blob<float> *BlobRes = new Blob<float>(10, 10);
  Blob<float> *BlobResT = new Blob<float>(10, 10);

  float A[10] = {};
  float B[10][10] = {};
  float BT[10][10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = 5*i -2;
    for (int j = 0; j < 10; j++) {
      B[i][j] = A[i];
      BT[j][i] = A[i];
    }
  }

  BlobA->set_cpu_data(A);
  BlobBT->set_transpose(true);

  BlobRes->set_cpu_data(B[0]);
  BlobResT->set_cpu_data(BT[0]);
  BlobResT->set_transpose(true);

  RepmatCol<float>(*BlobA, BlobB);
  RepmatCol<float>(*BlobA, BlobBT);

  ASSERT_EQ(BlobB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobBT->check_equal(BlobResT), true);
}

TEST(MathBlobTest, TestRepmatRow) {
  Blob<float> *BlobA = new Blob<float>(10);
  Blob<float> *BlobB = new Blob<float>(10, 10);
  Blob<float> *BlobBT = new Blob<float>(10, 10);
  Blob<float> *BlobRes = new Blob<float>(10, 10);
  Blob<float> *BlobResT = new Blob<float>(10, 10);

  float A[10] = {};
  float B[10][10] = {};
  float BT[10][10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = 5*i -2;
    for (int j = 0; j < 10; j++) {
      B[j][i] = A[i];
      BT[i][j] = A[i];
    }
  }

  BlobA->set_cpu_data(A);
  BlobBT->set_transpose(true);

  BlobRes->set_cpu_data(B[0]);
  BlobResT->set_cpu_data(BT[0]);
  BlobResT->set_transpose(true);

  RepmatRow<float>(*BlobA, BlobB);
  RepmatRow<float>(*BlobA, BlobBT);

  ASSERT_EQ(BlobB->check_equal(BlobRes), true);
  ASSERT_EQ(BlobBT->check_equal(BlobResT), true);
}

TEST(MathBlobTest, TestMVSumCol) {
  Blob<float> *BlobA = new Blob<float>(10);
  Blob<float> *BlobACopy = new Blob<float>(10);
  Blob<float> *BlobB = new Blob<float>(10, 10);
  Blob<float> *BlobBT = new Blob<float>(10, 10);
  Blob<float> *BlobRes = new Blob<float>(10);

  float A[10] = {};
  float B[10][10] = {};
  float BT[10][10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = 5*i -2;
    for (int j = 0; j < 10; j++) {
      B[i][j] = i * j + i - j;
      BT[j][i] = i * j + i - j;
    }
  }

  BlobA->set_cpu_data(A);
  BlobACopy->set_cpu_data(A);
  BlobB->set_cpu_data(B[0]);
  BlobBT->set_cpu_data(BT[0]);
  BlobBT->set_transpose(true);

  for (int i = 0; i < 10; i++) {
    A[i] *= 2.0;
    for (int j = 0; j < 10; j++) {
      A[i] += 3.0 * B[i][j];
    }
  }
  BlobRes->set_cpu_data(A);

  MVSumCol<float>(2.0, 3.0, *BlobB, BlobA);
  MVSumCol<float>(2.0, 3.0, *BlobBT, BlobACopy);

  ASSERT_EQ(BlobA->check_equal(BlobRes), true);
  ASSERT_EQ(BlobACopy->check_equal(BlobRes), true);
}

TEST(MathBlobTest, TestMVSumRow) {
  Blob<float> *BlobA = new Blob<float>(10);
  Blob<float> *BlobACopy = new Blob<float>(10);
  Blob<float> *BlobB = new Blob<float>(10, 10);
  Blob<float> *BlobBT = new Blob<float>(10, 10);
  Blob<float> *BlobRes = new Blob<float>(10);

  float A[10] = {};
  float B[10][10] = {};
  float BT[10][10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = 5*i -2;
    for (int j = 0; j < 10; j++) {
      B[j][i] = i * j + i - j;
      BT[i][j] = i * j + i - j;
    }
  }

  BlobA->set_cpu_data(A);
  BlobACopy->set_cpu_data(A);
  BlobB->set_cpu_data(B[0]);
  BlobBT->set_cpu_data(BT[0]);
  BlobBT->set_transpose(true);

  for (int i = 0; i < 10; i++) {
    A[i] *= 2.0;
    for (int j = 0; j < 10; j++) {
      A[i] += 3.0 * B[j][i];
    }
  }
  BlobRes->set_cpu_data(A);

  MVSumRow<float>(2.0, 3.0, *BlobB, BlobA);
  MVSumRow<float>(2.0, 3.0, *BlobBT, BlobACopy);

  ASSERT_EQ(BlobA->check_equal(BlobRes), true);
  ASSERT_EQ(BlobACopy->check_equal(BlobRes), true);
}

TEST(MathBlobTest, TestASum) {
  float A[10] = {};
  for (int i = 0; i < 10; i++) {
    A[i] = ((i % 3) -1) * i;
  }

  Blob<float> *BlobA = new Blob<float>(10);
  BlobA->set_cpu_data(A);

  float BlobRes = Asum<float>(*BlobA);
  float res = cblas_sasum(10, A, 1) / 10;

  ASSERT_EQ(BlobRes, res);
}

TEST(MathTest, TestGemmCPU) {
  float A[3][2] = {};
  float B[3][2] = {};
  float C[2][2] = {};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++) {
      A[i][j] = i+j;
      B[i][j] = i+j - i*j;
    }
  cpu_gemm(A[0], B[0], 2, 2, 3 , 1.0f, 0.0f, true, false, C[0]);
  float D[2][2] = {};
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      D[i][j] = 0;
      for (int k = 0; k < 3; k++)
        D[i][j] += A[k][i]*B[k][j];
    }
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) {
      ASSERT_EQ(C[i][j], D[i][j]);
    }
}

TEST(MathTest, TestGemvCPU) {
  float A[4][3] = {};
  float B[4]= {};
  float C[3] = {};
  float D[3] = {};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      A[j][i] = i-j + i*j;
    }
  }

  for (int i = 0; i < 4; i++)B[i] = i;
  for (int i = 0; i < 3; i++)C[i] = 10;
  cpu_gemv(A[0], B, 4, 3, 1.0f, 1.0f, true, C);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      D[i] += A[j][i]*B[j];
    }
  }
  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(C[i], D[i]+10);
  }
}


/*
TEST(MathTest, TestAxpyCPU) {
  float A[4][3] = {};
  float C[4][3] = {};
  float B[3][4] = {};
  float D[3][4] = {};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      A[i][j] = i-j + i*j;
      B[j][i] = i-j + i*j;
      C[i][j] = A[i][j];
      D[j][i] = B[j][i];
    }
  }

  cpu_axpy(A[0], 12, 2.0f, B[0]);
  for (int i = 0; i < 12; i++) {
    D[i / 4][i % 4] += 2*C[i / 3][i % 3];
  }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      ASSERT_EQ(B[i][j], D[i][j]);
    }
  }
}

TEST(MathTest, TestEopCPU) {

  float A[10] = {};
  float B[10] = {};
  float C[10] = {};
  float O[10] = {};

  for (int i = 0; i < 10; i++) {
    A[i] = i;
    B[i] = -i;
    C[i] = i;
  }
  cpu_e_f<singa::op::Set>(5, 15.0f, O, O);
  for (int i = 0; i < 5; i++) {
    ASSERT_EQ(O[i]-15,0);
  }
  for (int i = 5; i < 10; i++) {
    ASSERT_EQ(O[i],0);
  }
}
*/

#ifdef USE_GPU
TEST(MathTest, TestGemmGPU) {
  float A[3][2] = {};
  float B[3][2] = {};
  float C[2][2] = {};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      A[i][j] = i+j;
      B[i][j] = i+j - i*j;
    }
  }

  float* A_gpu = NULL;
  float* B_gpu = NULL;
  float* C_gpu = NULL;

  cudaMalloc(reinterpret_cast<void**>(&A_gpu), 3*2*sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&B_gpu), 3*2*sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&C_gpu), 2*2*sizeof(float));

  cudaMemcpy(A_gpu, A, 3*2*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, 3*2*sizeof(float), cudaMemcpyHostToDevice);
  auto context = Singleton<Context>::Instance();
  context->SetupDevice(std::this_thread::get_id(), 0);
  gpu_gemm<float>(context->cublas_handle(0), A_gpu, B_gpu, 2, 2, 3 , 1, 0, true,
                  false, C_gpu);

  cudaMemcpy(C, C_gpu, 2*2*sizeof(float), cudaMemcpyDeviceToHost);

  float D[2][2] = {};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      D[i][j] = 0;
      for (int k = 0; k < 3; k++) {
        D[i][j] += A[k][i]*B[k][j];
      }
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      ASSERT_EQ(C[i][j], D[i][j]);
    }
  }

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
}


TEST(MathTest, TestGemvGPU) {
  float A[4][3] = {};
  float B[4]= {};
  float C[3] = {};
  float D[3] = {};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      A[i][j] = i-j + i*j;
    }
  }

  for (int i = 0; i < 4; i++) B[i] = i;
  for (int i = 0; i < 3; i++) C[i] = 10;

  float* A_gpu = NULL;
  float* B_gpu = NULL;
  float* C_gpu = NULL;

  cudaMalloc(reinterpret_cast<void**>(&A_gpu), 4*3*sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&B_gpu), 4*sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&C_gpu), 3*sizeof(float));

  cudaMemcpy(A_gpu, A, 4*3*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, 4*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C, 3*sizeof(float), cudaMemcpyHostToDevice);
  auto context = Singleton<Context>::Instance();
  context->SetupDevice(std::this_thread::get_id(), 0);
  gpu_gemv<float>(context->cublas_handle(0), A_gpu, B_gpu, 4, 3, 1.0f, 1.0f,
                  true, C_gpu);

  cudaMemcpy(C, C_gpu, 3*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      D[i] += A[j][i]*B[j];
    }
  }

  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(C[i], D[i]+10);
  }

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
}


/*
TEST(MathTest, TestAxpyGPU) {
  float A[4][3] = {};
  float C[4][3] = {};
  float B[3][4] = {};
  float D[3][4] = {};

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      A[i][j] = i-j + i*j;
      B[j][i] = i-j + i*j;
      C[i][j] = A[i][j];
      D[j][i] = B[j][i];
    }
  }

  float* A_gpu=NULL;
  float* B_gpu=NULL;

  cudaMalloc((void**)&A_gpu, 4*3*sizeof(float));
  cudaMalloc((void**)&B_gpu, 3*4*sizeof(float));

  cudaMemcpy(A_gpu,A,4*3*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu,B,3*4*sizeof(float),cudaMemcpyHostToDevice);

  gpu_axpy<float>(A_gpu, 12, 2, B_gpu);

  cudaMemcpy(A,A_gpu,4*3*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(B,B_gpu,3*4*sizeof(float),cudaMemcpyDeviceToHost);

  //for (int i = 0; i < 12; i++)D[0][i] += 2*C[0][i];

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      D[i][j] += C[i][j];
      ASSERT_EQ(B[i][j],D[i][j]);
    }
  }

  cudaFree(A_gpu);
  cudaFree(B_gpu);
}
*/


TEST(MathTest, TestDotGPU) {
  float A[12];
  float B[12];
  for (int i = 0; i < 12; i++) {
    A[i] = i - 1;
    B[i] = i + 1;
  }

  float* A_gpu = NULL;
  float* B_gpu = NULL;

  cudaMalloc(reinterpret_cast<void**>(&A_gpu), 12*sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&B_gpu), 12*sizeof(float));

  cudaMemcpy(A_gpu, A, 12*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, 12*sizeof(float), cudaMemcpyHostToDevice);
  auto context = Singleton<Context>::Instance();
  context->SetupDevice(std::this_thread::get_id(), 0);
  float gpu_ret = gpu_dot<float>(context->cublas_handle(0), 12, A_gpu, B_gpu);

  float cpu_ret = 0.0f;
  for (int i = 0; i < 12; i++) {
    cpu_ret += A[i] * B[i];
  }

  ASSERT_EQ(gpu_ret, cpu_ret);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
}

TEST(MathTest, TestSingaSumRowGPU) {
  float A[3][4];
  float B[4];
  float C[4];

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      // A[i][j] = i + j;
      A[i][j] = 1.0f;
    }
  }

  for (int i = 0; i < 4; i++) {
    B[i] = 0.0f;
    C[i] = 0.0f;
  }

  float* A_gpu = NULL;
  float* B_gpu = NULL;

  cudaMalloc(reinterpret_cast<void**>(&A_gpu), 12*sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&B_gpu), 4*sizeof(float));
  cudaMemcpy(A_gpu, A, 12*sizeof(float), cudaMemcpyHostToDevice);
  singa_gpu_sum_row(A_gpu, B_gpu, 3, 4, 4);

  cudaMemcpy(B, B_gpu, 4*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      C[i] += A[j][i];
    }
  }

  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(B[i], C[i]);
  }

  cudaFree(A_gpu);
  cudaFree(B_gpu);
}

TEST(MathTest, TestSingaAddVecRowGPU) {
  float A[3][4];
  float B[4];
  float C[3][4];
  float D[3][4];

  for (int i = 0; i < 4; i++) {
    B[i] = i;
  }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      A[i][j] = i + j;
      D[i][j] = A[i][j] + B[j];
    }
  }

  float* A_gpu = NULL;
  float* B_gpu = NULL;
  float* C_gpu = NULL;

  cudaMalloc(reinterpret_cast<void**>(&A_gpu), 3*4*sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&B_gpu), 4*sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&C_gpu), 3*4*sizeof(float));
  cudaMemcpy(A_gpu, A, 3*4*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, 4*sizeof(float), cudaMemcpyHostToDevice);

  singa_gpu_add_vec_row(B_gpu, A_gpu, C_gpu, 3, 4, 4);

  cudaMemcpy(C, C_gpu, 3*4*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      ASSERT_EQ(C[i][j], D[i][j]);
    }
  }

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
}


TEST(MathTest, TestSingaSetValueGPU) {
  float A[3][4];
  float* A_gpu = NULL;

  cudaMalloc(reinterpret_cast<void**>(&A_gpu), 3*4*sizeof(float));

  cudaMemcpy(A_gpu, A, 3*4*sizeof(float), cudaMemcpyHostToDevice);

  singa_gpu_set_value(A_gpu, 4.0, 3*4);

  cudaMemcpy(A, A_gpu, 3*4*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      ASSERT_EQ(A[i][j], 4.0f);
    }
  }

  cudaFree(A_gpu);
}


TEST(MathTest, TestEopGPU) {
  float A[10] = {};
  float B[10] = {};

  for (int i = 0; i < 10; i++) {
    A[i] = i;
    B[i] = -i;
  }

  float* A_gpu = NULL;
  float* B_gpu = NULL;

  cudaMalloc(reinterpret_cast<void**>(&A_gpu), 10*sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&B_gpu), 10*sizeof(float));

  cudaMemcpy(A_gpu, A, 10*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, 10*sizeof(float), cudaMemcpyHostToDevice);

  gpu_e_f<singa::op::Sigmoid<float>, float>(10, A_gpu, B_gpu);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
}
#endif  // USE_GPU
