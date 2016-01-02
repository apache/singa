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
#include "singa/utils/math_addr.h"
#include "singa/utils/math_kernel.h"
#include "singa/utils/singa_op.h"
#include "singa/utils/context.h"
#include "singa/utils/singleton.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif

using namespace singa;
using namespace std;

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
  float gpu_ret = gpu_dot<float>(context->cublas_handle(0), A_gpu, B_gpu, 12);

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
