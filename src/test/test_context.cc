#include "gtest/gtest.h"
#include "singa/utils/singleton.h"
#include "singa/utils/context.h"

//#include <cuda_runtime.h>
//#include "cublas_v2.h"

using namespace singa;
using namespace std;

TEST(ContextTest, TestDevice) {
  auto context = Singleton<Context>::Instance();
  context->Setup();

  int index = 4;
  int device_id = context->DeviceID(index);
  ASSERT_EQ(4,device_id);

  context->SetDeviceID(index,6);
  device_id = context->DeviceID(index);
  ASSERT_EQ(6,device_id);
}

TEST(ContextTest, TestHandle) {
  auto context = Singleton<Context>::Instance();
  context->Setup();

  int index = 2;
  context->CreateHandle(index);

  float cpu_ret = 0.0f;
  float gpu_ret = 0.0f;

  float A[12];
  float B[12];
  
  for(int i = 0; i < 12; i++) {
	A[i]=i-1;
	B[i]=i+1;
  }

  float* A_gpu = NULL;
  float* B_gpu = NULL;
  
  cudaMalloc((void**)&A_gpu, 12*sizeof(float));
  cudaMalloc((void**)&B_gpu, 12*sizeof(float));

  cudaMemcpy(A_gpu,A,12*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu,B,12*sizeof(float),cudaMemcpyHostToDevice);

  cublasHandle_t handle = context->Handle(index);
  /*cublasHandle_t handle;
  cudaSetDevice(0);
  cublasCreate(&handle);*/

  cublasSdot(handle, 12, A_gpu, 1, B_gpu, 1, &gpu_ret);

  for(int i = 0; i < 12;++i) {
	cpu_ret += A[i] * B[i];
  }
  
  ASSERT_EQ(gpu_ret,cpu_ret);
  
  cudaFree(A_gpu);
  cudaFree(B_gpu);
}
