#include <thread>
#include "gtest/gtest.h"
#include "singa/utils/singleton.h"
#include "singa/utils/context.h"
#include "singa/utils/cuda_utils.h"

using namespace singa;
using namespace std;

TEST(ContextTest, TestDevice) {
  auto context = Singleton<Context>::Instance();

  auto id = std::this_thread::get_id();
  context->SetupDevice(id, 0);
  auto device_id = context->device_id(id);
  ASSERT_EQ(1,device_id);
}

TEST(ContextTest, TestHandle) {
  auto context = Singleton<Context>::Instance();

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
  context->SetupDevice(std::this_thread::get_id(), 0);

  cudaMalloc((void**)&A_gpu, 12 * sizeof(float));
  cudaMalloc((void**)&B_gpu, 12 * sizeof(float));

  cudaMemcpy(A_gpu, A, 12 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, 12 * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle = context->cublas_handle(std::this_thread::get_id());

  cublasSdot(handle, 12, A_gpu, 1, B_gpu, 1, &gpu_ret);

  for(int i = 0; i < 12;++i) {
    cpu_ret += A[i] * B[i];
  }

  ASSERT_EQ(gpu_ret,cpu_ret);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
}
