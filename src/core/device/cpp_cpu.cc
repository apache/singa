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

#include "singa/core/device.h"

namespace singa {

std::shared_ptr<Device> defaultDevice = std::make_shared<CppCPU>();

CppCPU::CppCPU() : Device(-1, 1) {
  lang_ = kCpp;
#ifdef USE_DNNL
  ctx_.dnnl_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
  ctx_.dnnl_stream = dnnl::stream(ctx_.dnnl_engine);
#endif  // USE_DNNL
  // host_ = nullptr;
}

CppCPU::~CppCPU(){};

void CppCPU::SetRandSeed(unsigned seed) { ctx_.random_generator.seed(seed); }

void CppCPU::DoExec(function<void(Context*)>&& fn, int executor) {
  CHECK_EQ(executor, 0);
  fn(&ctx_);
}

void CppCPU::TimeProfilingDoExec(function<void(Context*)>&& fn, int executor,
                                 Node* node) {
  CHECK_EQ(executor, 0);

  auto t_start = std::chrono::high_resolution_clock::now();
  fn(&ctx_);
  std::chrono::duration<float> duration =
      std::chrono::high_resolution_clock::now() - t_start;
  node->time_elapsed_inc(duration.count());
}

void CppCPU::EvaluateTimeElapsed(Node* node) {}

void* CppCPU::Malloc(int size) {
  if (size > 0) {
    void* ptr = malloc(size);
    memset(ptr, 0, size);
    return ptr;
  } else {
    return nullptr;
  }
}

void CppCPU::Free(void* ptr) {
  if (ptr != nullptr) free(ptr);
}

void CppCPU::CopyToFrom(void* dst, const void* src, size_t nBytes,
                        CopyDirection direction, Context* ctx) {
  memcpy(dst, src, nBytes);
}

}  // namespace singa
