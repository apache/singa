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

#ifndef SINGA_DIST_COMMUNICATOR_H_
#define SINGA_DIST_COMMUNICATOR_H_

#ifdef USE_DIST

#include <cuda_runtime.h>
#include <cusparse.h>
#include <mpi.h>
#include <nccl.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <memory>

#include "cuda_fp16.h"
#include "singa/core/tensor.h"
using std::vector;

namespace singa {

#define CUSPARSE_CHECK(cmd)                                             \
  do {                                                                  \
    cusparseStatus_t e = cmd;                                           \
    if (e != CUSPARSE_STATUS_SUCCESS) {                                 \
      printf("Falied: Cusparse Error %s:%d '%d'\n", __FILE__, __LINE__, \
             int(e));                                                   \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

class NcclIdHolder {
 public:
  ncclUniqueId id;
  NcclIdHolder();
  ~NcclIdHolder();
};

class Communicator {
 public:
  int global_rank;
  int world_size;
  int local_rank;

  Communicator(int limit);
  Communicator(int local_rank, int world_size, const NcclIdHolder &holder,
               int size);
  ~Communicator();
  void synch(Tensor &t);
  void fusedSynch(vector<Tensor> &t, bool send = true);
  void synchHalf(Tensor &t);
  void fusedSynchHalf(vector<Tensor> &t, bool send = true);
  void fusedSparsification(vector<Tensor> &t, Tensor &accumulation,
                           float sparsThreshold, bool topK);
  void fusedSparsification(vector<Tensor> &t, float sparsThreshold, bool topK);
  void sparsification(Tensor &t, Tensor &accumulation, float sparsThreshold,
                      bool topK);
  void sparsification(Tensor &t, float sparsThreshold, bool topK);
  void wait();

 private:
  void generateBlocks(Tensor &t);
  void generateBlocks(std::vector<Tensor> &t);
  void allReduce(int size, void *sendbuff, void *recvbuff,
                 ncclDataType_t ncclType, Context *ctx);
  void setup();
  void sparsInit();
  void halfInit();
  void _fusedSparsification(vector<Tensor> &t, Tensor *accumulation,
                            float sparsThreshold, bool topK, Context *ctx);
  void _sparsification(Tensor &t, Tensor *accumulation, float sparsThreshold,
                       bool topK, Context *ctx);
  void valSparsAllReduce(size_t num, void *accumulation, Context *ctx);
  void topKSparsAllReduce(size_t num, void *accumulation, Context *ctx);

  // last group of synchronized memory blocks
  std::shared_ptr<Device> device_ = nullptr;
  std::vector<Block *> blocks_;
  std::vector<Block *> prev_blocks_;

  ncclUniqueId id;
  ncclComm_t comm;
  cudaEvent_t event;

  bool UseMPI;
  size_t maxSize;

  // normal synch
  size_t sendBuffOffset = 0;
  void *fusedSendBuff;
  void *fusedRecvBuff;
  void *offsetPointer;
  size_t dataSize;
  ncclDataType_t ncclType;

  // half synch
  bool halfInitialized;
  void *fusedSendBuffHalf;
  void *fusedRecvBuffHalf;

  // sparsification
  cusparseHandle_t cusparse_handle;
  cusparseMatDescr_t descrC;
  bool sparsInitialized;
  int *xInd;
  float *xVal;
  int *nnz;
  int *nnzAll;
  int *nnzGPU;
  int *nnzAllGPU;
  float threshold;
  void *sparsSendBuff;
  void *sparsRecvBuff;
  void *backupBuff;
  int *fusedIndex;
};
}  // namespace singa

#endif  // USE_DIST
#endif
