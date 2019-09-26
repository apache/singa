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

#include <iostream>
#include <cstdint>
#include <unistd.h>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include "singa/core/tensor.h"

namespace singa{

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

class NcclIdHolder {
public:
  ncclUniqueId id;
  NcclIdHolder(); 
  ~NcclIdHolder();
};

class Communicator {
public:
  int MPIRankInGlobal;
  int totalMPIRanksInGlobal;
  int MPIRankInLocal;
  bool UseMPI;

  ncclUniqueId id;
  cudaStream_t s;
  ncclComm_t comm;

  Communicator();
  Communicator(int gpu_num, int gpu_per_node, const NcclIdHolder &holder);
  ~Communicator();
  void allReduce(int size, void* sendbuff, void* recvbuff);
  void wait();
};

void synch(Tensor &t, Communicator &c);

}

#endif // USE_DIST
#endif
