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

#include <iostream>

#include "singa/utils/cuda_utils.h"

#ifdef USE_DIST

#include "../core/tensor/math_kernel.h"
#include "singa/io/communicator.h"

namespace singa {

static uint64_t getHostHash(const char *string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char *hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

NcclIdHolder::NcclIdHolder() { ncclGetUniqueId(&id); }  // end of constructor

NcclIdHolder::~NcclIdHolder() {}

// contructer for application with python multi-processing module
Communicator::Communicator(int local_rank, int world_size,
                           const NcclIdHolder &holder, int buffSize) {
  maxSize = (size_t)buffSize;
  // this contructor is for NCCL WITHOUT MPI
  UseMPI = false;
  // Determine the rank of the collective communication
  this->world_size = world_size;
  this->local_rank = local_rank;
  this->global_rank = local_rank;

  // copy the nccl unqiue id from the input id holder
  id = holder.id;

  // setup cuda stream and nccl communicator
  setup();

}  // end of constructor

// contructer for application with MPI
Communicator::Communicator(int buffSize) {
  maxSize = (size_t)buffSize;
  // this contructor is for NCCL WITH MPI
  UseMPI = true;

  // MPI initialization
  MPICHECK(MPI_Init(NULL, NULL));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &global_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

  // calculating local_rank which is used in selecting a GPU
  local_rank = 0;
  uint64_t hostHashs[world_size];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[global_rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                         sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < world_size; p++) {
    if (p == global_rank) break;
    if (hostHashs[p] == hostHashs[global_rank]) local_rank++;
  }

  // generating NCCL unique nccl ID at one process and broadcasting it to all
  if (global_rank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // setup cuda stream and nccl communicator
  setup();

}  // end of constructor

void Communicator::setup() {
  CUDA_CHECK(cudaSetDevice(local_rank));
  NCCLCHECK(ncclCommInitRank(&comm, world_size, id, global_rank));
  CUDA_CHECK(cudaMalloc(&fusedSendBuff, maxSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fusedRecvBuff, maxSize * sizeof(float)));
  CUDA_CHECK(cudaEventCreateWithFlags(
      &event, cudaEventBlockingSync | cudaEventDisableTiming));
  halfInitialized = false;
  sparsInitialized = false;
}

void Communicator::halfInit() {
  // initialze the buffer
  CUDA_CHECK(cudaMalloc(&fusedSendBuffHalf, maxSize * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&fusedRecvBuffHalf, maxSize * sizeof(__half)));
  halfInitialized = true;
}

void Communicator::sparsInit() {
  // initize sparsification environment
  CUDA_CHECK(cudaSetDevice(local_rank));
  CUDA_CHECK(
      cudaMalloc(&sparsRecvBuff, (int)(maxSize * sizeof(float) * world_size)));
  CUDA_CHECK(cudaMalloc(&sparsSendBuff, (int)(maxSize * sizeof(float))));
  CUDA_CHECK(cudaMalloc(&backupBuff, maxSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fusedIndex, maxSize * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&xInd, (int)(sizeof(int) * maxSize)));
  CUDA_CHECK(cudaMalloc(&xVal, (int)(sizeof(float) * maxSize)));
  CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
  nnz = (int *)malloc(sizeof(int));
  nnzAll = (int *)malloc(sizeof(int) * world_size);
  CUDA_CHECK(cudaMalloc(&nnzGPU, sizeof(int) * world_size));
  CUDA_CHECK(cudaMalloc(&nnzAllGPU, sizeof(int) * world_size));
  sparsInitialized = true;
}

void Communicator::allReduce(int size, void *sendbuff, void *recvbuff,
                             ncclDataType_t ncclType, Context *ctx) {
  NCCLCHECK(ncclAllReduce((const void *)sendbuff, (void *)recvbuff, size,
                          ncclType, ncclSum, comm, ctx->s));
}

void Communicator::generateBlocks(Tensor &t) {
  device_ = t.device();

  blocks_.clear();
  blocks_.push_back(t.block());
}

void Communicator::generateBlocks(std::vector<Tensor> &t) {
  device_ = t[0].device();

  prev_blocks_ = blocks_;

  blocks_.clear();
  blocks_.reserve(t.size());
  prev_blocks_.reserve(prev_blocks_.size() + t.size());

  for (size_t i = 0; i < t.size(); ++i) {
    blocks_.push_back(t[i].block());
    prev_blocks_.push_back(t[i].block());
  }
}

void Communicator::wait() {
  if (!device_) {
    // just return if it has not been synchronized
    return;
  }

  device_->Exec(
      [this](Context *ctx) mutable {
        // synchronizing on all the CUDA streams used by communicator
        CUDA_CHECK(cudaEventRecord(event, ctx->s));
        CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, event, 0));
        CUDA_CHECK(cudaEventRecord(event, ctx->c1));
        CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, event, 0));
        CUDA_CHECK(cudaEventRecord(event, ctx->c2));
        CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, event, 0));
      },
      blocks_, blocks_, "Waiting");
}

Communicator::~Communicator() {
  // finalizing NCCL
  ncclCommDestroy(comm);
  if (UseMPI == true) MPICHECK(MPI_Finalize());
  CUDA_CHECK(cudaFree(fusedSendBuff));
  CUDA_CHECK(cudaFree(fusedRecvBuff));

  if (halfInitialized == true) {
    CUDA_CHECK(cudaFree(fusedSendBuffHalf));
    CUDA_CHECK(cudaFree(fusedRecvBuffHalf));
  }

  if (sparsInitialized == true) {
    CUDA_CHECK(cudaFree(sparsRecvBuff));
    CUDA_CHECK(cudaFree(sparsSendBuff));
    CUDA_CHECK(cudaFree(backupBuff));
    CUDA_CHECK(cudaFree(fusedIndex));
    CUDA_CHECK(cudaFree(xInd));
    CUDA_CHECK(cudaFree(xVal));
    CUDA_CHECK(cudaFree(nnzGPU));
    CUDA_CHECK(cudaFree(nnzAllGPU));
  }
}

void Communicator::fusedSynch(vector<Tensor> &t, bool send) {
  CHECK_GT(t.size(), 0);

  generateBlocks(t);

  if (!send) {
    // buffer the tensors
    device_->Exec(
        [this, t](Context *ctx) mutable {
          // record the event of the default cuda stream and follow it
          CUDA_CHECK(cudaEventRecord(event, ctx->stream));
          CUDA_CHECK(cudaStreamWaitEvent(ctx->c1, event, 0));
        },
        prev_blocks_, prev_blocks_, "Waiting");

    device_->Exec(
        [this, t](Context *ctx) mutable {
          // memory copy to fusedBuff
          for (size_t i = 0; i < t.size(); i++) {
            CUDA_CHECK(
                cudaMemcpyAsync((void *)(fusedSendBuff + sendBuffOffset),
                                (const void *)t[i].block()->mutable_data(),
                                t[i].Size() * sizeof(float),
                                cudaMemcpyDeviceToDevice, ctx->c1));
            sendBuffOffset += t[i].Size();
          }
        },
        prev_blocks_, blocks_, "Dist_c1_fusedSynch_filling");

  } else {
    // send the tensors in the buffer
    device_->Exec(
        [this](Context *ctx) mutable {
          // wait for the memcpy to complete
          CUDA_CHECK(cudaEventRecord(event, ctx->c1));
          CUDA_CHECK(cudaStreamWaitEvent(ctx->s, event, 0));
        },
        prev_blocks_, prev_blocks_, "Waiting");
    device_->Exec(
        [this](Context *ctx) mutable {
          allReduce((int)sendBuffOffset, (void *)fusedSendBuff,
                    (void *)fusedRecvBuff, ncclFloat, ctx);
          sendBuffOffset = 0;
        },
        prev_blocks_, blocks_, "Dist_s_fusedSynch_allreduce");
    device_->Exec(
        [this](Context *ctx) mutable {
          // wait for the allreduce to complete
          CUDA_CHECK(cudaEventRecord(event, ctx->s));
          CUDA_CHECK(cudaStreamWaitEvent(ctx->c1, event, 0));
        },
        blocks_, blocks_, "Waiting");
    device_->Exec(
        [this, t](Context *ctx) mutable {
          // copy data back to tensors after allreduce
          size_t offset = 0;
          for (size_t i = 0; i < t.size(); i++) {
            CUDA_CHECK(cudaMemcpyAsync((void *)t[i].block()->mutable_data(),
                                       (const void *)(fusedRecvBuff + offset),
                                       t[i].Size() * sizeof(float),
                                       cudaMemcpyDeviceToDevice, ctx->c1));
            offset += t[i].Size();
          }
        },
        blocks_, blocks_, "Dist_c1_fusedSynch_copyBackToTensor");
  }
}

void Communicator::synch(Tensor &t) {
  // generateBlocks(t);
  device_ = t.device();

  device_->Exec(
      [this, t](Context *ctx) mutable {
        // record the event of the default cuda stream and follow it
        CUDA_CHECK(cudaEventRecord(event, ctx->stream));
        CUDA_CHECK(cudaStreamWaitEvent(ctx->s, event, 0));
      },
      {t.block()}, {t.block()}, "Waiting");

  device_->Exec(
      [this, t](Context *ctx) mutable {
        void *addr = t.block()->mutable_data();
        allReduce(t.Size(), addr, addr, ncclFloat, ctx);
      },
      {t.block()}, {t.block()}, "Dist_s_synch_allreduce");
}

void Communicator::fusedSynchHalf(vector<Tensor> &t, bool send) {
  CHECK_GT(t.size(), 0);

  generateBlocks(t);

  if (halfInitialized == false) halfInit();

  if (!send) {
    // buffer the tensors and convert them into half
    device_->Exec(
        [this](Context *ctx) mutable {
          // record the event of the default cuda stream and follow it
          CUDA_CHECK(cudaEventRecord(event, ctx->stream));
          CUDA_CHECK(cudaStreamWaitEvent(ctx->c1, event, 0));
        },
        prev_blocks_, prev_blocks_, "Waiting");
    device_->Exec(
        [this, t](Context *ctx) mutable {
          size_t offset = 0;
          // memory copy to fusedBuff
          for (size_t i = 0; i < t.size(); i++) {
            CUDA_CHECK(
                cudaMemcpyAsync((void *)(fusedSendBuff + sendBuffOffset),
                                (const void *)t[i].block()->mutable_data(),
                                t[i].Size() * sizeof(float),
                                cudaMemcpyDeviceToDevice, ctx->c1));
            sendBuffOffset += t[i].Size();
            offset += t[i].Size();
          }
        },
        prev_blocks_, blocks_, "Dist_c1_fusedSynchHalf_filling");
  } else {
    // send the tensors in the buffer
    device_->Exec(
        [this](Context *ctx) mutable {
          cuda::float2half(sendBuffOffset, fusedSendBuff, fusedSendBuffHalf,
                           ctx->c1);
        },
        prev_blocks_, blocks_, "Dist_c1_fusedSynchHalf_float2half");
    device_->Exec(
        [this](Context *ctx) mutable {
          // wait for the memcpy to complete
          CUDA_CHECK(cudaEventRecord(event, ctx->c1));
          CUDA_CHECK(cudaStreamWaitEvent(ctx->s, event, 0));
        },
        blocks_, blocks_, "Waiting");
    device_->Exec(
        [this](Context *ctx) mutable {
          allReduce((int)sendBuffOffset, (void *)fusedSendBuffHalf,
                    (void *)fusedRecvBuffHalf, ncclHalf, ctx);
        },
        blocks_, blocks_, "Dist_s_fusedSynchHalf_allreduce");
    device_->Exec(
        [this](Context *ctx) mutable {
          // wait for the allreduce to complete
          CUDA_CHECK(cudaEventRecord(event, ctx->s));
          CUDA_CHECK(cudaStreamWaitEvent(ctx->c2, event, 0));          
        },
        blocks_, blocks_, "Waiting");
    device_->Exec(
        [this, t](Context *ctx) mutable {
          cuda::half2float(sendBuffOffset, fusedRecvBuffHalf, fusedRecvBuff,
                           ctx->c2);

          sendBuffOffset = 0;

          // copy data back to tensors after allreduce
          size_t offset = 0;
          for (size_t i = 0; i < t.size(); i++) {
            CUDA_CHECK(cudaMemcpyAsync((void *)t[i].block()->mutable_data(),
                                       (const void *)(fusedRecvBuff + offset),
                                       t[i].Size() * sizeof(float),
                                       cudaMemcpyDeviceToDevice, ctx->c2));
            offset += t[i].Size();
          }
        },
        blocks_, blocks_, "Dist_c2_fusedSynchHalf_half2floatcopy");
  }
}

void Communicator::synchHalf(Tensor &t) {
  generateBlocks(t);

  if (halfInitialized == false) halfInit();

  device_->Exec(
      [this, t](Context *ctx) mutable {
        // record the event of the default cuda stream and follow it
        CUDA_CHECK(cudaEventRecord(event, ctx->stream));
        CUDA_CHECK(cudaStreamWaitEvent(ctx->c1, event, 0));
      },
      blocks_, blocks_, "Waiting");
  device_->Exec(
      [this, t](Context *ctx) mutable {
        float *addr = static_cast<float *>(t.block()->mutable_data());
        cuda::float2half(t.Size(), addr, fusedSendBuffHalf, ctx->c1);
      },
      blocks_, blocks_, "Dist_c1_synchHalf_float2half");
  device_->Exec(
      [this, t](Context *ctx) mutable {
        // wait for conversion to half precision complete
        CUDA_CHECK(cudaEventRecord(event, ctx->c1));
        CUDA_CHECK(cudaStreamWaitEvent(ctx->s, event, 0));
      },
      blocks_, blocks_, "Waiting");
  device_->Exec(
      [this, t](Context *ctx) mutable {
        allReduce(t.Size(), (void *)fusedSendBuffHalf,
                  (void *)fusedRecvBuffHalf, ncclHalf, ctx);
      },
      blocks_, blocks_, "Dist_s_synchHalf_allreduce");
  device_->Exec(
      [this, t](Context *ctx) mutable {
        // wait for the allreduce to complete
        CUDA_CHECK(cudaEventRecord(event, ctx->s));
        CUDA_CHECK(cudaStreamWaitEvent(ctx->c2, event, 0));
      },
      blocks_, blocks_, "Waiting");
  device_->Exec(
      [this, t](Context *ctx) mutable {
        float *addr = static_cast<float *>(t.block()->mutable_data());
        cuda::half2float(t.Size(), fusedRecvBuffHalf, addr, ctx->c2);
      },
      blocks_, blocks_, "Dist_c2_synchHalf_half2float");

}

void Communicator::sparsification(Tensor &t, Tensor &accumulation,
                                  float sparsThreshold, bool topK) {
  generateBlocks(t);
  blocks_.push_back(accumulation.block());

  device_->Exec(
      [=](Context *ctx) mutable {
        _sparsification(t, &accumulation, sparsThreshold, topK, ctx);
      },
      blocks_, blocks_, "Dist_c1c2_sparsification");
}

void Communicator::sparsification(Tensor &t, float sparsThreshold, bool topK) {
  generateBlocks(t);

  t.device()->Exec(
      [=](Context *ctx) mutable {
        _sparsification(t, (Tensor *)NULL, sparsThreshold, topK, ctx);
      },
      blocks_, blocks_, "Dist_c1c2_sparsification");
}

void Communicator::_sparsification(Tensor &t, Tensor *accumulation,
                                   float sparsThreshold, bool topK,
                                   Context *ctx) {
  // threshold for sprasification
  threshold = sparsThreshold;

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, ctx->stream));
  CUDA_CHECK(cudaStreamWaitEvent(ctx->c1, event, 0));

  // memory copy to fusedBuff
  CUDA_CHECK(cudaMemcpyAsync(
      (void *)fusedSendBuff, (const void *)t.block()->mutable_data(),
      t.Size() * sizeof(float), cudaMemcpyDeviceToDevice, ctx->c1));

  float *accumPtr;

  if (accumulation != NULL)
    accumPtr = (float *)accumulation->block()->mutable_data();
  else
    accumPtr = NULL;

  if (topK == false)
    valSparsAllReduce(t.Size(), accumPtr, ctx);
  else
    topKSparsAllReduce(t.Size(), accumPtr, ctx);

  // copy data back to tensor after allreduce
  CUDA_CHECK(cudaMemcpyAsync(
      (void *)t.block()->mutable_data(), (const void *)fusedRecvBuff,
      t.Size() * sizeof(float), cudaMemcpyDeviceToDevice, ctx->c2));
}

void Communicator::fusedSparsification(vector<Tensor> &t, Tensor &accumulation,
                                       float sparsThreshold, bool topK) {
  CHECK_GT(t.size(), 0);

  generateBlocks(t);
  blocks_.push_back(accumulation.block());

  device_->Exec(
      [=](Context *ctx) mutable {
        _fusedSparsification(t, &accumulation, sparsThreshold, topK, ctx);
      },
      blocks_, blocks_, "Dist_c1c2_fusedSparsification");
}

void Communicator::fusedSparsification(vector<Tensor> &t, float sparsThreshold,
                                       bool topK) {
  CHECK_GT(t.size(), 0);

  generateBlocks(t);

  device_->Exec(
      [=](Context *ctx) mutable {
        _fusedSparsification(t, (Tensor *)NULL, sparsThreshold, topK, ctx);
      },
      blocks_, blocks_, "Dist_c1c2_fusedSparsification");
}

void Communicator::_fusedSparsification(vector<Tensor> &t, Tensor *accumulation,
                                        float sparsThreshold, bool topK,
                                        Context *ctx) {
  // threshold for sprasification
  threshold = sparsThreshold;

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, ctx->stream));
  CUDA_CHECK(cudaStreamWaitEvent(ctx->c1, event, 0));

  size_t offset = 0;

  // memory copy to fusedBuff
  for (size_t i = 0; i < t.size(); i++) {
    CUDA_CHECK(cudaMemcpyAsync((void *)(fusedSendBuff + offset),
                               (const void *)t[i].block()->mutable_data(),
                               t[i].Size() * sizeof(float),
                               cudaMemcpyDeviceToDevice, ctx->c1));
    offset += t[i].Size();
  }

  float *accumPtr;

  if (accumulation != NULL)
    accumPtr = (float *)accumulation->block()->mutable_data();
  else
    accumPtr = NULL;

  if (topK == false)
    valSparsAllReduce(offset, accumPtr, ctx);
  else
    topKSparsAllReduce(offset, accumPtr, ctx);

  // copy data back to tensors after allreduce
  offset = 0;
  for (size_t i = 0; i < t.size(); i++) {
    CUDA_CHECK(cudaMemcpyAsync((void *)t[i].block()->mutable_data(),
                               (const void *)(fusedRecvBuff + offset),
                               t[i].Size() * sizeof(float),
                               cudaMemcpyDeviceToDevice, ctx->c2));
    offset += t[i].Size();
  }
}

void Communicator::valSparsAllReduce(size_t num, float *accumulation,
                                     Context *ctx) {
  if (sparsInitialized == false) sparsInit();

  if (accumulation != NULL) {
    // add the previous accumulation
    cuda::add(num, fusedSendBuff, accumulation, fusedSendBuff, ctx->c1);
    // backup the fusedSendBuff
    CUDA_CHECK(cudaMemcpyAsync((void *)backupBuff, (const void *)fusedSendBuff,
                               sizeof(float) * num, cudaMemcpyDeviceToDevice,
                               ctx->c1));
  }

  // sparsification based on threshold
  cuda::sparsabs(num, threshold, fusedSendBuff, fusedSendBuff, ctx->c1);

  // output the gradient accumulation
  if (accumulation != NULL)
    cuda::sub(num, backupBuff, fusedSendBuff, accumulation, ctx->c1);

  // produce the index of the sparse array
  cuda::sparsindex(num, fusedSendBuff, fusedIndex, ctx->c1);

  // remove zero of index to become sprase array and get the num of non-zero nnz
  cuda::removezeroidx(num, fusedIndex, ctx->c1, nnz);

  CUDA_CHECK(cudaMemcpyAsync((void *)nnzGPU, (const void *)nnz, sizeof(int),
                             cudaMemcpyHostToDevice, ctx->c1));

  // all-gather all the nnz from different ranks
  NCCLCHECK(ncclAllGather((const void *)nnzGPU, (void *)nnzAllGPU, 1, ncclInt,
                          comm, ctx->c1));

  CUDA_CHECK(cudaMemcpyAsync((void *)nnzAll, (const void *)nnzAllGPU,
                             sizeof(int) * world_size, cudaMemcpyDeviceToHost,
                             ctx->c1));

  CUDA_CHECK(cudaStreamSynchronize(ctx->c1));

  int nnzMax = 0;
  for (int i = 0; i < world_size; i++)
    if (nnzAll[i] > nnzMax) nnzMax = nnzAll[i];

  // remove zero of values to become sprase array
  cuda::removezeroval(num, fusedSendBuff, ctx->c1);

  CUDA_CHECK(cudaMemcpyAsync((void *)(sparsSendBuff), (const void *)fusedIndex,
                             sizeof(int) * (*nnz), cudaMemcpyDeviceToDevice,
                             ctx->c1));
  CUDA_CHECK(cudaMemcpyAsync(
      (void *)(sparsSendBuff + (*nnz)), (const void *)fusedSendBuff,
      sizeof(float) * (*nnz), cudaMemcpyDeviceToDevice, ctx->c1));

  // wait for the memcpy to complete
  CUDA_CHECK(cudaEventRecord(event, ctx->c1));
  CUDA_CHECK(cudaStreamWaitEvent(ctx->s, event, 0));

  // all-gather all the sparse gradients
  NCCLCHECK(ncclAllGather((const void *)sparsSendBuff, (void *)sparsRecvBuff,
                          2 * nnzMax, ncclFloat, comm, ctx->s));

  // wait for the all-gather to complete
  CUDA_CHECK(cudaEventRecord(event, ctx->s));
  CUDA_CHECK(cudaStreamWaitEvent(ctx->c2, event, 0));

  // reduce the sparse gradients, firstly setting the sum buff value to zero
  CUDA_CHECK(cudaMemsetAsync(fusedRecvBuff, 0, num * sizeof(float), ctx->c2));

  size_t offset = 0;
  float alpha = 1.0;

  // add the spase gradent from each rank to the sum buff to finish the
  // all-reduce process
  CUSPARSE_CHECK(cusparseSetStream(cusparse_handle, ctx->c2));

  for (int i = 0; i < world_size; i++) {
    CUDA_CHECK(cudaMemcpyAsync(
        (void *)xInd, (const void *)(sparsRecvBuff + offset),
        sizeof(int) * nnzAll[i], cudaMemcpyDeviceToDevice, ctx->c2));
    offset += nnzAll[i];
    CUDA_CHECK(cudaMemcpyAsync(
        (void *)xVal, (const void *)(sparsRecvBuff + offset),
        sizeof(float) * nnzAll[i], cudaMemcpyDeviceToDevice, ctx->c2));
    offset += (2 * nnzMax - nnzAll[i]);
    CUSPARSE_CHECK(cusparseSaxpyi(cusparse_handle, nnzAll[i], &alpha, xVal,
                                  xInd, fusedRecvBuff,
                                  CUSPARSE_INDEX_BASE_ONE));
  }
}

void Communicator::topKSparsAllReduce(size_t num, float *accumulation,
                                      Context *ctx) {
  if (sparsInitialized == false) sparsInit();

  // use gradient accumulation
  if (accumulation != NULL) {
    // add the previous accumulation
    cuda::add(num, fusedSendBuff, accumulation, fusedSendBuff, ctx->c1);
    // backup the fusedSendBuff
    CUDA_CHECK(cudaMemcpyAsync((void *)backupBuff, (const void *)fusedSendBuff,
                               sizeof(float) * num, cudaMemcpyDeviceToDevice,
                               ctx->c1));
  }

  // generate an index and sort the fusedSendBuff from large to small values
  cuda::generateindex(num, fusedIndex, ctx->c1);
  cuda::sortbykey(num, fusedSendBuff, fusedIndex, ctx->c1);

  // determine the number of topK for communication
  int nnzMax = (int)ceil(threshold * num);

  // output the gradient accumulation
  float alpha = 1.0;
  if (accumulation != NULL) {
    CUDA_CHECK(cudaMemsetAsync(accumulation, 0, num * sizeof(float), ctx->c1));
    CUSPARSE_CHECK(cusparseSetStream(cusparse_handle, ctx->c1));
    CUSPARSE_CHECK(cusparseSaxpyi(cusparse_handle, nnzMax, &alpha,
                                  fusedSendBuff, fusedIndex, accumulation,
                                  CUSPARSE_INDEX_BASE_ONE));
    cuda::sub(num, backupBuff, accumulation, accumulation, ctx->c1);
  }

  // the topK value and index will be sent
  CUDA_CHECK(cudaMemcpyAsync((void *)(sparsSendBuff), (const void *)fusedIndex,
                             sizeof(int) * nnzMax, cudaMemcpyDeviceToDevice,
                             ctx->c1));
  CUDA_CHECK(cudaMemcpyAsync(
      (void *)(sparsSendBuff + nnzMax), (const void *)fusedSendBuff,
      sizeof(float) * nnzMax, cudaMemcpyDeviceToDevice, ctx->c1));

  // wait for the memcpy to complete
  CUDA_CHECK(cudaEventRecord(event, ctx->c1));
  CUDA_CHECK(cudaStreamWaitEvent(ctx->s, event, 0));

  // all-gather all the sparse gradients
  NCCLCHECK(ncclAllGather((const void *)sparsSendBuff, (void *)sparsRecvBuff,
                          2 * nnzMax, ncclFloat, comm, ctx->s));

  // wait for the all-gather to complete
  CUDA_CHECK(cudaEventRecord(event, ctx->s));
  CUDA_CHECK(cudaStreamWaitEvent(ctx->c2, event, 0));

  // reduce the sparse gradients, firstly setting the sum buff value to zero
  CUDA_CHECK(cudaMemsetAsync(fusedRecvBuff, 0, num * sizeof(float), ctx->c2));

  size_t offset = 0;

  CUSPARSE_CHECK(cusparseSetStream(cusparse_handle, ctx->c2));

  // add the spase gradent from each rank to the sum buff to finish the
  // all-reduce process
  for (int i = 0; i < world_size; i++) {
    CUDA_CHECK(cudaMemcpyAsync(
        (void *)xInd, (const void *)(sparsRecvBuff + offset),
        sizeof(int) * nnzMax, cudaMemcpyDeviceToDevice, ctx->c2));
    offset += nnzMax;
    CUDA_CHECK(cudaMemcpyAsync(
        (void *)xVal, (const void *)(sparsRecvBuff + offset),
        sizeof(float) * nnzMax, cudaMemcpyDeviceToDevice, ctx->c2));
    offset += nnzMax;
    CUSPARSE_CHECK(cusparseSaxpyi(cusparse_handle, nnzMax, &alpha, xVal, xInd,
                                  fusedRecvBuff, CUSPARSE_INDEX_BASE_ONE));
  }
}
}  // namespace singa

#endif  // USE_DIST
