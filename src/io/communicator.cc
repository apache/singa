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

#include "singa/utils/cuda_utils.h"
#include <iostream>

#ifdef USE_DIST

#include "singa/io/communicator.h"

namespace singa{


static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

NcclIdHolder::NcclIdHolder(){
  ncclGetUniqueId(&id); 
} // end of constructor 

NcclIdHolder::~NcclIdHolder(){  
} 

// contructer for application with python multi-processing module
Communicator::Communicator(int gpu_num, int gpu_per_node, const NcclIdHolder &holder, int buffSize){

  maxSize = (size_t) buffSize;
  // this contructor is for NCCL WITHOUT MPI
  UseMPI = false;
  // Determine the rank of the collective communication
  totalMPIRanksInGlobal=gpu_per_node;
  MPIRankInLocal=gpu_num;
  MPIRankInGlobal=gpu_num;

  // copy the nccl unqiue id from the input id holder
  id = holder.id;

  // setup cuda stream and nccl communicator
  setup(gpu_num);

} // end of constructor 

// contructer for application with MPI
Communicator::Communicator(int buffSize){

  maxSize = (size_t) buffSize;
  // this contructor is for NCCL WITH MPI
  UseMPI = true;

  // MPI initialization
  MPICHECK(MPI_Init(NULL, NULL));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &MPIRankInGlobal));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &totalMPIRanksInGlobal));

  // calculating MPIRankInLocal which is used in selecting a GPU
  MPIRankInLocal=0;
  uint64_t hostHashs[totalMPIRanksInGlobal];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[MPIRankInGlobal] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
    		 sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<totalMPIRanksInGlobal; p++) {
     if (p == MPIRankInGlobal) break;
     if (hostHashs[p] == hostHashs[MPIRankInGlobal]) MPIRankInLocal++;
  }

  // generating NCCL unique nccl ID at one process and broadcasting it to all
  if (MPIRankInGlobal == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // setup cuda stream and nccl communicator  
  setup(MPIRankInLocal);

} // end of constructor 

void Communicator::setup(int gpu_num){

  CUDA_CHECK(cudaSetDevice(gpu_num));
  NCCLCHECK(ncclCommInitRank(&comm, totalMPIRanksInGlobal, id, MPIRankInGlobal));
  CUDA_CHECK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, 0));
  CUDA_CHECK(cudaStreamCreateWithPriority(&c, cudaStreamNonBlocking, 1));
  CUDA_CHECK(cudaMalloc(&fusedSendBuff, maxSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fusedRecvBuff, maxSize * sizeof(float)));
  CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventBlockingSync | cudaEventDisableTiming));

}

void Communicator::allReduce(int size, void* sendbuff, void* recvbuff)
{

  NCCLCHECK(ncclAllReduce((const void*)sendbuff,
                             (void*)recvbuff,
                             size,
                             ncclFloat,
                             ncclSum,
                             comm, 
                             s));

}

void Communicator::wait(){
  //synchronizing on CUDA stream to complete NCCL communication
  CUDA_CHECK(cudaEventRecord(event, s));
  CUDA_CHECK(cudaStreamWaitEvent(NULL, event, 0));
  CUDA_CHECK(cudaEventRecord(event, c));
  CUDA_CHECK(cudaStreamWaitEvent(NULL, event, 0));
}

Communicator::~Communicator(){
  //finalizing NCCL
  ncclCommDestroy(comm);
  if (UseMPI == true) MPICHECK(MPI_Finalize());
  CUDA_CHECK(cudaFree(fusedSendBuff));
  CUDA_CHECK(cudaFree(fusedRecvBuff));
}

void Communicator::fusedSynch(vector<Tensor> &t){

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, NULL));
  CUDA_CHECK(cudaStreamWaitEvent(c, event, 0));
  
  size_t offset = 0;

  //memory copy to fusedBuff
  for (size_t i = 0; i < t.size(); i++)
  {
    CUDA_CHECK(cudaMemcpyAsync((void*) (fusedSendBuff + offset), (const void*) t[i].block()->mutable_data(), t[i].Size() * sizeof(float), cudaMemcpyDeviceToDevice, c));
    offset += t[i].Size();
  }

  // wait for the memcpy to complete
  CUDA_CHECK(cudaEventRecord(event, c));
  CUDA_CHECK(cudaStreamWaitEvent(s, event, 0));

  allReduce((int) offset, (void*) fusedSendBuff, (void*) fusedRecvBuff);

  // wait for the allreduce to complete
  CUDA_CHECK(cudaEventRecord(event, s));
  CUDA_CHECK(cudaStreamWaitEvent(c, event, 0));

  //copy data back to tensors after allreduce
  offset = 0;
  for (size_t i = 0; i < t.size(); i++)
  {
    CUDA_CHECK(cudaMemcpyAsync((void*) t[i].block()->mutable_data(), (const void*) (fusedRecvBuff + offset), t[i].Size() * sizeof(float), cudaMemcpyDeviceToDevice, c));
    offset += t[i].Size();
  }

}

void Communicator::synch(Tensor &t){

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, NULL));
  CUDA_CHECK(cudaStreamWaitEvent(s, event, 0));

  void* addr = t.block()->mutable_data();
  allReduce(t.Size(), addr, addr);

}

}

#endif // USE_DIST
