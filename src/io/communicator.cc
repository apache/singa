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
Communicator::Communicator(int gpu_num, int gpu_per_node, const NcclIdHolder &holder){

  // this contructor is for NCCL WITHOUT MPI
  UseMPI = false;

  // Determine the rank of the collective communication
  totalMPIRanksInGlobal=gpu_per_node;
  MPIRankInLocal=gpu_num;
  MPIRankInGlobal=gpu_num;

  // copy the nccl unqiue id from the input id holder
  id = holder.id;

  // setup cuda stream and nccl communicator
  CUDA_CHECK(cudaSetDevice(gpu_num));
  CUDA_CHECK(cudaStreamCreate(&s));
  NCCLCHECK(ncclCommInitRank(&comm, gpu_per_node, id, gpu_num));

} // end of constructor 

// contructer for application with MPI
Communicator::Communicator(){

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
  CUDA_CHECK(cudaSetDevice(MPIRankInLocal));
  CUDA_CHECK(cudaStreamCreate(&s));
  NCCLCHECK(ncclCommInitRank(&comm, totalMPIRanksInGlobal, id, MPIRankInGlobal));

} // end of constructor 


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
  CUDA_CHECK(cudaStreamSynchronize(s));
}

Communicator::~Communicator(){
  //finalizing NCCL
  ncclCommDestroy(comm);
  if (UseMPI == true) MPICHECK(MPI_Finalize());
}

void synch(Tensor &t, Communicator &c){
  void* addr = t.block()->mutable_data();
  c.allReduce(t.Size(), addr, addr);
  c.wait();
}

}

#endif // USE_DIST
