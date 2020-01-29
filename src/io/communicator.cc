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
#include "./math_kernel.h"

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
  setup();

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
  setup();

} // end of constructor 

void Communicator::setup(){

  CUDA_CHECK(cudaSetDevice(MPIRankInLocal));
  NCCLCHECK(ncclCommInitRank(&comm, totalMPIRanksInGlobal, id, MPIRankInGlobal));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&c1, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&c2, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&fusedSendBuff, maxSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fusedRecvBuff, maxSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fusedSendBuffHalf, maxSize * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&fusedRecvBuffHalf, maxSize * sizeof(__half)));
  CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventBlockingSync | cudaEventDisableTiming));
  sparsInitialized = false;
}

void Communicator::sparsInit(){

  //initize sparsification environment
  CUDA_CHECK(cudaSetDevice(MPIRankInLocal));
  CUDA_CHECK(cudaMalloc(&sparsRecvBuff, (int) (maxSize * sizeof(float) * totalMPIRanksInGlobal)));
  CUDA_CHECK(cudaMalloc(&sparsSendBuff, (int) (maxSize * sizeof(float))));
  CUDA_CHECK(cudaMalloc(&backupBuff, maxSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fusedIndex, maxSize * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&xInd, (int) (sizeof(int) * maxSize)));
  CUDA_CHECK(cudaMalloc(&xVal, (int) (sizeof(float) * maxSize)));
  CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
  CUSPARSE_CHECK(cusparseSetStream(cusparse_handle, c2));
  nnz = (int*) malloc(sizeof(int));
  nnzAll = (int*) malloc(sizeof(int) * totalMPIRanksInGlobal);
  CUDA_CHECK(cudaMalloc(&nnzGPU, sizeof(int) * totalMPIRanksInGlobal));
  CUDA_CHECK(cudaMalloc(&nnzAllGPU, sizeof(int) * totalMPIRanksInGlobal));
  sparsInitialized = true;

}

void Communicator::allReduce(int size, void* sendbuff, void* recvbuff, ncclDataType_t ncclType)
{

  NCCLCHECK(ncclAllReduce((const void*)sendbuff,
                             (void*)recvbuff,
                             size,
                             ncclType,
                             ncclSum,
                             comm, 
                             s));

}

void Communicator::wait(){
  //synchronizing on all the CUDA streams used by communicator
  CUDA_CHECK(cudaEventRecord(event, s));
  CUDA_CHECK(cudaStreamWaitEvent(NULL, event, 0));
  CUDA_CHECK(cudaEventRecord(event, c1));
  CUDA_CHECK(cudaStreamWaitEvent(NULL, event, 0));
  CUDA_CHECK(cudaEventRecord(event, c2));
  CUDA_CHECK(cudaStreamWaitEvent(NULL, event, 0));
}

Communicator::~Communicator(){
  //finalizing NCCL
  ncclCommDestroy(comm);
  if (UseMPI == true) MPICHECK(MPI_Finalize());
  CUDA_CHECK(cudaFree(fusedSendBuff));
  CUDA_CHECK(cudaFree(fusedRecvBuff));
  CUDA_CHECK(cudaFree(fusedSendBuffHalf));
  CUDA_CHECK(cudaFree(fusedRecvBuffHalf));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaStreamDestroy(c1));
  CUDA_CHECK(cudaStreamDestroy(c2));

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


void Communicator::fusedSynch(vector<Tensor> &t){

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, NULL));
  CUDA_CHECK(cudaStreamWaitEvent(c1, event, 0));
  
  size_t offset = 0;

  //memory copy to fusedBuff
  for (size_t i = 0; i < t.size(); i++)
  {
    CUDA_CHECK(cudaMemcpyAsync((void*) (fusedSendBuff + offset), (const void*) t[i].block()->mutable_data(), t[i].Size() * sizeof(float), cudaMemcpyDeviceToDevice, c1));
    offset += t[i].Size();
  }

  // wait for the memcpy to complete
  CUDA_CHECK(cudaEventRecord(event, c1));
  CUDA_CHECK(cudaStreamWaitEvent(s, event, 0));

  allReduce((int) offset, (void*) fusedSendBuff, (void*) fusedRecvBuff, ncclFloat);

  // wait for the allreduce to complete
  CUDA_CHECK(cudaEventRecord(event, s));
  CUDA_CHECK(cudaStreamWaitEvent(c1, event, 0));

  //copy data back to tensors after allreduce
  offset = 0;
  for (size_t i = 0; i < t.size(); i++)
  {
    CUDA_CHECK(cudaMemcpyAsync((void*) t[i].block()->mutable_data(), (const void*) (fusedRecvBuff + offset), t[i].Size() * sizeof(float), cudaMemcpyDeviceToDevice, c1));
    offset += t[i].Size();
  }

}

void Communicator::synch(Tensor &t){

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, NULL));
  CUDA_CHECK(cudaStreamWaitEvent(s, event, 0));

  void* addr = t.block()->mutable_data();
  allReduce(t.Size(), addr, addr, ncclFloat);

}

void Communicator::fusedSynchHalf(vector<Tensor> &t){

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, NULL));
  CUDA_CHECK(cudaStreamWaitEvent(c1, event, 0));
  
  size_t offset = 0;

  //memory copy to fusedBuff
  for (size_t i = 0; i < t.size(); i++)
  {
    CUDA_CHECK(cudaMemcpyAsync((void*) (fusedSendBuff + offset), (const void*) t[i].block()->mutable_data(), t[i].Size() * sizeof(float), cudaMemcpyDeviceToDevice, c1));
    offset += t[i].Size();
  }

  cuda::float2half(offset, fusedSendBuff, fusedSendBuffHalf, c1);

  // wait for the memcpy to complete
  CUDA_CHECK(cudaEventRecord(event, c1));
  CUDA_CHECK(cudaStreamWaitEvent(s, event, 0));

  allReduce((int) offset, (void*) fusedSendBuffHalf, (void*) fusedRecvBuffHalf, ncclHalf);

  // wait for the allreduce to complete
  CUDA_CHECK(cudaEventRecord(event, s));
  CUDA_CHECK(cudaStreamWaitEvent(c2, event, 0));

  cuda::half2float(offset, fusedRecvBuffHalf, fusedRecvBuff, c2);

  //copy data back to tensors after allreduce
  offset = 0;
  for (size_t i = 0; i < t.size(); i++)
  {
    CUDA_CHECK(cudaMemcpyAsync((void*) t[i].block()->mutable_data(), (const void*) (fusedRecvBuff + offset), t[i].Size() * sizeof(float), cudaMemcpyDeviceToDevice, c2));
    offset += t[i].Size();
  }

}

void Communicator::synchHalf(Tensor &t){

  float* addr = static_cast<float*>(t.block()->mutable_data());

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, NULL));
  CUDA_CHECK(cudaStreamWaitEvent(c1, event, 0));

  cuda::float2half(t.Size(), addr, fusedSendBuffHalf, c1);

  // wait for conversion to half precision complete
  CUDA_CHECK(cudaEventRecord(event, c1));
  CUDA_CHECK(cudaStreamWaitEvent(s, event, 0));

  allReduce(t.Size(), (void*) fusedSendBuffHalf, (void*) fusedRecvBuffHalf, ncclHalf);

  // wait for the allreduce to complete
  CUDA_CHECK(cudaEventRecord(event, s));
  CUDA_CHECK(cudaStreamWaitEvent(c2, event, 0));

  cuda::half2float(t.Size(), fusedRecvBuffHalf, addr, c2);

}

void Communicator::sparsification(Tensor &t, Tensor &accumulation, float sparsThreshold, bool topK){
  _sparsification(t, &accumulation, sparsThreshold, topK);
}

void Communicator::sparsification(Tensor &t, float sparsThreshold, bool topK){
  _sparsification(t, (Tensor *) NULL, sparsThreshold, topK);
}

void Communicator::_sparsification(Tensor &t, Tensor* accumulation, float sparsThreshold, bool topK){

  // threshold for sprasification
  threshold = sparsThreshold;

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, NULL));
  CUDA_CHECK(cudaStreamWaitEvent(c1, event, 0));

  //memory copy to fusedBuff
  CUDA_CHECK(cudaMemcpyAsync((void*) fusedSendBuff, (const void*) t.block()->mutable_data(), t.Size() * sizeof(float), cudaMemcpyDeviceToDevice, c1));

  float *accumPtr;

  if (accumulation != NULL)
  	accumPtr = (float*) accumulation->block()->mutable_data();
  else
  	accumPtr = NULL;

  if (topK == false)
    valSparsAllReduce(t.Size(), accumPtr);
  else
    topKSparsAllReduce(t.Size(), accumPtr);

  //copy data back to tensor after allreduce
  CUDA_CHECK(cudaMemcpyAsync((void*) t.block()->mutable_data(), (const void*) fusedRecvBuff, t.Size() * sizeof(float), cudaMemcpyDeviceToDevice, c2));

}

void Communicator::fusedSparsification(vector<Tensor> &t, Tensor &accumulation, float sparsThreshold, bool topK){
  _fusedSparsification(t, &accumulation, sparsThreshold, topK);
}

void Communicator::fusedSparsification(vector<Tensor> &t, float sparsThreshold, bool topK){
  _fusedSparsification(t, (Tensor *) NULL, sparsThreshold, topK);
}

void Communicator::_fusedSparsification(vector<Tensor> &t, Tensor* accumulation, float sparsThreshold, bool topK){

  // threshold for sprasification
  threshold = sparsThreshold;

  // record the event of the default cuda stream and follow it
  CUDA_CHECK(cudaEventRecord(event, NULL));
  CUDA_CHECK(cudaStreamWaitEvent(c1, event, 0));
  
  size_t offset = 0;

  //memory copy to fusedBuff
  for (size_t i = 0; i < t.size(); i++)
  {
    CUDA_CHECK(cudaMemcpyAsync((void*) (fusedSendBuff + offset), (const void*) t[i].block()->mutable_data(), t[i].Size() * sizeof(float), cudaMemcpyDeviceToDevice, c1));
    offset += t[i].Size();
  }

  float *accumPtr;

  if (accumulation != NULL)
  	accumPtr = (float*) accumulation->block()->mutable_data();
  else
  	accumPtr = NULL;

  if (topK == false)
    valSparsAllReduce(offset, accumPtr);
  else
    topKSparsAllReduce(offset, accumPtr);    

  //copy data back to tensors after allreduce
  offset = 0;
  for (size_t i = 0; i < t.size(); i++)
  {
    CUDA_CHECK(cudaMemcpyAsync((void*) t[i].block()->mutable_data(), (const void*) (fusedRecvBuff + offset), t[i].Size() * sizeof(float), cudaMemcpyDeviceToDevice, c2));
    offset += t[i].Size();
  }

}

void Communicator::valSparsAllReduce(size_t num, float* accumulation){

  if (sparsInitialized == false)
    sparsInit();

  if (accumulation != NULL)
  {
    // add the previous accumulation
  	cuda::add(num, fusedSendBuff, accumulation, fusedSendBuff, c1);
    // backup the fusedSendBuff
  	CUDA_CHECK(cudaMemcpyAsync((void*) backupBuff, (const void*) fusedSendBuff, sizeof(float) * num, cudaMemcpyDeviceToDevice, c1));
  }

  // sparsification based on threshold
  cuda::sparsabs(num, threshold, fusedSendBuff, fusedSendBuff, c1);

  // output the gradient accumulation
  if (accumulation != NULL)
	  cuda::sub(num, backupBuff, fusedSendBuff, accumulation, c1);

  // produce the index of the sparse array
  cuda::sparsindex(num, fusedSendBuff, fusedIndex, c1);

  // remove zero of index to become sprase array and get the num of non-zero nnz
  cuda::removezeroidx(num, fusedIndex, c1, nnz);

  CUDA_CHECK(cudaMemcpyAsync((void*) nnzGPU, (const void*) nnz, sizeof(int), cudaMemcpyHostToDevice, c1));

  // all-gather all the nnz from different ranks
  NCCLCHECK(ncclAllGather((const void*)nnzGPU,
                             (void*)nnzAllGPU,
                             1,
                             ncclInt,
                             comm, 
                             c1));

  CUDA_CHECK(cudaMemcpyAsync((void*) nnzAll, (const void*) nnzAllGPU, sizeof(int) * totalMPIRanksInGlobal, cudaMemcpyDeviceToHost, c1));

  CUDA_CHECK(cudaStreamSynchronize(c1));

  int nnzMax = 0;
  for (int i = 0; i < totalMPIRanksInGlobal; i++)
      if(nnzAll[i] > nnzMax)
          nnzMax = nnzAll[i];

  // remove zero of values to become sprase array
  cuda::removezeroval(num, fusedSendBuff, c1);

  CUDA_CHECK(cudaMemcpyAsync((void*) (sparsSendBuff), (const void*) fusedIndex, sizeof(int) * (*nnz), cudaMemcpyDeviceToDevice, c1));
  CUDA_CHECK(cudaMemcpyAsync((void*) (sparsSendBuff + (*nnz)), (const void*) fusedSendBuff, sizeof(float) * (*nnz), cudaMemcpyDeviceToDevice, c1));

  // wait for the memcpy to complete
  CUDA_CHECK(cudaEventRecord(event, c1));
  CUDA_CHECK(cudaStreamWaitEvent(s, event, 0));

  // all-gather all the sparse gradients
  NCCLCHECK(ncclAllGather((const void*)sparsSendBuff,
                             (void*)sparsRecvBuff,
                             2 * nnzMax,
                             ncclFloat,
                             comm, 
                             s));

  // wait for the all-gather to complete
  CUDA_CHECK(cudaEventRecord(event, s));
  CUDA_CHECK(cudaStreamWaitEvent(c2, event, 0));

  // reduce the sparse gradients, firstly setting the sum buff value to zero
  CUDA_CHECK(cudaMemsetAsync(fusedRecvBuff, 0, num *sizeof(float) , c2));

  size_t offset = 0;
  float alpha = 1.0;

  // add the spase gradent from each rank to the sum buff to finish the all-reduce process
  for (int i = 0; i < totalMPIRanksInGlobal; i++)
  {
      CUDA_CHECK(cudaMemcpyAsync((void*) xInd, (const void*) (sparsRecvBuff + offset), sizeof(int) * nnzAll[i], cudaMemcpyDeviceToDevice, c2));
      offset += nnzAll[i];
      CUDA_CHECK(cudaMemcpyAsync((void*) xVal, (const void*) (sparsRecvBuff + offset), sizeof(float) * nnzAll[i], cudaMemcpyDeviceToDevice, c2));
      offset += (2 * nnzMax - nnzAll[i]);
      CUSPARSE_CHECK(cusparseSaxpyi(cusparse_handle,
                     				nnzAll[i],
                     				&alpha,
                     				xVal,
                     				xInd,
                     				fusedRecvBuff,
                     				CUSPARSE_INDEX_BASE_ONE));
  }

}

void Communicator::topKSparsAllReduce(size_t num, float* accumulation){

  if (sparsInitialized == false)
    sparsInit();

  // use gradient accumulation
  if (accumulation != NULL)
  {
    // add the previous accumulation
  	cuda::add(num, fusedSendBuff, accumulation, fusedSendBuff, c1);
    // backup the fusedSendBuff
  	CUDA_CHECK(cudaMemcpyAsync((void*) backupBuff, (const void*) fusedSendBuff, sizeof(float) * num, cudaMemcpyDeviceToDevice, c1));
  }

  // generate an index and sort the fusedSendBuff from large to small values
  cuda::generateindex(num, fusedIndex, c1);
  cuda::sortbykey(num, fusedSendBuff, fusedIndex, c1);

  // determine the number of topK for communication
  int nnzMax = (int) ceil(threshold * num);

  // output the gradient accumulation
  float alpha = 1.0;
  if (accumulation != NULL)
  {
  	CUDA_CHECK(cudaMemsetAsync(accumulation, 0, num * sizeof(float) , c1));
  	CUSPARSE_CHECK(cusparseSetStream(cusparse_handle, c1));
  	CUSPARSE_CHECK(cusparseSaxpyi(cusparse_handle,
  	               				nnzMax,
  	               				&alpha,
  	               				fusedSendBuff,
  	               				fusedIndex,
  	               				accumulation,
  	               				CUSPARSE_INDEX_BASE_ONE));
  	cuda::sub(num, backupBuff, accumulation, accumulation, c1);
  }

  // the topK value and index will be sent
  CUDA_CHECK(cudaMemcpyAsync((void*) (sparsSendBuff), (const void*) fusedIndex, sizeof(int) * nnzMax, cudaMemcpyDeviceToDevice, c1));
  CUDA_CHECK(cudaMemcpyAsync((void*) (sparsSendBuff + nnzMax), (const void*) fusedSendBuff, sizeof(float) * nnzMax, cudaMemcpyDeviceToDevice, c1));

  // wait for the memcpy to complete
  CUDA_CHECK(cudaEventRecord(event, c1));
  CUDA_CHECK(cudaStreamWaitEvent(s, event, 0));

  // all-gather all the sparse gradients
  NCCLCHECK(ncclAllGather((const void*)sparsSendBuff,
                             (void*)sparsRecvBuff,
                             2 * nnzMax,
                             ncclFloat,
                             comm, 
                             s));

  // wait for the all-gather to complete
  CUDA_CHECK(cudaEventRecord(event, s));
  CUDA_CHECK(cudaStreamWaitEvent(c2, event, 0));

  // reduce the sparse gradients, firstly setting the sum buff value to zero
  CUDA_CHECK(cudaMemsetAsync(fusedRecvBuff, 0, num *sizeof(float) , c2));

  size_t offset = 0;

  CUSPARSE_CHECK(cusparseSetStream(cusparse_handle, c2));

  // add the spase gradent from each rank to the sum buff to finish the all-reduce process
  for (int i = 0; i < totalMPIRanksInGlobal; i++)
  {
    CUDA_CHECK(cudaMemcpyAsync((void*) xInd, (const void*) (sparsRecvBuff + offset), sizeof(int) * nnzMax, cudaMemcpyDeviceToDevice, c2));
    offset += nnzMax;
    CUDA_CHECK(cudaMemcpyAsync((void*) xVal, (const void*) (sparsRecvBuff + offset), sizeof(float) * nnzMax, cudaMemcpyDeviceToDevice, c2));
    offset += nnzMax;
    CUSPARSE_CHECK(cusparseSaxpyi(cusparse_handle,
                   				  nnzMax,
                     			  &alpha,
                     			  xVal,
                     			  xInd,
                     			  fusedRecvBuff,
                     			  CUSPARSE_INDEX_BASE_ONE));
  }

}


}

#endif // USE_DIST
