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

/*interface file for swig */

%module dist_communicator
%include "std_vector.i"

%{
#include "singa/io/communicator.h"
%}

namespace singa{

#if USE_DIST

class NcclIdHolder {
public:
  ncclUniqueId id;
  NcclIdHolder();
};

class Communicator {
public:
  int global_rank;
  int world_size;
  int local_rank;
  Communicator(int limit);
  Communicator(int local_rank, int world_size, const NcclIdHolder &holder, int limit);
  void synch(Tensor &t);
  void fusedSynch(std::vector<Tensor> &t, bool send = true);
  void synchHalf(Tensor &t);
  void fusedSynchHalf(std::vector<Tensor> &t, bool send = true);
  void sparsification(Tensor &t, Tensor &accumulation, float sparsThreshold, bool topK);
  void sparsification(Tensor &t, float sparsThreshold, bool topK);
  void fusedSparsification(std::vector<Tensor> &, Tensor &accumulation, float sparsThreshold, bool topK);
  void fusedSparsification(std::vector<Tensor> &, float sparsThreshold, bool topK);
  void wait();
};


#endif  // USE_DIST

}
