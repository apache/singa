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

%module io_snapshot

%{
#include "singa/io/snapshot.h"
%}

namespace std{
%template(PairStrTensor) std::pair<string, singa::Tensor>;
%template(VecPairStrTensor) std::vector<std::pair<string, singa::Tensor>>;
}

namespace singa {

class Snapshot {
 public:
  enum Mode { kRead, kWrite };
  Snapshot(const std::string& prefix, Mode mode, int max_param_size = 10);
  ~Snapshot() {}
  std::vector<std::pair<std::string, Tensor>> Read();
  void Write(const std::string& key, const Tensor& param);
};

}
