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

#ifndef SINGA_STUB_H_
#define SINGA_STUB_H_

#include <queue>
#include <unordered_map>
#include <vector>
#include <string>
#include "singa/comm/socket.h"
#include "singa/neuralnet/neuralnet.h"
#include "singa/proto/job.pb.h"
#include "singa/proto/singa.pb.h"
#include "singa/utils/factory.h"
#include "singa/utils/param.h"
#include "singa/utils/singleton.h"
#include "singa/server.h"
#include "singa/worker.h"

namespace singa {

class Stub {
 public:
  ~Stub();
  /**
   * Find an endpoint to bind.
   */
  void Setup();
  /**
   * The Stub instance runs this function in the main thread to handle (e.g.,
   * forward) messages from workers and servers.
   *
   * @param[in] slice2server the k-th value is the ID of the server that is in
   * charge of updating the Param slice with ID k. Large Param objects are
   * sliced into subsets for load-balance. Different subsets are updated by
   * different servers.
   */
  void Run(const vector<int>& slice2server,
      const std::vector<Worker*>& workers,
      const std::vector<Server*>& servers);

  void set_router(Router* router) {
    router_ = router;
  }

 protected:
  /**
   * Create a socket to send msg to the specified process
   * @param dst_procs the dst process (logical) ID
   * @return the newly created socket
   */
  Dealer* CreateInterProcsDealer(int dst_procs);
  /**
   * Generate a request message to Get the parameter object.
   */
  const std::vector<Msg*> HandleGetRequest(ParamEntry* entry, Msg** msg);
  void HandleGetResponse(ParamEntry* entry, Msg** msg);
  /**
   * Generate a request message to Update the parameter object.
   */
  const std::vector<Msg*> HandleUpdateRequest(ParamEntry* entry, Msg** msg);
  /**
   * Handle response msg from servers for the update requests.
   */
  void HandleUpdateResponse(ParamEntry* entry, Msg** msg);
  /**
   * Generate a request message to Put the parameter object.
   */
  const std::vector<Msg*> HandlePutRequest(ParamEntry* entry, Msg** msg);
  /**
   * Called by HandlePut, HandleUpdate and HandleGet functions
   * @param type message type
   * @param version param version
   * @param entry
   * @param msg
   * @param ret generated messages
   */
  void GenMsgs(int type, int version, ParamEntry* entry,
    Msg* msg, std::vector<Msg*> *ret);


 protected:
  Router *router_ = nullptr;
  std::vector<int> slice2server_;
};

}  // namespace singa

#endif  // SINGA_STUB_H_
