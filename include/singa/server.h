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

#ifndef SINGA_SERVER_H_
#define SINGA_SERVER_H_

#include <unordered_map>
#include <vector>
#include "singa/comm/socket.h"
#include "singa/proto/job.pb.h"
#include "singa/utils/param.h"
#include "singa/utils/updater.h"

namespace singa {

 /* Repsond to worker's get/put/udpate request, and periodically syncing with
  * other servers.
  *
  * Normally, the Server creates a response message for each request which
  * will be sent back to the one who issued the request. However, if the request
  * are not processed successfully, the original message will be returned. The
  * sever does not know the returned message is a response or the original
  * message. It just sends it to the router. The router will decided to
  * re-send the request to the server or send it to the worker.
  */
class Server {
 public:
  ~Server();
  Server(int group_id, int server_id,
      const JobProto& job_conf,
      const std::vector<int>& slice2group,
      const std::vector<int>& slice2server);
  void Run();
  inline int grp_id() const { return grp_id_; }
  inline int id() const { return id_; }

 protected:
  /**
   * Process GET request.
   *
   * @return the orignal message or a response message which contains the values
   * of the Param with the request version.
   */
  Msg* HandleGet(Msg** msg);
  /**
   * Process Update request.
   *
   * It waits until received the gradients from all workers from the same worker
   * group. After updating, it responses to each sender with the new Param
   * values. It may generate a sync message to the server group that maintains
   * the global version of the updated Param (slice).
   *
   * Note: there is no counter for each worker group on the number of received
   * update requests. Hence it is possible that the server would conduct the
   * update when it receives x requests from group a and y requests from group
   * b where x + y = group size. To avoid this problem, we can
   * -# maintain request list for each group for each Param at the server side
   * -# do not span a worker group among multiple nodes. then the updates from
   * the same group would be locally aggregated on the worker node. And the
   * server would conduct the update immediately after receiving the aggregated
   * request.
   * -# launch only one worker group.
   *
   * @return the orignal message or response message
   */
  const std::vector<Msg*> HandleUpdate(Msg **msg);
  /**
   * Process PUT request.
   *
   * @return the original message or response message. If we don't want to
   * acknowledge the put request, then return nullptr.
   */
  Msg* HandlePut(Msg **msg);
  /**
   * Handle sync request from other server groups.
   *
   * It adds updates of Param (slice) from other server groups directly to
   * local Param (slice). Currently, each Param (slice) has a master group,
   * i.e., slice2group_[sliceid], which would receive such requests from all
   * other server groups for the Param object.
   *
   * @param msg request msg containing the parameter updates
   * @return response msg that contains the fresh parameter values.
   */
  Msg* HandleSyncRequest(Msg** msg);
  /**
   * Handle sync response.
   *
   * The response msg includes the latest values of a Param object from the
   * server group that maintainers this Param object.
   * The local Param values are replaced with the addition result of local
   * udpates since the sync request was sent and the received Param values.
   *
   * @param response message
   */
  void HandleSyncResponse(Msg** msg);

 protected:
  int grp_id_ = -1;
  int id_ = -1;
  Updater* updater_ = nullptr;
  //!< map from slice ID to slice and deleted in the destructor
  std::unordered_map<int, ParamEntry*> shard_;
  std::vector<int> slice2group_, slice2server_;
  //!< num of updates from last sync with master server group for a param/slice
  std::vector<int> n_updates_;
  //!< num of sync requests that have not been responded
  std::vector<int> n_pending_sync_;
  std::vector<Blob<float>> last_sync_;
  std::unordered_map<int, std::vector<Msg*>> buffer_requests_;
};

}  // namespace singa

#endif  // SINGA_SERVER_H_
