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

#ifndef SINGA_UTILS_CLUSTER_RT_H_
#define SINGA_UTILS_CLUSTER_RT_H_

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace singa {

typedef void (*rt_callback)(void *contest);

struct RTCallback {
  rt_callback fn;
  void* ctx;
};

/**
 * ClusterRuntime is a runtime service that manages dynamic configuration
 * and status of the whole cluster. It mainly provides following services:
 *    1)  Provide running status of each server/worker
 *    2)  Translate process id to (hostname:port)
 */
class ClusterRuntime {
 public:
  // ClusterRuntime have different implementation determined when compiling
  static ClusterRuntime* Create(const std::string&host, int job_id);

  virtual ~ClusterRuntime() {}
  /**
   * Initialize the runtime instance
   */
  virtual bool Init() = 0;
  /**
   * register the process, and get a unique process id
   *
   * \return the process id, -1 if failed
   */
  virtual int RegistProc(const std::string& host_addr, int pid) = 0;
  /**
   * translate the process id to host address
   *
   * \return the host and port, "" if no such proc id 
   */
  virtual std::string GetProcHost(int proc_id) = 0;
  /**
   * Server: watch all workers in a server group,
   * will be notified when all workers have left
   */
  virtual bool WatchSGroup(int gid, int sid, rt_callback fn, void* ctx) = 0;
  /**
   * Worker: join a server group (i.e. start to read/update these servers)
   */
  virtual bool JoinSGroup(int gid, int wid, int s_group) = 0;
  /**
   * Worker: leave a server group (i.e. finish its all work)
   */
  virtual bool LeaveSGroup(int gid, int wid, int s_group) = 0;
};

/*
 * A ClusterRuntime implementation for single-process environment
 */
class SPClusterRT : public ClusterRuntime {
 public:
  ~SPClusterRT();

  bool Init() override;
  int RegistProc(const std::string& host_addr, int pid) override;
  std::string GetProcHost(int proc_id) override;
  bool WatchSGroup(int gid, int sid, rt_callback fn, void* ctx) override;
  bool JoinSGroup(int gid, int wid, int s_group) override;
  bool LeaveSGroup(int gid, int wid, int s_group) override;

 private:
  std::vector<std::string> proc_list_;
  std::map<int, std::vector<RTCallback*>> grp_callbacks_;
  std::map<int, int> grp_count_;
  std::mutex lock_;
};

}  // namespace singa

#endif  // SINGA_UTILS_CLUSTER_RT_H_
