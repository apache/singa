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

#ifndef SINGA_UTILS_ZK_SERVICE_H_
#define SINGA_UTILS_ZK_SERVICE_H_

#include <zookeeper/zookeeper.h>
#include <string>
#include <vector>

#include "singa/utils/cluster_rt.h"

namespace singa {

const int kZKBufSize = 100;
// following paths are global
const std::string kZKPathSinga = "/singa";
const std::string kZKPathSys =   "/singa/sys";
const std::string kZKPathJLock = "/singa/sys/job-lock";
const std::string kZKPathHostIdx = "/singa/sys/host-idx";
const std::string kZKPathApp =   "/singa/app";
const std::string kZKPathJob =   "/singa/app/job-";
// following paths are local under /singa/app/job-X
const std::string kZKPathJobGroup = "/group";
const std::string kZKPathJobProc =  "/proc";
const std::string kZKPathJobPLock = "/proc-lock";

inline std::string GetZKJobWorkspace(int job_id) {
  char buf[kZKBufSize];
  snprintf(buf, kZKBufSize, "%010d", job_id);
  return kZKPathJob + buf;
}

/*
 * A wrapper for zookeeper service which handles error code and reconnections
 */
class ZKService {
 public:
  static void ChildChanges(zhandle_t* zh, int type, int state,
                           const char *path, void* watcherCtx);

  ~ZKService();
  bool Init(const std::string& host, int timeout);
  bool CreateNode(const char* path, const char* val, int flag, char* output);
  bool DeleteNode(const char* path);
  bool Exist(const char* path);
  bool UpdateNode(const char* path, const char* val);
  bool GetNode(const char* path, char* output);
  bool GetChild(const char* path, std::vector<std::string>* vt);
  bool WGetChild(const char* path, std::vector<std::string>* vt,
                   RTCallback *cb);

 private:
  const int kNumRetry = 5;
  const int kSleepSec = 1;

  static void WatcherGlobal(zhandle_t* zh, int type, int state,
                            const char *path, void* watcherCtx);

  zhandle_t* zkhandle_ = nullptr;
};

/*
 * A ClusterRuntime implementation using zookeeper
 */
class ZKClusterRT : public ClusterRuntime {
 public:
  ZKClusterRT(const std::string& host, int job_id);
  ~ZKClusterRT();

  bool Init() override;
  int RegistProc(const std::string& host_addr, int pid) override;
  std::string GetProcHost(int proc_id) override;
  bool WatchSGroup(int gid, int sid, rt_callback fn, void* ctx) override;
  bool JoinSGroup(int gid, int wid, int s_group) override;
  bool LeaveSGroup(int gid, int wid, int s_group) override;

 private:
  inline std::string groupPath(int gid) {
    return group_path_ + "/sg" + std::to_string(gid);
  }
  inline std::string workerPath(int gid, int wid) {
    return "/g" + std::to_string(gid) + "_w" + std::to_string(wid);
  }

  int timeout_ = 30000;
  std::string host_ = "";
  ZKService zk_;
  std::string workspace_ = "";
  std::string group_path_ = "";
  std::string proc_path_ = "";
  std::string proc_lock_path_ = "";
  std::vector<RTCallback*> cb_vec_;
};

}  // namespace singa

#endif  // SINGA_UTILS_ZK_SERVICE_H_
